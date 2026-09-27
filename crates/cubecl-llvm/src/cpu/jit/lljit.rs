//! The LLJIT of the CPU target. cubecl owns it, not `pliron-llvm`, so that it can hook the
//! object layers: the object transform reads the symbol sizes for the perf map, and the JIT event
//! listeners give the object code and its DWARF to gdb and to `perf`.

use super::symbols::{JitSymbols, SymbolSizes};
use crate::shared::llvm_module::{LlvmModule, error_message};
use llvm_sys::{
    core::LLVMDisposeMessage,
    error::LLVMErrorRef,
    execution_engine::{LLVMCreateGDBRegistrationListener, LLVMCreatePerfJITEventListener},
    object::{
        LLVMCreateBinary, LLVMDisposeBinary, LLVMDisposeSymbolIterator, LLVMGetSymbolName,
        LLVMGetSymbolSize, LLVMMoveToNextSymbol, LLVMObjectFileCopySymbolIterator,
        LLVMObjectFileIsSymbolIteratorAtEnd,
    },
    orc2::{
        LLVMOrcCreateNewThreadSafeContextFromLLVMContext, LLVMOrcCreateNewThreadSafeModule,
        LLVMOrcDisposeThreadSafeContext, LLVMOrcDisposeThreadSafeModule,
        LLVMOrcExecutionSessionRef, LLVMOrcObjectLayerRef, LLVMOrcObjectTransformLayerSetTransform,
        ee::{
            LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManagerReserveAlloc,
            LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener,
        },
        lljit::{
            LLVMOrcCreateLLJIT, LLVMOrcCreateLLJITBuilder, LLVMOrcDisposeLLJIT,
            LLVMOrcLLJITAddLLVMIRModule, LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator,
            LLVMOrcLLJITGetMainJITDylib, LLVMOrcLLJITGetObjTransformLayer, LLVMOrcLLJITLookup,
            LLVMOrcLLJITRef,
        },
    },
    prelude::LLVMMemoryBufferRef,
};
use std::{
    ffi::{CStr, CString, c_char, c_void},
    sync::Once,
};

/// An LLJIT that owns the modules added to it.
pub(crate) struct Jit {
    jit: LLVMOrcLLJITRef,
    /// The object transform writes here. It must live as long as the JIT.
    sizes: Option<Box<SymbolSizes>>,
}

impl Jit {
    /// A JIT for kernels. With `listeners`, the object code and its DWARF go to gdb, and to the
    /// perf jitdump if `symbols` asks for it. Without, the JIT has the default settings of LLVM.
    ///
    /// # Errors
    /// The message LLVM gives, when it cannot create the JIT for the host.
    pub(crate) fn new(symbols: JitSymbols, listeners: bool) -> Result<Self, String> {
        let mut jit = std::ptr::null_mut();
        // SAFETY: a null builder asks for the default settings. `LLVMOrcCreateLLJIT` takes the
        // builder. The creator reads `jitdump` only as a flag, never as a pointer.
        unsafe {
            let builder = if listeners {
                let builder = LLVMOrcCreateLLJITBuilder();
                let jitdump = symbols.jitdump as usize as *mut c_void;
                LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator(
                    builder,
                    create_listened_layer,
                    jitdump,
                );
                builder
            } else {
                std::ptr::null_mut()
            };
            error_message(LLVMOrcCreateLLJIT(&mut jit, builder))?;
        }

        let sizes = symbols.perf_map.then(|| {
            let sizes = Box::<SymbolSizes>::default();
            // SAFETY: the JIT is live, and `sizes` lives as long as the JIT (see `Drop`).
            unsafe {
                LLVMOrcObjectTransformLayerSetTransform(
                    LLVMOrcLLJITGetObjTransformLayer(jit),
                    record_symbol_sizes,
                    &*sizes as *const SymbolSizes as *mut c_void,
                );
            }
            sizes
        });

        Ok(Self { jit, sizes })
    }

    /// Adds `module` to the main library of the JIT.
    ///
    /// # Errors
    /// The message LLVM gives, when the JIT does not accept the module.
    pub(crate) fn add_module(&self, module: LlvmModule) -> Result<(), String> {
        let (ctx, module) = module.into_raw();
        // SAFETY: the thread-safe context takes the context, and the thread-safe module takes
        // the module. The JIT takes the thread-safe module only if the call succeeds.
        unsafe {
            let ts_ctx = LLVMOrcCreateNewThreadSafeContextFromLLVMContext(ctx);
            let ts_module = LLVMOrcCreateNewThreadSafeModule(module, ts_ctx);
            LLVMOrcDisposeThreadSafeContext(ts_ctx);
            let dylib = LLVMOrcLLJITGetMainJITDylib(self.jit);
            let result = error_message(LLVMOrcLLJITAddLLVMIRModule(self.jit, dylib, ts_module));
            if result.is_err() {
                LLVMOrcDisposeThreadSafeModule(ts_module);
            }
            result
        }
    }

    /// The address of the symbol `name`. The first lookup compiles the module.
    ///
    /// # Errors
    /// The message LLVM gives, when the symbol is not defined or does not compile.
    pub(crate) fn lookup(&self, name: &str) -> Result<u64, String> {
        let c_name = CString::new(name).map_err(|_| format!("symbol '{name}' contains a NUL"))?;
        let mut addr = 0;
        // SAFETY: the JIT is live, and `c_name` is a C string.
        error_message(unsafe { LLVMOrcLLJITLookup(self.jit, &mut addr, c_name.as_ptr()) })?;
        Ok(addr)
    }

    /// The size of the symbol `name` in the object code, if the perf map asked for it and the
    /// module is compiled.
    pub(crate) fn symbol_size(&self, name: &str) -> Option<u64> {
        self.sizes.as_ref()?.get(name)
    }
}

impl Drop for Jit {
    fn drop(&mut self) {
        // SAFETY: the JIT is owned by `self`. It is disposed before `sizes`.
        if let Err(err) = error_message(unsafe { LLVMOrcDisposeLLJIT(self.jit) }) {
            log::warn!("Can't dispose the kernel JIT: {err}");
        }
    }
}

/// The object layer of a JIT with listeners: `RuntimeDyld`, because the JIT event listeners work only
/// with it. It reserves one block of memory for each object, so the code and its constants stay
/// within the reach of 32-bit relocations. `jitdump` is non-null to add the perf listener.
extern "C" fn create_listened_layer(
    jitdump: *mut c_void,
    session: LLVMOrcExecutionSessionRef,
    _triple: *const c_char,
) -> LLVMOrcObjectLayerRef {
    // SAFETY: the session is live, and the layer takes no ownership of the listeners, which are
    // process-wide singletons in LLVM.
    unsafe {
        let layer =
            LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManagerReserveAlloc(session, 1);
        LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener(
            layer,
            LLVMCreateGDBRegistrationListener(),
        );
        if !jitdump.is_null() {
            let perf = LLVMCreatePerfJITEventListener();
            if perf.is_null() {
                static WARN: Once = Once::new();
                WARN.call_once(|| {
                    log::warn!(
                        "This LLVM has no perf JIT listener (LLVM_USE_PERF), so no jitdump is \
                         written. The perf map still names each kernel."
                    )
                });
            } else {
                LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener(layer, perf);
            }
        }
        layer
    }
}

/// The object transform: it records the size of each symbol and returns the object unchanged.
extern "C" fn record_symbol_sizes(
    ctx: *mut c_void,
    object: *mut LLVMMemoryBufferRef,
) -> LLVMErrorRef {
    // SAFETY: `ctx` is the `SymbolSizes` of the JIT (see `Jit::new`), and `object` holds a live
    // buffer. The binary reads the buffer and does not take it.
    unsafe {
        let sizes = &*(ctx as *const SymbolSizes);
        let mut message = std::ptr::null_mut();
        let binary = LLVMCreateBinary(*object, std::ptr::null_mut(), &mut message);
        if binary.is_null() {
            if !message.is_null() {
                LLVMDisposeMessage(message);
            }
            return std::ptr::null_mut();
        }
        let symbols = LLVMObjectFileCopySymbolIterator(binary);
        while LLVMObjectFileIsSymbolIteratorAtEnd(binary, symbols) == 0 {
            let size = LLVMGetSymbolSize(symbols);
            if size > 0 {
                let name = CStr::from_ptr(LLVMGetSymbolName(symbols));
                sizes.insert(name.to_string_lossy().into_owned(), size);
            }
            LLVMMoveToNextSymbol(symbols);
        }
        LLVMDisposeSymbolIterator(symbols);
        LLVMDisposeBinary(binary);
    }
    std::ptr::null_mut()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With the jitdump asked for, the perf listener writes `jit-<pid>.dump` under `$JITDUMPDIR`.
    /// Skipped when this LLVM was built without `LLVM_USE_PERF`.
    #[test]
    fn the_jitdump_is_written() {
        // SAFETY: LLVM gives a process-wide listener, or null.
        if unsafe { LLVMCreatePerfJITEventListener() }.is_null() {
            eprintln!("skipped: this LLVM has no perf JIT listener");
            return;
        }
        let dir = std::env::temp_dir().join(format!("cubecl-jitdump-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // SAFETY: no other test of this binary reads `JITDUMPDIR`.
        unsafe { std::env::set_var("JITDUMPDIR", &dir) };
        pliron_llvm::llvm_sys::target::initialize_native().unwrap();

        let symbols = JitSymbols {
            perf_map: false,
            jitdump: true,
        };
        let jit = Jit::new(symbols, true).unwrap();
        jit.add_module(LlvmModule::new("define i32 @f() {\n  ret i32 1\n}\n").unwrap())
            .unwrap();
        jit.lookup("f").unwrap();

        let name = format!("jit-{}.dump", std::process::id());
        let found = walk(&dir).iter().any(|path| path.ends_with(&name));
        std::fs::remove_dir_all(&dir).ok();
        assert!(found, "no {name} under {}", dir.display());
    }

    fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Vec::new();
        };
        entries
            .flatten()
            .flat_map(|entry| {
                let path = entry.path();
                if path.is_dir() {
                    walk(&path)
                } else {
                    vec![path]
                }
            })
            .collect()
    }
}
