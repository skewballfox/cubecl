//! The LLJIT of the CPU target. cubecl owns it, not `pliron-llvm`, so that it can hook the
//! object layer: the object transform reads the symbol sizes for the perf map.

use super::symbols::{JitSymbols, SymbolSizes};
use crate::shared::llvm_module::{LlvmModule, error_message};
use llvm_sys::{
    core::LLVMDisposeMessage,
    error::LLVMErrorRef,
    object::{
        LLVMCreateBinary, LLVMDisposeBinary, LLVMDisposeSymbolIterator, LLVMGetSymbolName,
        LLVMGetSymbolSize, LLVMMoveToNextSymbol, LLVMObjectFileCopySymbolIterator,
        LLVMObjectFileIsSymbolIteratorAtEnd,
    },
    orc2::{
        LLVMOrcCreateNewThreadSafeContextFromLLVMContext, LLVMOrcCreateNewThreadSafeModule,
        LLVMOrcDisposeThreadSafeContext, LLVMOrcDisposeThreadSafeModule,
        LLVMOrcObjectTransformLayerSetTransform,
        lljit::{
            LLVMOrcCreateLLJIT, LLVMOrcDisposeLLJIT, LLVMOrcLLJITAddLLVMIRModule,
            LLVMOrcLLJITGetMainJITDylib, LLVMOrcLLJITGetObjTransformLayer, LLVMOrcLLJITLookup,
            LLVMOrcLLJITRef,
        },
    },
    prelude::LLVMMemoryBufferRef,
};
use std::ffi::{CStr, CString, c_void};

/// An LLJIT that owns the modules added to it.
pub(crate) struct Jit {
    jit: LLVMOrcLLJITRef,
    /// The object transform writes here. It must live as long as the JIT.
    sizes: Option<Box<SymbolSizes>>,
}

impl Jit {
    /// # Errors
    /// The message LLVM gives, when it cannot create the JIT for the host.
    pub(crate) fn new(symbols: JitSymbols) -> Result<Self, String> {
        let mut jit = std::ptr::null_mut();
        // SAFETY: a null builder asks for the default settings.
        error_message(unsafe { LLVMOrcCreateLLJIT(&mut jit, std::ptr::null_mut()) })?;

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
