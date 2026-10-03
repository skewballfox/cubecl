// C bindings for the perf support plugin of ORC. The LLVM C API has no
// binding for it.

#include <llvm-c/Error.h>
#include <llvm-c/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>
#include <llvm/ExecutionEngine/Orc/Debugging/DebugInfoSupport.h>
#include <llvm/ExecutionEngine/Orc/Debugging/PerfSupportPlugin.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderPerf.h>
#include <llvm/Support/Error.h>

using namespace llvm;
using namespace llvm::orc;

/// Adds the perf support plugin to `jit`, as `LLVMOrcLLJITEnableDebugSupport`
/// adds the debugger plugin. The plugin writes `jit-<pid>.dump` in
/// `$JITDUMPDIR/.debug/jit` for `perf inject --jit`. It records the code of
/// each symbol, the line table when `emit_debug_info` is true, and the
/// `.eh_frame` when `emit_unwind_info` is true.
///
/// The JIT must link with JITLink. The plugin supports ELF only, and the
/// jitdump writer supports Linux only. Returns null on success, or an error
/// that the caller owns.
extern "C" LLVMErrorRef
cubecl_orc_lljit_enable_perf_support(LLVMOrcLLJITRef jit_ref,
                                     bool emit_debug_info,
                                     bool emit_unwind_info) {
  LLJIT &jit = *reinterpret_cast<LLJIT *>(jit_ref);
  auto *layer = dyn_cast<ObjectLinkingLayer>(&jit.getObjLinkingLayer());
  if (layer == nullptr) {
    return wrap(make_error<StringError>(
        "perf support requires JITLink", inconvertibleErrorCode()));
  }

  // The plugin finds the jitdump writer by a lookup of exported symbols in a
  // JITDylib. A static LLVM does not export the writer to the dynamic linker,
  // so this defines its functions from their addresses.
  auto &session = jit.getExecutionSession();
  JITDylibSP process_symbols = jit.getProcessSymbolsJITDylib();
  JITDylib &dylib =
      process_symbols ? *process_symbols : jit.getMainJITDylib();
  if (auto err = dylib.define(absoluteSymbols({
          {session.intern("llvm_orc_registerJITLoaderPerfStart"),
           ExecutorSymbolDef::fromPtr(&llvm_orc_registerJITLoaderPerfStart,
                                     JITSymbolFlags::Exported)},
          {session.intern("llvm_orc_registerJITLoaderPerfEnd"),
           ExecutorSymbolDef::fromPtr(&llvm_orc_registerJITLoaderPerfEnd,
                                     JITSymbolFlags::Exported)},
          {session.intern("llvm_orc_registerJITLoaderPerfImpl"),
           ExecutorSymbolDef::fromPtr(&llvm_orc_registerJITLoaderPerfImpl,
                                     JITSymbolFlags::Exported)},
      }))) {
    return wrap(std::move(err));
  }

  auto plugin =
      PerfSupportPlugin::Create(session.getExecutorProcessControl(), dylib,
                                emit_debug_info, emit_unwind_info);
  if (!plugin) {
    return wrap(plugin.takeError());
  }
  // JITLink drops the debug sections before the perf plugin reads the line
  // table, unless this plugin keeps them. `llvm-jitlink -perf-support` does
  // the same.
  if (emit_debug_info) {
    auto preservation = DebugInfoPreservationPlugin::Create();
    if (!preservation) {
      return wrap(preservation.takeError());
    }
    layer->addPlugin(std::move(*preservation));
  }
  layer->addPlugin(std::move(*plugin));
  return nullptr;
}
