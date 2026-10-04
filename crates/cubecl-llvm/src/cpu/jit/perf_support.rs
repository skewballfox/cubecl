//! The perf support plugin of ORC, through the C++ shim `cpu/cpp_shims/perf_support.cpp`.
//!
//! The LLVM C API has no binding for the plugin, and `pliron-llvm` does not add one. This module
//! is the only user of the shim.

use crate::shared::llvm_module::error_message;
use llvm_sys::{error::LLVMErrorRef, orc2::lljit::LLVMOrcLLJITRef};

/// Adds the perf support plugin to `jit`. The plugin writes `jit-<pid>.dump` in
/// `$JITDUMPDIR/.debug/jit`, with the line table (`debug_info`) and the unwind data
/// (`unwind_info`) of each symbol.
///
/// # Errors
/// The message LLVM gives, when the JIT does not link with `JITLink`, the object format is not ELF,
/// or the jitdump cannot be opened (for example, not on Linux).
///
/// # Safety
/// `jit` must be a live LLJIT. Call it before the JIT links an object.
pub(crate) unsafe fn enable_perf_support(
    jit: LLVMOrcLLJITRef,
    debug_info: bool,
    unwind_info: bool,
) -> Result<(), String> {
    // SAFETY: the caller's contract. The shim returns an error that `error_message` takes.
    error_message(unsafe { cubecl_orc_lljit_enable_perf_support(jit, debug_info, unwind_info) })
}

unsafe extern "C" {
    /// Returns null on success, or an error that the caller owns.
    fn cubecl_orc_lljit_enable_perf_support(
        jit: LLVMOrcLLJITRef,
        emit_debug_info: bool,
        emit_unwind_info: bool,
    ) -> LLVMErrorRef;
}
