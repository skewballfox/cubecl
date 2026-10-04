//! Kernel source locations as LLVM debug data.
//!
//! `pliron-llvm` converts the `Location` of each op to a `!dbg` location: one `DISubprogram` for
//! each function, and one for each inlined `#[cube]` function.

use crate::{prelude::*, shared::llvm_module::LlvmModule};
use cubecl_core::ir::{ContextExt, debug::DebugState, settings::DebugInfo};
use cubecl_runtime::debug_source::{kernel_source_root, source_md5s};
use pliron_llvm::{
    debug_info_conversions::to_llvm_ir::{
        DebugInfoOptions, EmissionKind, LLVMDWARFSourceLanguage, SourceText,
    },
    llvm_sys::core::{LLVMContext, LLVMModule},
    to_llvm_ir,
};

/// Converts `module` to LLVM IR, with the debug data of `level`.
///
/// At `Full`, the `DIFile` of each file embeds its source text if `embed_source` is set. NVPTX
/// must not set it: `ptxas` rejects the `.file` directive with a source text.
///
/// The compile directory is the directory that has the relative source files on this computer,
/// if one does ([`kernel_source_root`]). At `Full`, the file must have the compiled text.
pub(crate) fn convert_module(
    ctx: &Context,
    llvm_ctx: &LLVMContext,
    module: ModuleOp,
    level: DebugInfo,
    embed_source: bool,
) -> pliron::result::Result<LLVMModule> {
    if level == DebugInfo::None {
        return to_llvm_ir::convert_module(ctx, llvm_ctx, module);
    }
    let mut options = DebugInfoOptions::default();
    options.emission_kind = match level {
        DebugInfo::Full => EmissionKind::Full,
        _ => EmissionKind::LineTablesOnly,
    };
    options.language = LLVMDWARFSourceLanguage::LLVMDWARFSourceLanguageRust;
    options.producer = "cubecl".to_string();
    options.optimized = true;
    let Some(debug) = ctx.try_aux_ty::<DebugState>() else {
        return to_llvm_ir::convert_module_with_debug_info(ctx, llvm_ctx, module, options);
    };
    // Only `Full` has the source text, as the macro records it only then.
    let md5s = source_md5s(debug, level);
    if let Some(root) = kernel_source_root(debug, &md5s) {
        options.directory = root;
    }
    if embed_source && !md5s.is_empty() {
        for (path, text) in debug.sources() {
            let md5 = md5s[path.as_str()].to_string();
            options
                .source_text
                .insert(path.clone(), SourceText { text, md5 });
        }
        // LLVM writes the source text into the object only with DWARF 5.
        options.dwarf_version = 5;
    }
    to_llvm_ir::convert_module_with_debug_info(ctx, llvm_ctx, module, options)
}

/// Runs LLVM's verifier on the debug data of `module`, in a debug build of cubecl only. The test
/// suites build with debug data, so they check every kernel.
///
/// # Panics
/// When the debug data does not verify: the conversion has a bug.
#[cfg_attr(not(debug_assertions), allow(unused_variables))]
pub(crate) fn check_debug_info(module: &LlvmModule, kernel_name: &str, level: DebugInfo) {
    #[cfg(debug_assertions)]
    if level != DebugInfo::None
        && let Err(err) = module.verify()
    {
        panic!("the debug data of '{kernel_name}' does not verify: {err}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cpu::jit::engine::to_llvm_module,
        shared::{
            PlironOptions,
            base::lower_cpu,
            offline_kernels::{device_properties, scale_with_source_kernel},
        },
    };
    use cubecl_core as cubecl;
    use cubecl_core::prelude::*;
    use cubecl_runtime::kernel::CubeKernel;
    use pliron::{graph::walkers::uninterruptible::immutable::walk_op, location::Location};
    use std::sync::Arc;

    #[cube]
    fn inner(x: f32) -> f32 {
        let y = x * x;
        y / 3.0
    }

    #[cube]
    fn mid(x: f32) -> f32 {
        inner(x) * 2.0
    }

    #[cube(launch)]
    fn outer(input: &[f32], output: &mut [f32]) {
        if ABSOLUTE_POS < input.len() {
            output[ABSOLUTE_POS] = mid(input[ABSOLUTE_POS]);
        }
    }

    fn lowered(level: DebugInfo) -> super::super::base::LoweredCpu {
        let settings = KernelSettings::new(
            *CubeDim::new_1d(64),
            ExecutionMode::Checked,
            AddressType::U32,
        )
        .debug_info(level);
        let kernel = outer::Outer::new(
            settings,
            device_properties(32),
            Arc::new(TargetProperties::default()),
            BufferCompilationArg { inplace: None },
            BufferCompilationArg { inplace: None },
        );
        lower_cpu(kernel.define(), &PlironOptions::default()).unwrap()
    }

    /// The LLVM IR of `kernel` for the CPU.
    fn cpu_ir(kernel: &impl CubeKernel) -> String {
        let lowered = lower_cpu(kernel.define(), &PlironOptions::default()).unwrap();
        let module = to_llvm_module(
            &lowered.ctx,
            lowered.module,
            &lowered.kernel_name,
            lowered.debug_info,
        )
        .unwrap();
        module.verify().unwrap();
        module.print()
    }

    /// The function name, line and column of the innermost frame of `loc`.
    fn innermost(loc: &Location) -> Option<(&str, i32, i32)> {
        match loc {
            Location::CallSite { callee, .. } => innermost(callee),
            Location::Named { name, child_loc } => match child_loc.as_ref() {
                Location::SrcPos { pos, .. } => Some((name, pos.line, pos.column)),
                _ => None,
            },
            _ => None,
        }
    }

    /// Each float op and each store without a location. These ops come from the kernel source.
    /// The entry ABI adds the loops over the units, which have no source and get line 0 in the
    /// DWARF.
    fn unlocated_source_ops(lowered: &super::super::base::LoweredCpu) -> Vec<String> {
        let mut ops = Vec::new();
        walk_op(
            &lowered.ctx,
            &mut ops,
            &WALKCONFIG_PREORDER_FORWARD,
            lowered.module.get_operation(),
            |ctx, ops, node| {
                let IRNode::Operation(op) = node else {
                    return;
                };
                let name = Operation::get_opid(op, ctx).to_string();
                let from_source = name == "llvm.store"
                    || name.starts_with("llvm.call")
                    || (name.starts_with("llvm.f")
                        && !["llvm.func", "llvm.fence"].contains(&name.as_str()));
                if from_source && op.deref(ctx).loc().is_unknown() {
                    ops.push(name);
                }
            },
        );
        ops
    }

    /// The rewrites and the fallback pass keep a location on each op that comes from source.
    #[test]
    fn lowering_keeps_source_locations() {
        let lowered = lowered(DebugInfo::LineTables);
        let unlocated = unlocated_source_ops(&lowered);
        assert!(
            unlocated.is_empty(),
            "ops without a location: {unlocated:?}"
        );

        // `x * x` and `y / 3.0` in `inner` keep their own positions. The fallback pass alone would
        // give both the position of the op before them.
        let mut ops = Vec::new();
        walk_op(
            &lowered.ctx,
            &mut ops,
            &WALKCONFIG_PREORDER_FORWARD,
            lowered.module.get_operation(),
            |_ctx, ops, node| {
                if let IRNode::Operation(op) = node {
                    ops.push(op);
                }
            },
        );
        let mut positions = ops
            .into_iter()
            .map(|op| op.deref(&lowered.ctx).loc())
            .filter_map(|loc| {
                let (name, line, column) = innermost(&loc)?;
                (name == "inner").then_some((line, column))
            })
            .collect::<Vec<_>>();
        positions.sort();
        positions.dedup();
        assert!(positions.len() >= 2, "positions in `inner`: {positions:?}");
    }

    #[test]
    fn inlined_functions_are_dwarf_frames() {
        let lowered = lowered(DebugInfo::LineTables);
        let module = to_llvm_module(
            &lowered.ctx,
            lowered.module,
            &lowered.kernel_name,
            lowered.debug_info,
        )
        .unwrap();
        module.verify().unwrap();
        let ir = module.print();
        for name in ["outer", "mid", "inner"] {
            assert!(
                ir.contains(&format!("DISubprogram(name: \"{name}\"")),
                "{ir}"
            );
        }
        assert!(ir.contains("inlinedAt:"), "{ir}");
        assert!(ir.contains("emissionKind: LineTablesOnly"), "{ir}");
    }

    /// At `Full`, the `DIFile` of the kernel file embeds its text and MD5, in DWARF 5.
    #[test]
    fn full_debug_info_embeds_the_source_text() {
        let ir = cpu_ir(&scale_with_source_kernel());
        let file = ir
            .lines()
            .find(|line| line.contains("DIFile(") && line.contains("offline_kernels.rs"))
            .unwrap_or_else(|| panic!("no `DIFile` of the kernel file:\n{ir}"));
        assert!(file.contains("source: \""), "{file}");
        assert!(file.contains("checksumkind: CSK_MD5"), "{file}");
        assert!(ir.contains("!\"Dwarf Version\", i32 5"), "{ir}");
    }

    /// Without the text from the macro, `LineTables` embeds no source and keeps DWARF 4.
    #[test]
    fn line_tables_embed_no_source_text() {
        let lowered = lowered(DebugInfo::LineTables);
        let module = to_llvm_module(
            &lowered.ctx,
            lowered.module,
            &lowered.kernel_name,
            lowered.debug_info,
        )
        .unwrap();
        let ir = module.print();
        assert!(!ir.contains("source: \""), "{ir}");
        assert!(ir.contains("!\"Dwarf Version\", i32 4"), "{ir}");
    }

    /// The directory of a `DIFile` line of `ir`.
    fn file_directory<'a>(ir: &'a str, file: &str) -> &'a str {
        let line = ir
            .lines()
            .find(|line| line.contains("DIFile(") && line.contains(file))
            .unwrap_or_else(|| panic!("no `DIFile` of {file}:\n{ir}"));
        line.split_once("directory: \"")
            .and_then(|(_, rest)| rest.split_once('"'))
            .map_or("", |(directory, _)| directory)
    }

    /// The tests run in the crate directory, and the workspace root, which `file!()` is relative
    /// to, is a parent. At both levels, the compile directory is that root.
    #[test]
    fn the_compile_directory_has_the_source_files() {
        let full = cpu_ir(&scale_with_source_kernel());
        let lowered = lowered(DebugInfo::LineTables);
        let line_tables = to_llvm_module(
            &lowered.ctx,
            lowered.module,
            &lowered.kernel_name,
            lowered.debug_info,
        )
        .unwrap()
        .print();
        for (ir, file) in [
            (&full, "crates/cubecl-llvm/src/shared/offline_kernels.rs"),
            (&line_tables, "crates/cubecl-llvm/src/shared/debug_info.rs"),
        ] {
            let directory = file_directory(ir, file);
            assert!(
                std::path::Path::new(directory).join(file).is_file(),
                "`{directory}` does not have {file}"
            );
        }
    }

    /// The ops have locations in a `dev` build. The level `None` must still give no debug data.
    #[test]
    fn no_debug_data_without_debug_info() {
        let lowered = lowered(DebugInfo::None);
        let module = to_llvm_module(
            &lowered.ctx,
            lowered.module,
            &lowered.kernel_name,
            DebugInfo::None,
        )
        .unwrap();
        let ir = module.print();
        assert!(
            !ir.contains("!dbg") && !ir.contains("DICompileUnit"),
            "{ir}"
        );
    }
}
