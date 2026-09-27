//! `OpLine` for the ops of a kernel with debug data.
//!
//! An `OpLine` gives its line to the instructions after it, up to the next `OpLine` or the end of
//! the block. Core SPIR-V has no inlined frames, so each `OpLine` has the innermost frame of the
//! location of its op.

use cubecl_ir::{debug::leaf_line, rewrite::visit_all_ops_of_type_mut};
use pliron::{
    basic_block::BasicBlock,
    builtin::ops::FuncOp,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    location::Located,
    op::Op,
    operation::Operation,
};
use pliron_spirv::ops::LineOp;

/// Inserts a `LineOp` before each op of the functions in `module` whose source line is not the
/// line of the op before it. The column is 0, as `#line` in C++ gives no column.
pub(crate) fn insert_line_ops(ctx: &mut Context, module: Ptr<Operation>) {
    let mut funcs = Vec::new();
    visit_all_ops_of_type_mut::<FuncOp, _>(ctx, &mut funcs, module, |_ctx, funcs, func| {
        funcs.push(func.get_operation());
    });
    for func in funcs {
        insert_in_regions(ctx, func);
    }
}

/// Inserts the `LineOp`s in each block of the regions of `op`.
fn insert_in_regions(ctx: &mut Context, op: Ptr<Operation>) {
    let regions = op.deref(ctx).regions().collect::<Vec<_>>();
    let blocks = regions
        .into_iter()
        .flat_map(|region| region.deref(ctx).iter(ctx).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    for block in blocks {
        insert_in_block(ctx, block);
    }
}

fn insert_in_block(ctx: &mut Context, block: Ptr<BasicBlock>) {
    // An instruction between a merge instruction and its branch is not valid. The merge comes
    // just before the terminator, so no `OpLine` goes before a terminator.
    let terminator = block.deref(ctx).get_terminator(ctx);
    let mut current = None;
    let ops = block.deref(ctx).iter(ctx).collect::<Vec<_>>();
    for op in ops {
        let loc = op.deref(ctx).loc();
        let nested = op.deref(ctx).num_regions() > 0;
        if Some(op) != terminator
            && let Some(line) = leaf_line(ctx, &loc)
            && current.as_ref() != Some(&line)
        {
            let (file, number) = line.clone();
            LineOp::new(ctx, file, number, 0u32)
                .get_operation()
                .insert_before(ctx, op);
            current = Some(line);
        }
        if nested {
            insert_in_regions(ctx, op);
            // The instructions after a structured op are in the merge block, a new block.
            current = None;
        }
    }
}

#[cfg(test)]
// A `#[cube]` kernel loops over a range, not over an iterator.
#[allow(clippy::needless_range_loop)]
mod tests {
    use crate::SpirvCompiler;
    use cubecl_core as cubecl;
    use cubecl_core::{
        Compiler, VulkanCompilationOptions, WgpuCompilationOptions,
        ir::{
            DeviceIdentity, DeviceProperties, ElemType, FloatKind, HardwareProperties,
            MemoryDeviceProperties, UIntKind,
            features::{Features, TypeUsage},
            settings::DebugInfo,
        },
        prelude::*,
    };
    use cubecl_runtime::kernel::CubeKernel;
    use rspirv::binary::Disassemble;
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

    /// The branches and the loop give selection merges and a loop merge, which no `OpLine` may
    /// separate from their branches.
    #[cube(launch)]
    fn outer(input: &[f32], output: &mut [f32]) {
        if ABSOLUTE_POS < input.len() {
            let mut acc = 0.0;
            for i in 0..input.len() {
                if i != ABSOLUTE_POS {
                    acc += mid(input[i]);
                }
            }
            output[ABSOLUTE_POS] = acc;
        }
    }

    fn properties() -> Arc<DeviceProperties> {
        let hardware = HardwareProperties {
            load_width: 128,
            plane_size_min: 32,
            plane_size_max: 32,
            max_bindings: 32,
            max_shared_memory_size: 65536,
            max_cube_count: (u32::MAX, u16::MAX as u32, u16::MAX as u32),
            max_units_per_cube: 1024,
            max_cube_dim: (1024, 1024, 1024),
            num_streaming_multiprocessors: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            num_cpu_cores: None,
            last_level_cache_size: None,
            max_vector_size: VectorSize::MAX,
            cube_mma_reserved_shared_memory: 0,
        };
        let mut properties = DeviceProperties::new(
            Features::default(),
            MemoryDeviceProperties::new(u64::MAX, 256),
            hardware,
            cubecl_core::profile::TimingMethod::Device,
            DeviceIdentity {
                name: "offline".to_string(),
                fingerprint: "offline".to_string(),
                physical: None,
            },
        );
        properties.register_address_type(AddressType::U32);
        for ty in [
            ElemType::Index,
            ElemType::UInt(UIntKind::U32),
            ElemType::Float(FloatKind::F32),
            ElemType::Bool,
        ] {
            properties.register_type_usage(ty, TypeUsage::all());
        }
        Arc::new(properties)
    }

    /// The disassembled SPIR-V of `outer` at the debug level `level`. The level is set after the
    /// resolution, which gives a `dev` build at least line tables, as `CUBECL_DEBUG_INFO` does.
    fn disassembly(level: DebugInfo) -> String {
        let settings = KernelSettings::new(
            *CubeDim::new_1d(64),
            ExecutionMode::Checked,
            AddressType::U32,
        );
        let kernel = outer::Outer::new(
            settings,
            properties(),
            Arc::new(TargetProperties::default()),
            BufferCompilationArg { inplace: None },
            BufferCompilationArg { inplace: None },
        );
        let mut definition = kernel.define();
        definition.settings.debug_info = level;
        let options = WgpuCompilationOptions {
            supports_u64: true,
            supports_vulkan_compiler: true,
            supports_msl_compiler: false,
            vulkan: VulkanCompilationOptions {
                max_spirv_version: (1, 6),
                max_vector_size: 4,
                ..Default::default()
            },
        };
        let kernel = SpirvCompiler.compile(definition, &options).unwrap();
        kernel.module.unwrap().disassemble()
    }

    /// Each op gets the line of its innermost `#[cube]` frame, so the lines of `inner`, `mid` and
    /// `outer` are in the module.
    #[test]
    fn source_lines_are_op_lines() {
        let module = disassembly(DebugInfo::LineTables);
        let mut lines = module
            .lines()
            .filter_map(|line| line.trim().strip_prefix("OpLine "))
            .map(|rest| {
                rest.split_whitespace()
                    .nth(1)
                    .unwrap()
                    .parse::<u32>()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        lines.sort();
        lines.dedup();
        assert!(lines.len() >= 4, "lines {lines:?} in:\n{module}");
        assert!(module.contains("lines.rs\""), "{module}");
    }

    /// Without debug data, the module has no `OpLine`, although the ops have locations.
    #[test]
    fn no_op_lines_without_debug_data() {
        let module = disassembly(DebugInfo::None);
        assert!(!module.contains("OpLine"), "{module}");
    }
}
