//! Debug data in the SPIR-V of a kernel.
//!
//! `pliron-spirv` converts the location of each op to debug data. The format is core `OpLine`, or
//! `NonSemantic.Shader.DebugInfo.100` with each inlined `#[cube]` function as a separate frame.
//! [`debug_format`] selects the format from the configuration and the device. `pliron-spirv` owns the placement rules, for example no instruction between a merge
//! instruction and its branch.

use cubecl_core::WgpuCompilationOptions;
use cubecl_ir::{ContextExt, debug::DebugState, settings::DebugInfo};
use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig, compilation::SpirvDebugFormat};
use pliron::context::Context;
use pliron_spirv::{
    PlironBuilder,
    debug_info::{DebugInfoFormat, DebugInfoOptions},
};
use rspirv::spirv::SourceLanguage;

/// The builder of the module of a kernel with the debug data `level`. At [`DebugInfo::None`], the
/// module has no debug data.
pub(crate) fn builder(ctx: &Context, level: DebugInfo) -> PlironBuilder {
    if level == DebugInfo::None {
        return PlironBuilder::default();
    }
    let supported = ctx
        .aux_ty::<WgpuCompilationOptions>()
        .vulkan
        .supports_non_semantic_info;
    let configured = CubeClRuntimeConfig::get().compilation.spirv_debug_format;
    let mut options = DebugInfoOptions::default();
    options.format = debug_format(configured, supported);
    options.language = SourceLanguage::Rust;
    options.producer = "cubecl".to_string();
    if let Some(debug) = ctx.try_aux_ty::<DebugState>() {
        // The directory that has the relative source files on this computer, if one does. At
        // `Full`, the file must have the compiled text.
        #[cfg(feature = "std")]
        {
            use cubecl_runtime::debug_source::{kernel_source_root, source_md5s};
            if let Some(root) = kernel_source_root(debug, &source_md5s(debug, level)) {
                options.directory = root;
            }
        }
        // Only `Full` embeds the source text, as the macro records it only then.
        if level == DebugInfo::Full {
            options.source_text = debug
                .sources()
                .iter()
                .map(|(path, text)| (path.clone(), text.to_string()))
                .collect();
        }
    }
    PlironBuilder::with_debug_info(options)
}

/// The format for the configured format `configured`, on a device that `supported`
/// `NonSemantic.Shader.DebugInfo.100` or not. A device without support gets `OpLine`.
fn debug_format(configured: SpirvDebugFormat, supported: bool) -> DebugInfoFormat {
    match configured {
        SpirvDebugFormat::Auto | SpirvDebugFormat::NonSemantic if supported => {
            DebugInfoFormat::NonSemantic
        }
        SpirvDebugFormat::Auto | SpirvDebugFormat::OpLine => DebugInfoFormat::OpLine,
        SpirvDebugFormat::NonSemantic => {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                log::warn!(
                    "The device does not support NonSemantic.Shader.DebugInfo.100. \
                     SPIR-V kernels get OpLine debug data."
                );
            });
            DebugInfoFormat::OpLine
        }
    }
}

#[cfg(test)]
// A `#[cube]` kernel loops over a range, not over an iterator.
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::debug_format;
    use crate::SpirvCompiler;
    use cubecl_core as cubecl;
    use cubecl_core::{
        Compiler, VulkanCompilationOptions, WgpuCompilationOptions,
        ir::{
            DeviceProperties, ElemType, FloatKind, UIntKind, features::TypeUsage,
            settings::DebugInfo,
        },
        prelude::*,
        runtime_tests::offline::offline_device_properties,
    };
    use cubecl_runtime::config::compilation::SpirvDebugFormat;
    use cubecl_runtime::kernel::CubeKernel;
    use pliron_spirv::debug_info::DebugInfoFormat;
    use rspirv::{
        binary::Disassemble,
        dr::{Instruction, Module, Operand},
        spirv::{DebugInfoOp, Op as SpirvOp},
    };
    use std::{
        io::{ErrorKind, Write},
        process::{Command, Stdio},
        sync::Arc,
    };

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
        let mut properties = offline_device_properties(32);
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

    /// Compiles `kernel` at the debug level `level`, on a device that supports
    /// `NonSemantic.Shader.DebugInfo.100` or not. The level is set after the resolution, which
    /// gives a `dev` build at least line tables, as `CUBECL_DEBUG_INFO` does.
    fn compile(kernel: impl CubeKernel, level: DebugInfo, non_semantic: bool) -> Module {
        let mut definition = kernel.define();
        definition.settings.debug_info = level;
        let options = WgpuCompilationOptions {
            supports_u64: true,
            supports_vulkan_compiler: true,
            supports_msl_compiler: false,
            vulkan: VulkanCompilationOptions {
                max_spirv_version: (1, 6),
                max_vector_size: 4,
                supports_non_semantic_info: non_semantic,
                ..Default::default()
            },
        };
        let kernel = SpirvCompiler.compile(definition, &options).unwrap();
        let module = Arc::unwrap_or_clone(kernel.module.unwrap());
        validate(&kernel.assembled_module);
        module
    }

    fn settings() -> KernelSettings {
        KernelSettings::new(
            *CubeDim::new_1d(64),
            ExecutionMode::Checked,
            AddressType::U32,
        )
    }

    /// The SPIR-V of `outer`.
    fn compile_outer(level: DebugInfo, non_semantic: bool) -> Module {
        let kernel = outer::Outer::new(
            settings(),
            properties(),
            Arc::new(TargetProperties::default()),
            BufferCompilationArg { inplace: None },
            BufferCompilationArg { inplace: None },
        );
        compile(kernel, level, non_semantic)
    }

    /// The disassembled SPIR-V of `outer`.
    fn disassembly(level: DebugInfo, non_semantic: bool) -> String {
        compile_outer(level, non_semantic).disassemble()
    }

    /// Runs `spirv-val` on `words`. Does nothing if `spirv-val` is not installed.
    fn validate(words: &[u32]) {
        let mut child = match Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3", "-"])
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(err) if err.kind() == ErrorKind::NotFound => return,
            Err(err) => panic!("spirv-val: {err}"),
        };
        let bytes = words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>();
        child.stdin.take().unwrap().write_all(&bytes).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "spirv-val: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    /// Without device support, each op gets the `OpLine` of its innermost `#[cube]` frame, so the
    /// lines of `inner`, `mid` and `outer` are in the module.
    #[test]
    fn source_lines_are_op_lines() {
        let module = disassembly(DebugInfo::LineTables, false);
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
        assert!(module.contains("debug_info.rs\""), "{module}");
        assert!(!module.contains("NonSemantic"), "{module}");
    }

    /// The file name of the debug data is the absolute path of this file: the search finds the
    /// workspace root, a parent of the working directory of the test.
    #[test]
    fn the_file_name_is_the_absolute_path() {
        let module = compile_outer(DebugInfo::LineTables, false);
        let file = module
            .debug_string_source
            .iter()
            .filter_map(|inst| match inst.operands.first() {
                Some(Operand::LiteralString(text)) => Some(text.as_str()),
                _ => None,
            })
            .find(|text| text.ends_with("debug_info.rs"))
            .expect("the file name of this file");
        assert!(std::path::Path::new(file).is_absolute(), "{file}");
        assert!(std::path::Path::new(file).is_file(), "{file}");
    }

    /// The extended instructions `op` of `NonSemantic.Shader.DebugInfo.100` in `module`.
    fn debug_instructions(module: &Module, op: DebugInfoOp) -> Vec<&Instruction> {
        let functions = module
            .functions
            .iter()
            .flat_map(|func| &func.blocks)
            .flat_map(|block| &block.instructions);
        module
            .types_global_values
            .iter()
            .chain(functions)
            .filter(|inst| {
                inst.class.opcode == SpirvOp::ExtInst
                    && inst.operands[1] == Operand::LiteralExtInstInteger(op as u32)
            })
            .collect()
    }

    /// The text of the `OpString` operand `index` of the extended instruction `inst`.
    fn string_operand(module: &Module, inst: &Instruction, index: usize) -> String {
        let Some(Operand::IdRef(id)) = inst.operands.get(index + 2) else {
            panic!("operand {index} of {inst:?} is not an id");
        };
        let string = module
            .debug_string_source
            .iter()
            .find(|it| it.result_id == Some(*id))
            .unwrap();
        match &string.operands[0] {
            Operand::LiteralString(text) => text.clone(),
            operand => panic!("%{id} is not a string: {operand:?}"),
        }
    }

    /// With device support, `inner` and `mid` are separate frames, inlined in `outer`.
    #[test]
    fn non_semantic_has_inlined_frames() {
        let module = compile_outer(DebugInfo::LineTables, true);
        let mut names = debug_instructions(&module, DebugInfoOp::DebugFunction)
            .into_iter()
            .map(|inst| string_operand(&module, inst, 0))
            .collect::<Vec<_>>();
        names.sort();
        assert_eq!(names, ["inner", "mid", "outer"]);
        assert!(!debug_instructions(&module, DebugInfoOp::DebugInlinedAt).is_empty());
        assert!(!module.disassemble().contains("OpLine"));
    }

    /// Without debug data, the module has no debug instruction, although the ops have locations.
    #[test]
    fn no_debug_data_without_level() {
        for non_semantic in [false, true] {
            let module = disassembly(DebugInfo::None, non_semantic);
            assert!(!module.contains("OpLine"), "{module}");
            assert!(!module.contains("NonSemantic"), "{module}");
        }
    }

    #[test]
    fn format_follows_configuration_and_device() {
        use SpirvDebugFormat::*;
        let cases = [
            (Auto, true, DebugInfoFormat::NonSemantic),
            (Auto, false, DebugInfoFormat::OpLine),
            (OpLine, true, DebugInfoFormat::OpLine),
            (NonSemantic, true, DebugInfoFormat::NonSemantic),
            (NonSemantic, false, DebugInfoFormat::OpLine),
        ];
        for (configured, supported, expected) in cases {
            assert_eq!(debug_format(configured, supported), expected);
        }
    }

    /// `debug_symbols` embeds the text of this file. `spirv-val` checks each column against the
    /// length of its line in the text, so the columns of the macro must fit the lines.
    #[cube(launch, debug_symbols)]
    fn outer_full(input: &[f32], output: &mut [f32]) {
        if ABSOLUTE_POS < input.len() {
            output[ABSOLUTE_POS] = mid(input[ABSOLUTE_POS]);
        }
    }

    #[test]
    fn full_debug_data_embeds_the_source() {
        let kernel = outer_full::OuterFull::new(
            settings(),
            properties(),
            Arc::new(TargetProperties::default()),
            BufferCompilationArg { inplace: None },
            BufferCompilationArg { inplace: None },
        );
        let module = compile(kernel, DebugInfo::Full, true);
        let texts = debug_instructions(&module, DebugInfoOp::DebugSource)
            .into_iter()
            .filter(|inst| inst.operands.len() > 3)
            .map(|inst| string_operand(&module, inst, 1))
            .collect::<Vec<_>>();
        assert_eq!(texts.len(), 1, "{}", module.disassemble());
        assert!(texts[0].starts_with("//! Debug data in the SPIR-V of a kernel."));
    }
}
