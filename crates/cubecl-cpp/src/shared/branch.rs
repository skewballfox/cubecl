use core::fmt::Write;
use cubecl_core::ir::{
    ContextExt,
    debug::leaf_line,
    dialect::{branch::*, general::SelectOp},
    prelude::*,
};
use pliron::{basic_block::BasicBlock, linked_list::ContainsLinkedList, std_deps::path::PathBuf};

use crate::{
    error::EmissionErrors,
    shared::{
        CppValue, OpExtCPP, scoped_block, shared_op, shared_op_with_out, ty::TypeExtCPP,
        unroll::unrolling,
    },
};

/// Marks a kernel with debug data: [`block_to_cpp`] then gives each op the `#line` of its source.
#[derive(Clone, Copy, Debug, Default)]
pub struct LineDirectives;

pub fn block_to_cpp(ctx: &Context, block: Ptr<BasicBlock>) -> String {
    let lines = ctx.try_aux_ty::<LineDirectives>().is_some();
    let mut out = String::new();
    // The file and line that the C++ compiler gives to the next line of `out`, if known.
    let mut next_line: Option<(&PathBuf, u32)> = None;
    let ops = block.deref(ctx).iter(ctx);
    for op in ops {
        // `Display` can't fail, so record the error and let `compile_ir` fail the compilation.
        let cpp = match op.to_cpp(ctx) {
            Ok(cpp) => cpp,
            Err(err) => {
                ctx.aux_ty::<EmissionErrors>().record(err);
                continue;
            }
        };
        if cpp.is_empty() {
            continue;
        }
        if lines {
            next_line = line_directive(ctx, op, next_line, &mut out);
        }
        out.push_str(&cpp);
        next_line = match next_line {
            // A nested block wrote its own directives, so the line after it is not known.
            Some(_) if cpp.contains("#line ") => None,
            Some((file, line)) => Some((file, line + cpp.matches('\n').count() as u32)),
            None => None,
        };
    }
    out
}

/// Writes `#line` for `op` into `out`, unless the next line of `out` already has the source line
/// of `op`. Returns the source line of the next line of `out`. C++ cannot show inlined frames, so
/// the directive has the innermost frame of the location.
fn line_directive<'c>(
    ctx: &'c Context,
    op: Ptr<Operation>,
    next_line: Option<(&'c PathBuf, u32)>,
    out: &mut String,
) -> Option<(&'c PathBuf, u32)> {
    let Some(source) = leaf_line(ctx, &op.deref(ctx).loc()) else {
        return next_line;
    };
    if next_line == Some(source) {
        return next_line;
    }
    // A directive must start a line. The caller can put a block after other text on its line, as
    // `case N: { <block> }`, so an empty `out` also gets the newline.
    if !out.ends_with('\n') {
        out.push('\n');
    }
    let (file, line) = source;
    let _ = write!(out, "#line {line} \"");
    // The file name is a C string literal.
    let file = file.display().to_string();
    let escaped = file.chars().flat_map(|c| {
        let backslash = matches!(c, '\\' | '"').then_some('\\');
        backslash.into_iter().chain([c])
    });
    out.extend(escaped);
    out.push_str("\"\n");
    Some(source)
}

shared_op!(IfOp, |op, ctx| {
    let cond = op.condition(ctx).name(ctx);
    let else_block = op.else_block(ctx);
    let mut out = format!("if({cond}) {{\n");
    out.push_str(&block_to_cpp(ctx, op.then_block(ctx)));
    if else_block.deref(ctx).iter(ctx).count() > 1 {
        out.push_str("}\n else {\n");
        out.push_str(&block_to_cpp(ctx, else_block));
    }
    out.push_str("}\n");
    out
});

shared_op!(SwitchOp, |op, ctx| {
    let value = op.value(ctx).name(ctx);
    let mut out = format!("switch({value}) {{\n");
    for (value, block) in op.cases(ctx) {
        let block = block_to_cpp(ctx, block);
        let case = format!("case {}: {{ {block} break; }}\n", value.value().to_i128());
        out.push_str(&case);
    }
    let block = block_to_cpp(ctx, op.default_block(ctx));
    out.push_str(&format!("default: {{ {block} break; }}\n"));
    out.push_str("}\n");
    out
});

// Only relevant for IR structure
shared_op!(YieldOp, |_, _| String::new());
shared_op!(ConditionOp, |op, ctx| {
    format!("return {};", op.condition(ctx).name(ctx))
});

shared_op!(ReturnOp, |op, ctx| {
    if let Some(value) = op.value(ctx) {
        format!("return {};", value.name(ctx))
    } else {
        "return;".into()
    }
});

shared_op!(UnreachableOp, |_, _| "__builtin_unreachable();".into());

shared_op!(RangeLoopOp, |op, ctx| {
    let i = op.iter_var(ctx).name(ctx);
    let i_ty = op.iter_var(ctx).get_type(ctx).to_cpp(ctx);
    let start = op.start(ctx).name(ctx);
    let end = op.end(ctx).name(ctx);
    let step = op.step(ctx).name(ctx);
    let mut out = format!("for({i_ty} {i} = {start}; {i} < {end}; {i} += {step}) {{\n");
    out.push_str(&block_to_cpp(ctx, op.loop_body(ctx)));
    out.push_str("}\n");
    out
});

shared_op!(WhileOp, |op, ctx| {
    let cond = scoped_block! {
        block_to_cpp(ctx, op.before_block(ctx))
    };
    let mut out = format!("while({cond}) {{\n");
    out.push_str(&block_to_cpp(ctx, op.after_block(ctx)));
    out.push_str("}\n");
    out
});

shared_op_with_out!(SelectOp, |op, ctx| {
    let cond = op.condition(ctx).name(ctx);
    let then = op.true_value(ctx).name(ctx);
    let or_else = op.false_value(ctx).name(ctx);
    format!("{} ? {} : {}", cond, then, or_else)
});
unrolling!(SelectOp);

#[cfg(test)]
mod tests {
    use crate::{
        shared::{CompilationOptions, CppCompiler, register_supported_types},
        target::Cuda,
    };
    use cubecl_core as cubecl;
    use cubecl_core::{
        Compiler,
        ir::{DeviceProperties, settings::DebugInfo},
        prelude::*,
        runtime_tests::offline::offline_device_properties,
    };
    use cubecl_runtime::kernel::CubeKernel;
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

    fn properties() -> Arc<DeviceProperties> {
        let mut properties = offline_device_properties(32);
        register_supported_types(&mut properties);
        Arc::new(properties)
    }

    /// The CUDA source of `outer` at the debug level `level`. The level is set after the
    /// resolution, which gives a `dev` build at least line tables, as `CUBECL_DEBUG_INFO` does.
    fn source(level: DebugInfo) -> String {
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
        CppCompiler::<Cuda>::default()
            .compile(definition, &CompilationOptions::default())
            .unwrap()
            .to_string()
    }

    /// The lines of the `#line` directives in `source`.
    fn directive_lines(source: &str) -> Vec<u32> {
        source
            .lines()
            .filter_map(|line| line.strip_prefix("#line "))
            .map(|rest| {
                let (line, file) = rest.split_once(' ').unwrap();
                assert!(file.ends_with("branch.rs\""), "{rest}");
                line.parse().unwrap()
            })
            .collect()
    }

    /// Each op gets the line of its innermost `#[cube]` frame, so the lines of `inner`, `mid` and
    /// `outer` are in the source.
    #[test]
    fn source_lines_are_line_directives() {
        let source = source(DebugInfo::LineTables);
        let starts_a_line = source
            .match_indices("#line")
            .all(|(at, _)| at == 0 || source.as_bytes()[at - 1] == b'\n');
        assert!(
            starts_a_line,
            "a directive is not at the start of a line:\n{source}"
        );
        let mut lines = directive_lines(&source);
        lines.sort();
        lines.dedup();
        assert!(lines.len() >= 4, "lines {lines:?} in:\n{source}");
    }

    /// Without debug data, the source has no directive, although the ops have locations.
    #[test]
    fn no_line_directives_without_debug_data() {
        let source = source(DebugInfo::None);
        assert!(!source.contains("#line"), "{source}");
    }
}
