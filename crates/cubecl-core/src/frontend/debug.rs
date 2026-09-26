use alloc::{string::String, vec::Vec};
use cubecl_ir::{
    dialect::general::PrintfOp,
    pliron::{combine::stream::position::SourcePosition, location::Source, value::Value},
};

use crate::ir::Scope;

use super::CubeDebug;

/// Moves the current `#[cube]` function to `line` and `column` until the returned guard drops.
pub fn debug_span_expand(scope: &Scope, line: u32, column: u32) -> DebugSpan<'_> {
    let previous = scope
        .debug_state()
        .and_then(|debug| debug.set_pos(line, column));
    DebugSpan { scope, previous }
}

/// Restores the position that [`debug_span_expand`] replaced.
pub struct DebugSpan<'a> {
    scope: &'a Scope,
    previous: Option<SourcePosition>,
}

impl Drop for DebugSpan<'_> {
    fn drop(&mut self) {
        if let Some(previous) = self.previous
            && let Some(debug) = self.scope.debug_state()
        {
            debug.restore_pos(previous);
        }
    }
}

/// Calls a function from `line` and `column` of the current `#[cube]` function.
pub fn debug_call_expand<C>(
    scope: &Scope,
    line: u32,
    column: u32,
    call: impl FnOnce(&Scope) -> C,
) -> C {
    let _span = debug_span_expand(scope, line, column);
    call(scope)
}

/// Opens the frame of the `#[cube]` function `name`, defined at `line` and `column` of `file`.
/// The frame closes when the returned guard drops.
pub fn debug_source_expand<'a>(
    scope: &'a Scope,
    name: &'static str,
    file: &'static str,
    _source_text: &'static str,
    line: u32,
    column: u32,
) -> DebugFrame<'a> {
    if scope.debug_state().is_none() {
        return DebugFrame { scope: None };
    }
    let file = Source::new_from_file(scope.ctx_mut(), file.replace('\\', "/"));
    if let Some(debug) = scope.debug_state() {
        debug.enter_fn(name, file, line, column);
    }
    DebugFrame { scope: Some(scope) }
}

/// Closes the frame that [`debug_source_expand`] opened.
pub struct DebugFrame<'a> {
    scope: Option<&'a Scope>,
}

impl Drop for DebugFrame<'_> {
    fn drop(&mut self) {
        if let Some(debug) = self.scope.and_then(Scope::debug_state) {
            debug.exit_fn();
        }
    }
}

/// Names the value of variable `name`, when the kernel records debug data.
pub fn debug_var_expand<E: CubeDebug>(scope: &Scope, name: &'static str, expand: E) -> E {
    if scope.debug_state().is_some() {
        expand.set_debug_name(scope, name);
    }
    expand
}

/// Prints a formatted message using the print debug layer in Vulkan, or `printf` in CUDA.
pub fn printf_expand(scope: &Scope, format_string: impl Into<String>, args: Vec<Value>) {
    scope.register(&PrintfOp::new(scope.ctx_mut(), format_string.into(), args));
}

/// Print a formatted message using the target's debug print facilities. The format string is target
/// specific, but Vulkan and CUDA both use the C++ conventions. WGSL isn't currently supported.
#[macro_export]
macro_rules! debug_print {
    ($format:literal, $($args:expr),*) => {
        {
            let _ = $format;
            $(let _ = $args;)*
        }
    };
    ($format:literal, $($args:expr,)*) => {
        $crate::debug_print!($format, $($args),*);
    };
}

/// Print a formatted message using the target's debug print facilities. The format string is target
/// specific, but Vulkan and CUDA both use the C++ conventions. WGSL isn't currently supported.
#[macro_export]
macro_rules! __expand_debug_print {
    ($scope:expr, $format:expr, $($args:expr),*) => {
        {
            let args = $crate::__private::vec![$($crate::ir::ExpandValue::from($args).read_value($scope)),*];
            $crate::frontend::printf_expand($scope, $format, args);
        }
    };
    ($format:literal, $($args:expr,)*) => {
        $crate::__expand_debug_print!($format, $($args),*)
    };
}

pub mod cube_comment {
    use alloc::string::ToString;

    use cubecl_ir::{Scope, dialect::general::CommentOp};

    pub fn expand(scope: &Scope, content: &str) {
        scope.register(&CommentOp::new(scope.ctx_mut(), content.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use crate as cubecl;
    use crate::prelude::*;
    use alloc::vec::Vec;
    use cubecl_ir::{
        pliron::{
            linked_list::ContainsLinkedList,
            location::{Located, Location},
        },
        settings::{DebugInfo, Dim3, ExecutionMode, KernelSettings},
    };

    const FIRST_LINE: u32 = line!() + 1;
    #[cube]
    fn double(x: u32) -> u32 {
        x + x
    }

    #[cube]
    fn quadruple(x: u32) -> u32 {
        double(x) + double(x)
    }

    /// Runs `expand` on a new kernel scope and returns the location of each op it inserted.
    fn locations(
        debug_info: DebugInfo,
        expand: impl FnOnce(&Scope, NativeExpand<u32>),
    ) -> Vec<Location> {
        let settings =
            KernelSettings::new(Dim3::new_single(), ExecutionMode::Checked, AddressType::U32)
                .debug_info(debug_info);
        let scope = &Scope::root(settings);
        let x = NativeExpand::<u32>::from_lit(scope, 2);
        expand(scope, x);

        let ctx = scope.ctx();
        let block = scope.state().entry_func.get_entry_block(ctx);
        let ops = block.deref(ctx).iter(ctx).collect::<Vec<_>>();
        ops.into_iter().map(|op| op.deref(ctx).loc()).collect()
    }

    #[test]
    fn ops_get_the_line_of_their_expression() {
        let named = locations(DebugInfo::LineTables, |scope, x| {
            double::expand(scope, x);
        })
        .into_iter()
        .filter_map(|loc| match loc {
            Location::Named { name, child_loc } => Some((name, *child_loc)),
            _ => None,
        })
        .collect::<Vec<_>>();

        assert!(!named.is_empty());
        for (name, loc) in named {
            assert_eq!(name, "double");
            let Location::SrcPos { pos, .. } = loc else {
                panic!("expected a source position, got {loc:?}");
            };
            assert!((FIRST_LINE..FIRST_LINE + 4).contains(&(pos.line as u32)));
        }
    }

    #[test]
    fn inlined_calls_keep_their_call_site() {
        let call_sites = locations(DebugInfo::LineTables, |scope, x| {
            quadruple::expand(scope, x);
        })
        .into_iter()
        .filter_map(|loc| match loc {
            Location::CallSite { callee, caller } => Some((*callee, *caller)),
            _ => None,
        })
        .collect::<Vec<_>>();

        // Every op of the two `double` calls records that `quadruple` called it.
        assert!(!call_sites.is_empty());
        for (callee, caller) in call_sites {
            assert!(matches!(callee, Location::Named { name, .. } if name == "double"));
            assert!(matches!(caller, Location::Named { name, .. } if name == "quadruple"));
        }
    }

    #[test]
    fn no_locations_without_debug_info() {
        let locations = locations(DebugInfo::None, |scope, x| {
            double::expand(scope, x);
        });
        assert!(locations.iter().all(Location::is_unknown));
    }
}
