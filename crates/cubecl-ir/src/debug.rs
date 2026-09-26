//! Source locations of the ops that kernel expansion inserts.
//!
//! The macro-generated code of each `#[cube]` function opens a [frame](DebugState::enter_fn) and
//! moves its [position](DebugState::set_pos). [`LocationListener`] gives each inserted op the
//! location of the current frame, nested in the call sites of its callers.
//!
//! Expansion inlines every `#[cube]` call, so the call chain is the only record of the source-level
//! call stack. For an op in `inner`, called from `outer`, called from kernel `k`:
//!
//! ```text
//! CallSite {
//!     callee: Named("inner", SrcPos(op)),
//!     caller: CallSite {
//!         callee: Named("outer", SrcPos(call to inner)),
//!         caller: Named("k", SrcPos(call to outer)),
//!     },
//! }
//! ```

use alloc::{boxed::Box, string::String, vec::Vec};
use pliron::{
    basic_block::BasicBlock,
    combine::stream::position::SourcePosition,
    context::{Context, Ptr},
    irbuild::listener::InsertionListener,
    location::{Located, Location, Source},
    operation::Operation,
};

use crate::ContextExt;

/// The `#[cube]` call stack during kernel expansion.
///
/// A function without debug data, such as a hand-written expand function, opens no frame. Its ops
/// get the location of the call to it.
#[derive(Debug, Default)]
pub struct DebugState {
    enabled: bool,
    frames: Vec<Frame>,
}

/// One `#[cube]` function on the call stack.
#[derive(Debug)]
struct Frame {
    name: String,
    file: Source,
    /// The expression the function is at: the call site, while a callee runs.
    pos: SourcePosition,
}

impl DebugState {
    /// An empty call stack that records locations only when `enabled`.
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            frames: Vec::new(),
        }
    }

    /// Whether expansion records locations.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Opens the frame of function `name`, defined at `line` and `column` of `file`.
    pub fn enter_fn(&mut self, name: &str, file: Source, line: u32, column: u32) {
        self.frames.push(Frame {
            name: name.into(),
            file,
            pos: position(line, column),
        });
    }

    /// Closes the innermost frame.
    pub fn exit_fn(&mut self) {
        self.frames.pop();
    }

    /// Moves the innermost frame to `line` and `column`, and returns its previous position.
    pub fn set_pos(&mut self, line: u32, column: u32) -> Option<SourcePosition> {
        let frame = self.frames.last_mut()?;
        Some(core::mem::replace(&mut frame.pos, position(line, column)))
    }

    /// Moves the innermost frame back to `pos`, a value that [`set_pos`](Self::set_pos) returned.
    pub fn restore_pos(&mut self, pos: SourcePosition) {
        if let Some(frame) = self.frames.last_mut() {
            frame.pos = pos;
        }
    }

    /// Stops recording, at the end of kernel expansion. Ops that compiler passes insert later get
    /// the location of the op they replace.
    pub fn finish(&mut self) {
        *self = Self::default();
    }

    /// The location of the current expression, or `None` outside a `#[cube]` function.
    fn location(&self) -> Option<Location> {
        let (kernel, callees) = self.frames.split_first()?;
        let location =
            callees
                .iter()
                .fold(kernel.location(), |caller, callee| Location::CallSite {
                    callee: Box::new(callee.location()),
                    caller: Box::new(caller),
                });
        Some(location)
    }
}

impl Frame {
    fn location(&self) -> Location {
        Location::Named {
            name: self.name.clone(),
            child_loc: Box::new(Location::SrcPos {
                src: self.file,
                pos: self.pos,
            }),
        }
    }
}

fn position(line: u32, column: u32) -> SourcePosition {
    SourcePosition {
        line: line as i32,
        column: column as i32,
    }
}

/// Gives each inserted op without a location the location of the current expression.
#[derive(Default)]
pub struct LocationListener;

impl InsertionListener for LocationListener {
    fn notify_operation_inserted(&mut self, ctx: &Context, operation: Ptr<Operation>) {
        let Some(debug) = ctx.try_aux_ty::<DebugState>() else {
            return;
        };
        if let Some(loc) = debug.location()
            && operation.deref(ctx).loc().is_unknown()
        {
            operation.deref_mut(ctx).set_loc(loc);
        }
    }

    fn notify_block_inserted(&mut self, _ctx: &Context, _block: Ptr<BasicBlock>) {}
}
