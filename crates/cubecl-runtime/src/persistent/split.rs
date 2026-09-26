//! Emulates a grid sync by splitting the kernel. Each part between two grid syncs, a phase, runs
//! as its own launch. Launches on one stream run in order, and each sees the writes of the one
//! before, so the launch boundary is the barrier.
//!
//! A phase keeps the ops of its part, and the pure ops of earlier parts that compute values it
//! uses. It cannot keep a value that an earlier part loaded or stored, because that value lived
//! in registers, which do not survive the launch boundary. Shared memory survives only with
//! [`SharedAfterGridSync::Spill`]: each phase copies it to a spill buffer at its end, and the
//! next phase copies it back at its start.

use alloc::{boxed::Box, format, string::String, sync::Arc, vec, vec::Vec};
use core::result::Result;

use cubecl_environment::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};
use cubecl_ir::{
    AddressSpace, Builtin, ElemType, OpInserter, Scope, UIntKind,
    dialect::{
        branch::{RangeLoopOp, YieldOp},
        general::{CastOp, ReadBuiltinOp},
        math::{IAddOp, IMulOp},
        memory::{DeclareVariableOp, IndexOp, LoadOp, StoreOp},
        synchronization::{GridSyncOp, SyncOp, SyncScope},
    },
    features::{GridSync, GridSyncEmulation},
    interfaces::{MemoryEffects, TypedExt},
    prelude::*,
    settings::{Persistence, SharedAfterGridSync},
    types::{ArrayType, AtomicType, scalar::IndexType},
};
use pliron::{basic_block::BasicBlock, linked_list::ContainsLinkedList, r#type::TypeHandle};

use crate::{
    client::Client,
    id::KernelId,
    kernel::{BufferIOAttr, CubeKernel, KernelDefinition, KernelMetadata},
    server::{CubeCount, KernelArguments, KernelResource},
};

/// What a split kernel needs from the launch.
#[derive(Clone, Debug)]
pub struct SplitPlan {
    /// The number of phases, one more than the number of grid syncs.
    pub phases: usize,
    /// The size of each spill buffer, per cube. The buffers follow the kernel's own buffers.
    pub spill_bytes_per_cube: Vec<usize>,
}

/// One phase of a split kernel, launched as a kernel of its own.
pub struct SplitPhase {
    kernel: Arc<dyn CubeKernel>,
    phase: usize,
}

impl SplitPhase {
    /// Launches every phase of `kernel` with `cubes` cubes. Gives `kernel` back if the client
    /// does not split it.
    pub fn launch(
        client: &Client,
        kernel: Box<dyn CubeKernel>,
        cubes: u32,
        mut bindings: KernelArguments,
    ) -> Result<(), Box<dyn CubeKernel>> {
        let Some(plan) = plan(client, kernel.as_ref()) else {
            return Err(kernel);
        };
        for bytes in plan.spill_bytes_per_cube {
            let spill = client.empty(bytes * cubes as usize);
            bindings
                .resources
                .push(KernelResource::Buffer(spill.binding()));
            bindings.declared_io.push(BufferIOAttr::ReadWrite);
        }
        let kernel: Arc<dyn CubeKernel> = kernel.into();
        for phase in 0..plan.phases {
            let phase = Self {
                kernel: kernel.clone(),
                phase,
            };
            client.launch(Box::new(phase), CubeCount::new_1d(cubes), bindings.clone());
        }
        Ok(())
    }
}

impl KernelMetadata for SplitPhase {
    fn name(&self) -> &'static str {
        self.kernel.name()
    }

    fn id(&self) -> KernelId {
        let inner = self.kernel.id();
        KernelId::new::<Self>()
            .cube_dim(inner.cube_dim)
            .address_type(inner.address_type)
            .mode(inner.mode)
            .persistence(Persistence::Persistent)
            .info((inner, self.phase))
    }

    fn address_type(&self) -> ElemType {
        self.kernel.address_type()
    }
}

impl CubeKernel for SplitPhase {
    fn define(&self) -> KernelDefinition {
        let mut definition = self.kernel.define();
        if let Err(error) = split(&mut definition, self.phase) {
            definition.body.push_error(error);
        }
        definition
    }
}

/// The split plan, if `kernel` is cooperative and the client emulates its grid syncs by
/// splitting. Expands the kernel only the first time, because the plan is cached.
fn plan(client: &Client, kernel: &dyn CubeKernel) -> Option<SplitPlan> {
    static PLANS: LazyLock<Mutex<HashMap<KernelId, SplitPlan>>> = LazyLock::new(Default::default);

    let id = kernel.id();
    let Persistence::Cooperative(options) = id.persistence else {
        return None;
    };
    let GridSync::Emulated(emulations) = client.properties().features.grid_sync else {
        return None;
    };
    let spin = options.emulation == GridSyncEmulation::Spin
        && emulations.contains(GridSyncEmulation::Spin);
    if spin || !emulations.contains(GridSyncEmulation::Split) {
        return None;
    }

    let mut plans = PLANS.lock();
    let plan = plans.entry(id).or_insert_with(|| {
        // A kernel that does not split is still launched, as one phase that fails to compile.
        split(&mut kernel.define(), 0).unwrap_or(SplitPlan {
            phases: 1,
            spill_bytes_per_cube: Vec::new(),
        })
    });
    Some(plan.clone())
}

/// Reduces `definition` to phase `phase`.
pub fn split(definition: &mut KernelDefinition, phase: usize) -> Result<SplitPlan, String> {
    let shared_mode = match definition.settings.persistence {
        Persistence::Cooperative(options) => options.shared,
        _ => SharedAfterGridSync::default(),
    };
    definition.settings.persistence = Persistence::Persistent;
    let entry_func = definition.body.state().entry_func;
    let ctx = definition.body.ctx_mut();
    let entry_block = entry_func.get_entry_block(ctx);
    let ops: Vec<_> = entry_block.deref(ctx).iter(ctx).collect();

    let segments = segments(ctx, entry_func.get_operation(), &ops)?;
    let phases = segments.iter().filter(|s| s.is_none()).count() + 1;
    if phase >= phases {
        return Err(format!(
            "phase {phase} does not exist: the kernel has {phases}"
        ));
    }
    let segment_of = |op: Ptr<Operation>| {
        let top = top_level(ctx, entry_block, op);
        ops.iter()
            .position(|it| *it == top)
            .and_then(|i| segments[i])
    };

    let spills = match shared_mode {
        SharedAfterGridSync::Spill => spilled_shared(ctx, &ops, segment_of)?,
        SharedAfterGridSync::Discard => Vec::new(),
    };
    let restored = |spill: &Spill| spill.first < phase && spill.last >= phase;
    let saved = |spill: &Spill| spill.first <= phase && spill.last > phase;

    let mut keep: Vec<_> = ops
        .iter()
        .zip(&segments)
        .filter(|(_, segment)| **segment == Some(phase))
        .map(|(op, _)| *op)
        .collect();
    let phase_start = keep.first().copied();
    keep.extend(
        spills
            .iter()
            .filter(|spill| restored(spill) || saved(spill))
            .map(|spill| spill.var.get_operation()),
    );

    let mut next = 0;
    while next < keep.len() {
        for operand in operands_within(ctx, keep[next]) {
            let Some(def) = operand.defining_op() else {
                continue;
            };
            if keep.contains(&def) || segment_of(def).is_none_or(|s| s >= phase) {
                continue;
            }
            check_recomputable(ctx, def, |user| segment_of(user).is_some_and(|s| s < phase))?;
            keep.push(def);
        }
        next += 1;
    }

    for op in ops.iter().rev().filter(|op| !keep.contains(op)) {
        Operation::erase(*op, ctx);
    }

    let first_spill = definition.body.next_buffer_pos();
    let mut spill_bytes_per_cube = Vec::with_capacity(spills.len());
    for (i, spill) in spills.iter().enumerate() {
        let buffer = definition.body.global(first_spill + i, None, spill.elem_ty);
        let ctx = definition.body.ctx_mut();
        spill_bytes_per_cube.push(spill.elem_ty.size(ctx) * spill.len);
        if restored(spill) {
            let at = match phase_start {
                Some(op) => OpInserter::new_before_operation(op),
                None => OpInserter::new_at_block_end(entry_block),
            };
            copy_shared(ctx, at, spill, buffer, Direction::Restore);
        }
        if saved(spill) {
            let at = OpInserter::new_at_block_end(entry_block);
            copy_shared(ctx, at, spill, buffer, Direction::Save);
        }
    }

    Ok(SplitPlan {
        phases,
        spill_bytes_per_cube,
    })
}

/// A shared memory variable that is used in more than one phase.
struct Spill {
    var: DeclareVariableOp,
    elem_ty: TypeHandle,
    len: usize,
    first: usize,
    last: usize,
}

fn spilled_shared(
    ctx: &Context,
    ops: &[Ptr<Operation>],
    segment_of: impl Fn(Ptr<Operation>) -> Option<usize>,
) -> Result<Vec<Spill>, String> {
    let mut spills = Vec::new();
    for op in ops {
        let Some(var) = Operation::get_op::<DeclareVariableOp>(*op, ctx) else {
            continue;
        };
        if var.addr_space(ctx).0 != AddressSpace::Shared {
            continue;
        }
        let segments: Vec<_> = derived_users(ctx, *op)
            .into_iter()
            .filter_map(&segment_of)
            .collect();
        let (Some(first), Some(last)) = (segments.iter().min(), segments.iter().max()) else {
            continue;
        };
        if first == last {
            continue;
        }

        let value_ty = var.value_ty(ctx).get_type(ctx);
        let (elem_ty, len) = match value_ty.deref(ctx).downcast_ref::<ArrayType>() {
            Some(array) => (array.inner, array.length),
            None => (value_ty, 1),
        };
        if elem_ty.deref(ctx).is::<AtomicType>() {
            return Err(
                "atomic shared memory cannot be kept across an emulated `sync_grid`: \
                        use `shared_after_grid_sync = \"discard\"`"
                    .into(),
            );
        }
        spills.push(Spill {
            var,
            elem_ty,
            len,
            first: *first,
            last: *last,
        });
    }
    Ok(spills)
}

/// The users of the results of `op`, and of every pure op that derives a value from them,
/// such as a slice of a shared array or a pointer into it.
fn derived_users(ctx: &Context, op: Ptr<Operation>) -> Vec<Ptr<Operation>> {
    let mut users: Vec<Ptr<Operation>> = Vec::new();
    let mut defs = vec![op];
    while let Some(def) = defs.pop() {
        let results: Vec<_> = def.deref(ctx).results().collect();
        for user in results.iter().flat_map(|result| result.uses(ctx)) {
            let user = user.user_op();
            if users.contains(&user) {
                continue;
            }
            users.push(user);
            if is_pure(ctx, user) {
                defs.push(user);
            }
        }
    }
    users
}

fn is_pure(ctx: &Context, op: Ptr<Operation>) -> bool {
    let no_effects =
        op_cast::<dyn MemoryEffects>(&*op.dyn_op(ctx)).is_some_and(|e| !e.has_effects(ctx));
    no_effects && op.deref(ctx).num_regions() == 0
}

enum Direction {
    Save,
    Restore,
}

/// Copies the shared variable of `spill` to or from the part of `buffer` that belongs to this
/// cube. Each unit copies every `CUBE_DIM`-th element, between two cube barriers.
fn copy_shared(
    ctx: &mut Context,
    mut at: OpInserter,
    spill: &Spill,
    buffer: Value,
    direction: Direction,
) {
    let scope = Scope::from_context_and_inserter(ctx, &mut at);
    let sync_cube = || scope.register(&SyncOp::new(scope.ctx_mut(), SyncScope::Cube));
    let index_ty: TypeHandle = IndexType::get(scope.ctx()).into();

    sync_cube();
    let cube_pos = ReadBuiltinOp::new(scope.ctx_mut(), index_ty, Builtin::CubePos);
    let cube_pos = scope.register_with_result(&cube_pos);
    let len = scope.const_usize(spill.len);
    let cube_offset = scope.register_with_result(&IMulOp::new(scope.ctx_mut(), cube_pos, len));
    let unit = builtin_as_index(&scope, Builtin::UnitPos);
    let step = builtin_as_index(&scope, Builtin::CubeDim);
    let copy = RangeLoopOp::new(scope.ctx_mut(), unit, len, step);
    let i = copy.iter_var(scope.ctx());
    let body = scope.child(OpInserter::new_at_block_end(copy.loop_body(scope.ctx())));

    let shared = spill.var.get_result(body.ctx());
    let shared = match spill.len {
        1 => shared,
        _ => body.register_with_result(&IndexOp::new(body.ctx_mut(), shared, i, None)),
    };
    let offset = body.register_with_result(&IAddOp::new(body.ctx_mut(), cube_offset, i));
    let global = body.register_with_result(&IndexOp::new(body.ctx_mut(), buffer, offset, None));
    let (from, to) = match direction {
        Direction::Save => (shared, global),
        Direction::Restore => (global, shared),
    };
    let value = body.register_with_result(&LoadOp::new(body.ctx_mut(), from));
    body.register(&StoreOp::new(body.ctx_mut(), to, value));
    body.register(&YieldOp::new(body.ctx_mut()));

    scope.register(&copy);
    sync_cube();
}

/// Reads a `u32` builtin and casts it to the index type.
fn builtin_as_index(scope: &Scope, builtin: Builtin) -> Value {
    let u32_ty = ElemType::UInt(UIntKind::U32).to_type(scope.ctx_mut());
    let raw = ReadBuiltinOp::new(scope.ctx_mut(), u32_ty, builtin);
    let raw = scope.register_with_result(&raw);
    let index_ty: TypeHandle = IndexType::get(scope.ctx()).into();
    scope.register_with_result(&CastOp::new(scope.ctx_mut(), index_ty, raw))
}

/// The phase of each op in the entry block, or `None` for a grid sync.
fn segments(
    ctx: &Context,
    entry_func: Ptr<Operation>,
    ops: &[Ptr<Operation>],
) -> Result<Vec<Option<usize>>, String> {
    let mut syncs = 0;
    visit_all_ops_of_type::<GridSyncOp, _>(ctx, &mut syncs, entry_func, |_, n, _| *n += 1);
    let top_level = ops
        .iter()
        .filter(|op| Operation::get_op::<GridSyncOp>(**op, ctx).is_some());
    if top_level.count() != syncs {
        return Err(
            "`sync_grid` inside a branch or a loop needs native grid sync or \
                    `grid_sync_emulation = \"spin\"`"
                .into(),
        );
    }

    let mut phase = 0;
    Ok(ops
        .iter()
        .map(|op| match Operation::get_op::<GridSyncOp>(*op, ctx) {
            Some(_) => {
                phase += 1;
                None
            }
            None => Some(phase),
        })
        .collect())
}

/// Errors if a later phase cannot compute the result of `op` again. `earlier` tells if a user
/// of the result is in an earlier phase.
fn check_recomputable(
    ctx: &Context,
    op: Ptr<Operation>,
    earlier: impl Fn(Ptr<Operation>) -> bool,
) -> Result<(), String> {
    if let Some(var) = Operation::get_op::<DeclareVariableOp>(op, ctx) {
        let used_before = op
            .deref(ctx)
            .results()
            .any(|result| result.uses(ctx).iter().any(|u| earlier(u.user_op())));
        return match var.addr_space(ctx).0 {
            AddressSpace::Shared => Ok(()),
            _ if !used_before => Ok(()),
            _ => Err(across_error(ctx, op)),
        };
    }

    match is_pure(ctx, op) {
        true => Ok(()),
        false => Err(across_error(ctx, op)),
    }
}

fn across_error(ctx: &Context, op: Ptr<Operation>) -> String {
    format!(
        "a value from before `sync_grid` is used after it, but it cannot be computed again \
         after it (`{}`). Values in registers do not survive an emulated grid sync: compute it \
         again after `sync_grid`, or keep it in a buffer.",
        op.dyn_op(ctx).get_opid()
    )
}

/// The operands of `op` and of every op nested in it.
fn operands_within(ctx: &Context, op: Ptr<Operation>) -> Vec<Value> {
    let mut operands: Vec<_> = op.deref(ctx).operands().collect();
    for region in op.deref(ctx).regions() {
        for block in region.deref(ctx).iter(ctx) {
            for inner in block.deref(ctx).iter(ctx) {
                operands.extend(operands_within(ctx, inner));
            }
        }
    }
    operands
}

/// The op of the entry block that contains `op`.
fn top_level(
    ctx: &Context,
    entry_block: Ptr<BasicBlock>,
    mut op: Ptr<Operation>,
) -> Ptr<Operation> {
    while op.deref(ctx).get_parent_block() != Some(entry_block) {
        match op.deref(ctx).get_parent_op(ctx) {
            Some(parent) => op = parent,
            None => break,
        }
    }
    op
}
