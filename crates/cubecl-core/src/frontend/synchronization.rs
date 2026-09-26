use crate::{
    frontend::{NativeExpand, element::Atomic},
    ir::{
        Scope,
        dialect::synchronization::{GridSyncOp, SyncAsyncProxyOp, SyncOp, SyncScope},
        features::GridSync,
        settings::Persistence,
    },
    prelude::{CubePrimitive, Numeric},
    unexpanded,
};

// Among all backends, the memory order guarantee of WebGPU is the weakest
// So Cubecl's memory order cannot be stronger than that of WebGPU

/// # Coordinates the following among all invocations in the current cube:
///
/// * Memory writes to variables in cube address space(shared memory) complete,
///   e.g. writes that were initiated actually land in the cube address space memory.
///
/// * Then all the invocations in the cube wait for each other to arrive at the barrier, i.e. this step.
///
/// * Then all the invocations int the cube begin executing after the barrier, and all writes to cube address space made before the barrier are now visible to any invocation in this cube.
pub fn sync_cube() {}

pub mod sync_cube {
    use super::*;

    pub fn expand(scope: &Scope) {
        scope.register(&SyncOp::new(scope.ctx_mut(), SyncScope::Cube));
    }
}

/// Synchronizes units within their plane (e.g., warp or SIMD group).
///
/// Warning: not all targets support plane-level synchronization.
pub fn sync_plane() {
    unexpanded!()
}

pub mod sync_plane {
    use super::*;

    pub fn expand(scope: &Scope) {
        scope.register(&SyncOp::new(scope.ctx_mut(), SyncScope::Plane));
    }
}

/// [`sync_cube`] over storage memory (input args) rather than shared memory, and the one
/// synchronization that reaches past the cube.
///
/// Every unit of the cube must reach it, and it is a [`sync_cube`] as well as what follows, so a
/// value one unit leaves in shared memory before it is readable by the rest after it. On top of
/// that cube barrier it is a release and an acquire at device scope: every write this cube made to storage before it is
/// visible to any other cube that calls it afterwards, and every write another cube published
/// before its own call is visible here after it. That is what lets cubes hand each other partial
/// results — publish, `sync_storage`, then announce through an atomic — with no second dispatch.
///
/// **Only where the runtime says so.** WebGPU's memory model promises nothing across workgroups,
/// so on a backend whose
/// [`device_memory_scope`](cubecl_ir::Features::device_memory_scope) is `false` this is a cube
/// barrier and no more, and a cube reading another's writes may read them stale. Ask before
/// launching a kernel that depends on the promise.
pub fn sync_storage() {}

pub mod sync_storage {
    use super::*;

    pub fn expand(scope: &Scope) {
        scope.register(&SyncOp::new(scope.ctx_mut(), SyncScope::Device));
    }
}

/// Barrier that every unit of every cube of the launch must reach: a [`sync_storage`] for the
/// whole launch. After it, every unit sees every storage write that any cube made before it.
///
/// Only a kernel marked `cooperative` may call it, because all cubes must run at the same time.
/// Every unit must reach the same call, so a call inside a branch that some units skip is an
/// error.
pub fn sync_grid() {}

pub mod sync_grid {
    use super::*;

    pub fn expand(scope: &Scope) {
        if !matches!(scope.state().persistence, Persistence::Cooperative(_)) {
            scope.push_error("`sync_grid` needs a kernel marked `cooperative`");
        }
        register_grid_sync(scope, true);
    }
}

/// [`sync_grid`] in a kernel that is not marked `cooperative`.
///
/// # Safety
///
/// All cubes of the launch must run at the same time. Else the kernel hangs.
pub unsafe fn sync_grid_unchecked() {}

pub mod sync_grid_unchecked {
    use super::*;

    /// # Safety
    ///
    /// See [`sync_grid_unchecked`](super::sync_grid_unchecked).
    pub unsafe fn expand(scope: &Scope) {
        register_grid_sync(scope, false);
    }
}

fn register_grid_sync(scope: &Scope, checked: bool) {
    let supported = scope
        .state()
        .device_properties
        .as_ref()
        .is_none_or(|props| props.features.grid_sync != GridSync::None);
    if !supported {
        scope.push_error("`sync_grid` is not supported on this runtime");
    }
    scope.register(&GridSyncOp::new(scope.ctx_mut(), checked));
}

/// `sync_async_proxy_shared` is a synchronization fence for the experimental SM 9.0+ copy
/// functions, applying bidirectionally between the async proxy (i.e. TMA) and shared memory.
/// Should be used after initializing the barriers, and before the copy operation.
/// PTX: `fence.proxy.async.shared::cta`
/// Experimental and subject to change.
pub fn sync_async_proxy_shared() {
    unexpanded!()
}

pub mod sync_async_proxy_shared {
    use super::*;

    pub fn expand(scope: &Scope) {
        scope.register(&SyncAsyncProxyOp::new(scope.ctx_mut()))
    }
}

/// Barrier, then load `reference` with the result marked workgroup-uniform —
/// mirrors WGSL's `workgroupUniformLoad`. Lets a workgroup-shared value gate
/// control flow that contains barriers. Non-WGSL backends lower it to
/// [`sync_cube`] plus a plain load.
///
/// Use [`workgroup_uniform_load_atomic`] for `Atomic<E>`.
#[allow(unused_variables)]
pub fn workgroup_uniform_load<E: CubePrimitive>(reference: &E) -> E {
    unexpanded!()
}

/// Module containing the expand function for [`workgroup_uniform_load()`].
pub mod workgroup_uniform_load {
    use cubecl_ir::{
        dialect::plane::UniformLoadOp, pliron::builtin::op_interfaces::OneResultInterface,
    };

    use crate::frontend::HasValue;

    use super::*;

    /// Expand method of [`workgroup_uniform_load()`].
    pub fn expand<E: CubePrimitive>(scope: &Scope, reference: &NativeExpand<E>) -> NativeExpand<E> {
        let ptr = reference.value(scope);
        let op = UniformLoadOp::new(scope.ctx_mut(), ptr);
        scope.register(&op);
        op.get_result(scope.ctx()).into()
    }
}

/// Atomic counterpart of [`workgroup_uniform_load`]: barrier + atomic load,
/// returning the underlying numeric (WGSL's atomic `workgroupUniformLoad`
/// overload).
#[allow(unused_variables)]
pub fn workgroup_uniform_load_atomic<E: CubePrimitive<Scalar: Numeric>>(
    reference: &Atomic<E>,
) -> E {
    unexpanded!()
}

/// Module containing the expand function for [`workgroup_uniform_load_atomic()`].
pub mod workgroup_uniform_load_atomic {
    use cubecl_ir::{
        dialect::plane::AtomicUniformLoadOp, pliron::builtin::op_interfaces::OneResultInterface,
    };

    use crate::frontend::HasValue;

    use super::*;

    /// Expand method of [`workgroup_uniform_load_atomic()`].
    pub fn expand<E: CubePrimitive<Scalar: Numeric>>(
        scope: &Scope,
        reference: &NativeExpand<Atomic<E>>,
    ) -> NativeExpand<E> {
        let ptr = reference.value(scope);
        let op = AtomicUniformLoadOp::new(scope.ctx_mut(), ptr);
        scope.register(&op);
        op.get_result(scope.ctx()).into()
    }
}
