//! Kernel-side users of the launch workspace: counters that the runtime zeroes before each
//! launch of a persistent kernel.

use cubecl_ir::{dialect::memory::IndexOp, settings::Persistence};

use crate::{self as cubecl, prelude::*, unexpanded};

/// The slot of the work queue counter in the launch workspace.
const WORK_QUEUE_SLOT: usize = 0;
/// The slot of the spin barrier counter in the launch workspace.
const SPIN_BARRIER_SLOT: usize = 1;

/// The counter in `slot` of the launch workspace.
fn workspace_counter(scope: &Scope, slot: usize) -> NativeExpand<Atomic<u32>> {
    let workspace = scope.launch_workspace();
    let slot = scope.const_usize(slot);
    let counter = IndexOp::new(scope.ctx_mut(), workspace, slot, None);
    scope.register_with_result(&counter).into()
}

/// Takes the next item of the launch's work queue. Every unit of the cube gets the same item.
///
/// The runtime resets the queue before each launch, so the items of every launch start at `0`.
/// Only a kernel marked `persistent` or `cooperative` may call it. Take items until one is at
/// least the number of items:
///
/// ```ignore
/// let mut item = next_work_item();
/// while item < total {
///     // Process `item`.
///     item = next_work_item();
/// }
/// ```
pub fn next_work_item() -> u32 {
    unexpanded!()
}

pub mod next_work_item {
    use super::*;

    pub fn expand(scope: &Scope) -> NativeExpand<u32> {
        if scope.state().persistence == Persistence::None {
            scope
                .push_error("`next_work_item` needs a kernel marked `persistent` or `cooperative`");
        }
        take_work_item::expand(scope, &workspace_counter(scope, WORK_QUEUE_SLOT))
    }
}

/// [`next_work_item`] with a counter that the caller owns, for example for a second queue.
///
/// # Safety
///
/// The caller must write `0` to `counter` before each launch. Else the launch skips items.
#[allow(unused_variables)]
pub unsafe fn next_work_item_from(counter: &Atomic<u32>) -> u32 {
    unexpanded!()
}

pub mod next_work_item_from {
    use super::*;

    /// # Safety
    ///
    /// See [`next_work_item_from`](super::next_work_item_from).
    pub unsafe fn expand(scope: &Scope, counter: &NativeExpand<Atomic<u32>>) -> NativeExpand<u32> {
        take_work_item::expand(scope, counter)
    }
}

/// Unit 0 takes the item, and shares it with the rest of the cube.
#[cube]
fn take_work_item(counter: &Atomic<u32>) -> u32 {
    let mut item = Shared::<u32>::new();
    if UNIT_POS == 0 {
        *item = counter.fetch_add(1);
    }
    sync_cube();
    let taken = *item;
    // No unit may take the next item before every unit read this one.
    sync_cube();
    taken
}

/// Expands a grid barrier for a runtime without native grid sync (see [`spin_barrier`]).
pub(crate) fn spin_grid_barrier(scope: &Scope) {
    spin_barrier::expand(scope, &workspace_counter(scope, SPIN_BARRIER_SLOT));
}

/// Unit 0 of each cube takes a ticket, then spins until every cube of the launch took a ticket
/// of the same round. The counter is never reset in a launch.
#[cube]
fn spin_barrier(counter: &Atomic<u32>) {
    // Release: every storage write of this cube is visible before it arrives.
    sync_storage();
    if UNIT_POS == 0 {
        let cubes = CUBE_COUNT as u32;
        let round_end = (counter.fetch_add(1) / cubes + 1) * cubes;
        while counter.load() < round_end {}
    }
    // Acquire, and the other units of the cube wait for unit 0 here.
    sync_storage();
}
