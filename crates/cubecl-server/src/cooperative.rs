//! Launches on a runtime with native grid sync (CUDA, HIP).
//!
//! A cooperative launch runs every cube of the launch on the device at the same time. So its cube
//! count must not exceed the capacity: the number of cubes of the compiled kernel that the device
//! runs at the same time.

use cubecl_environment::{
    backtrace::BackTrace,
    collections::HashSet,
    sync::{LazyLock, Mutex},
};
use cubecl_ir::settings::Persistence;
use cubecl_runtime::{
    id::KernelId,
    persistent::PersistentCount,
    server::{CubeCount, LaunchError, ResourceLimitError},
};

/// The cube count of a launch, as the server got it.
pub enum LaunchCount {
    /// From [`Server::launch`](cubecl_runtime::server::Server::launch).
    Grid(CubeCount),
    /// From [`Server::launch_persistent`](cubecl_runtime::server::Server::launch_persistent).
    Persistent(PersistentCount),
}

impl LaunchCount {
    /// The count that [`launch_lowering`] may lower, if the launch is persistent.
    pub fn lowering(&self) -> Option<PersistentCount> {
        match self {
            Self::Grid(_) => None,
            Self::Persistent(count) => Some(*count),
        }
    }
}

/// Whether the device must run every cube of a launch of `kernel_id` at the same time.
pub fn is_cooperative(kernel_id: &KernelId) -> bool {
    matches!(kernel_id.persistence, Persistence::Cooperative(_))
}

/// The error for a cooperative launch of `requested` cubes when the device runs only `max`.
pub fn too_many_cubes(requested: u32, max: u32) -> ResourceLimitError {
    ResourceLimitError::CooperativeGrid {
        requested,
        max,
        backtrace: BackTrace::capture(),
    }
}

/// The cube count of a persistent launch of `kernel_id`, from the `capacity` that the runtime
/// queried. `max` is the largest cube count that the device launches.
///
/// # Errors
///
/// [`ResourceLimitError::CooperativeGrid`] if the launch is cooperative and the count exceeds the
/// capacity. Only [`PersistentCount::Exact`] can do this, or any count if the capacity is `0`.
pub fn resolve_count(
    kernel_id: &KernelId,
    count: PersistentCount,
    capacity: u32,
    max: u32,
) -> Result<u32, ResourceLimitError> {
    let cubes = count.resolve(capacity, max);
    if is_cooperative(kernel_id) && cubes > capacity {
        return Err(too_many_cubes(cubes, capacity));
    }
    Ok(cubes)
}

/// Calls `launch` with `cubes` cubes.
///
/// The driver can refuse a count that [`resolve_count`] accepted, for example when another
/// process shares the device. `launch` then returns [`ResourceLimitError::CooperativeGrid`]. If
/// `count` is not [`PersistentCount::Exact`], this function halves the count and calls `launch`
/// again, until the launch succeeds or the count is `1`. It logs one warning per kernel.
///
/// # Errors
///
/// The last error of `launch`.
pub fn launch_lowering(
    kernel_id: &KernelId,
    count: PersistentCount,
    cubes: u32,
    mut launch: impl FnMut(u32) -> Result<(), LaunchError>,
) -> Result<(), LaunchError> {
    let may_lower = !matches!(count, PersistentCount::Exact(_));
    let mut used = cubes;
    loop {
        match launch(used) {
            Err(LaunchError::TooManyResources(ResourceLimitError::CooperativeGrid { .. }))
                if may_lower && used > 1 =>
            {
                used /= 2;
            }
            result => {
                if result.is_ok() && used < cubes {
                    warn_lowered(kernel_id, cubes, used);
                }
                return result;
            }
        }
    }
}

/// The kernels whose count was lowered, so each one warns only one time.
static LOWERED: LazyLock<Mutex<HashSet<KernelId>>> = LazyLock::new(Default::default);

fn warn_lowered(kernel_id: &KernelId, requested: u32, used: u32) {
    if LOWERED.lock().insert(kernel_id.clone()) {
        log::warn!(
            "The driver refused {requested} cubes for the cooperative kernel {kernel_id}; it runs \
             with {used}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use cubecl_ir::settings::CooperativeOptions;

    fn cooperative() -> KernelId {
        KernelId::new::<()>().persistence(Persistence::Cooperative(CooperativeOptions::default()))
    }

    /// A driver that runs at most `capacity` cubes, and records each count it gets.
    fn driver(capacity: u32, calls: &mut Vec<u32>) -> impl FnMut(u32) -> Result<(), LaunchError> {
        move |cubes| {
            calls.push(cubes);
            if cubes > capacity {
                Err(too_many_cubes(cubes, capacity).into())
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn exact_above_the_capacity_fails() {
        let error = resolve_count(&cooperative(), PersistentCount::Exact(11), 10, 1000);
        assert!(matches!(
            error,
            Err(ResourceLimitError::CooperativeGrid {
                requested: 11,
                max: 10,
                ..
            })
        ));
    }

    #[test]
    fn exact_above_the_capacity_is_valid_without_cooperation() {
        let kernel_id = KernelId::new::<()>().persistence(Persistence::Persistent);
        let cubes = resolve_count(&kernel_id, PersistentCount::Exact(11), 10, 1000);
        assert_eq!(cubes.unwrap(), 11);
    }

    #[test]
    fn a_refused_count_is_halved() {
        let mut calls = Vec::new();
        let result = launch_lowering(
            &cooperative(),
            PersistentCount::Fill,
            40,
            driver(12, &mut calls),
        );
        assert!(result.is_ok());
        assert_eq!(calls, [40, 20, 10]);
    }

    #[test]
    fn exact_is_not_lowered() {
        let mut calls = Vec::new();
        let result = launch_lowering(
            &cooperative(),
            PersistentCount::Exact(40),
            40,
            driver(12, &mut calls),
        );
        assert!(result.is_err());
        assert_eq!(calls, [40]);
    }

    #[test]
    fn lowering_stops_at_one_cube() {
        let mut calls = Vec::new();
        let result = launch_lowering(
            &cooperative(),
            PersistentCount::AtMost(4),
            4,
            driver(0, &mut calls),
        );
        assert!(result.is_err());
        assert_eq!(calls, [4, 2, 1]);
    }
}
