//! The capacity is how many cubes of one compiled kernel the device runs at the same time.

use crate::{
    client::Client,
    config::{CubeClRuntimeConfig, RuntimeConfig},
    id::KernelId,
    kernel::CubeKernel,
    server::KernelArguments,
};
use alloc::{boxed::Box, vec::Vec};
use core::time::Duration;
use cubecl_environment::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};
use cubecl_ir::LAUNCH_WORKSPACE_WORDS;
use split::SplitPlan;

pub mod split;

/// How many cubes a persistent kernel launches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PersistentCount {
    /// The capacity, scaled by the `persistent.fill_fraction` configuration value.
    Fill,
    /// The capacity, scaled by the given fraction in `(0, 1]`.
    Fraction(f32),
    /// The capacity, but no more than the given count.
    AtMost(u32),
    /// Exactly the given count. A cooperative launch fails if it exceeds the capacity.
    Exact(u32),
}

impl PersistentCount {
    /// The cube count on a device that runs `capacity` cubes at once and launches at most `max`.
    pub fn resolve(self, capacity: u32, max: u32) -> u32 {
        let scaled = |fraction: f32| (capacity as f32 * checked_fraction(fraction)) as u32;
        let cubes = match self {
            Self::Fill => scaled(*FILL_FRACTION),
            Self::Fraction(fraction) => scaled(fraction),
            Self::AtMost(count) => count.min(capacity),
            Self::Exact(count) => return count,
        };
        cubes.max(1).min(max)
    }
}

/// What a persistent launch binds after the kernel's own buffers. Found by expanding the kernel
/// the first time it launches, then cached.
#[derive(Clone, Debug)]
pub(crate) struct LaunchPlan {
    /// Whether the kernel uses the launch workspace.
    pub workspace: bool,
    /// How the kernel splits, if the client emulates its grid syncs by splitting.
    pub split: Option<SplitPlan>,
}

impl LaunchPlan {
    /// The plan of `kernel`. An `exclusive` launch does not split where the device has native
    /// grid sync (D9).
    pub fn of(client: &Client, kernel: &dyn CubeKernel, exclusive: bool) -> Self {
        static PLANS: LazyLock<Mutex<HashMap<(KernelId, bool), LaunchPlan>>> =
            LazyLock::new(Default::default);

        let id = kernel.id();
        let native = exclusive && client.properties().features.exclusive_grid_sync;
        let splits = !native && split::splits(client, &id);
        let mut plans = PLANS.lock();
        let plan = plans.entry((id, splits)).or_insert_with(|| {
            let mut definition = kernel.define();
            let workspace = definition.body.state().workspace.is_some();
            // A kernel that does not split is still launched, as one phase that fails to compile.
            let split = splits.then(|| {
                split::split(&mut definition, 0).unwrap_or(SplitPlan {
                    phases: 1,
                    spill_bytes_per_cube: Vec::new(),
                })
            });
            LaunchPlan { workspace, split }
        });
        plan.clone()
    }

    /// Binds a zeroed launch workspace, if the kernel uses it.
    pub fn bind_workspace(&self, client: &Client, bindings: &mut KernelArguments) {
        if self.workspace {
            let zeros = [0u32; LAUNCH_WORKSPACE_WORDS];
            let workspace = client.create_from_slice(bytemuck::cast_slice(&zeros));
            bindings.push_hidden_buffer(workspace.binding());
        }
    }
}

/// Gives the capacity on a runtime that cannot query it.
///
/// A runtime that can query its capacity (CUDA, HIP) uses its own answer instead.
pub trait CapacityHint {
    /// How many cubes of `kernel` the device runs at the same time.
    fn capacity(client: &Client, kernel: &dyn CubeKernel) -> u32;

    /// Launches the persistent `kernel` with the capacity of this hint.
    fn launch(
        client: &Client,
        kernel: Box<dyn CubeKernel>,
        count: PersistentCount,
        bindings: KernelArguments,
    ) {
        let capacity = Self::capacity(client, kernel.as_ref());
        client.launch_persistent(kernel, count, capacity, bindings)
    }

    /// Launches the persistent `kernel` with the capacity of this hint, as an exclusive launch.
    ///
    /// # Safety
    ///
    /// See [`Client::launch_persistent_exclusive`].
    unsafe fn launch_exclusive(
        client: &Client,
        kernel: Box<dyn CubeKernel>,
        count: PersistentCount,
        bindings: KernelArguments,
    ) {
        let capacity = Self::capacity(client, kernel.as_ref());
        unsafe { client.launch_persistent_exclusive(kernel, count, capacity, bindings) }
    }
}

/// The capacity estimate of [`DefaultCapacity`] on a GPU. Large enough to fill most GPUs.
pub const DEFAULT_PERSISTENT_CUBES: u32 = 256;

/// One cube per core on a CPU, else [`DEFAULT_PERSISTENT_CUBES`].
pub struct DefaultCapacity;

impl CapacityHint for DefaultCapacity {
    fn capacity(client: &Client, _kernel: &dyn CubeKernel) -> u32 {
        client
            .properties()
            .hardware
            .num_cpu_cores
            .unwrap_or(DEFAULT_PERSISTENT_CUBES)
    }
}

/// Tunes the capacity on the first launches of each kernel, with real work.
///
/// Each of the first launches uses one of [`CANDIDATES`](Self::CANDIDATES), and is timed. The
/// later launches use the fastest. Because every tuning launch does the caller's work, the
/// kernel's buffers stay correct. A runtime that can query its capacity ignores the hint.
pub struct AutotunedCapacity;

impl AutotunedCapacity {
    /// The capacities that the tuning tries, each at most the maximum cube count.
    pub const CANDIDATES: [u32; 5] = [64, 128, 256, 512, 1024];
}

impl CapacityHint for AutotunedCapacity {
    fn capacity(client: &Client, kernel: &dyn CubeKernel) -> u32 {
        tuning(&kernel.id())
            .best()
            .unwrap_or(DefaultCapacity::capacity(client, kernel))
    }

    fn launch(
        client: &Client,
        kernel: Box<dyn CubeKernel>,
        count: PersistentCount,
        bindings: KernelArguments,
    ) {
        let id = kernel.id();
        let Some(candidate) = tuning(&id).next_candidate() else {
            let capacity = Self::capacity(client, kernel.as_ref());
            return client.launch_persistent(kernel, count, capacity, bindings);
        };
        let candidate = candidate.min(client.properties().hardware.max_cube_count.0);
        let name = kernel.name();
        let launch = || client.launch_persistent(kernel, count, candidate, bindings);
        let time = client
            .profile(launch, name)
            .ok()
            .and_then(|((), profile)| cubecl_environment::future::block_on(profile.resolve()))
            .map(|ticks| ticks.duration());
        TUNINGS.lock().entry(id).or_default().record(time);
    }
}

/// The timings of one kernel's tuning launches, by candidate.
#[derive(Clone, Default)]
struct Tuning {
    times: Vec<Option<Duration>>,
}

impl Tuning {
    fn next_candidate(&self) -> Option<u32> {
        AutotunedCapacity::CANDIDATES.get(self.times.len()).copied()
    }

    /// Records the time of the launch with the next candidate. `None` if it was not measured.
    fn record(&mut self, time: Option<Duration>) {
        if self.next_candidate().is_some() {
            self.times.push(time);
        }
    }

    /// The fastest candidate, once every candidate launched.
    fn best(&self) -> Option<u32> {
        self.next_candidate().is_none().then_some(())?;
        let fastest = self
            .times
            .iter()
            .enumerate()
            .filter_map(|(i, time)| time.map(|time| (i, time)))
            .min_by_key(|(_, time)| *time)?;
        Some(AutotunedCapacity::CANDIDATES[fastest.0])
    }
}

static TUNINGS: LazyLock<Mutex<HashMap<KernelId, Tuning>>> = LazyLock::new(Default::default);

fn tuning(id: &KernelId) -> Tuning {
    TUNINGS.lock().get(id).cloned().unwrap_or_default()
}

/// Read once, because reading the configuration takes a global lock.
static FILL_FRACTION: LazyLock<f32> =
    LazyLock::new(|| CubeClRuntimeConfig::get().persistent.fill_fraction);

fn checked_fraction(fraction: f32) -> f32 {
    if fraction > 0.0 && fraction <= 1.0 {
        fraction
    } else {
        log::warn!("Persistent fill fraction {fraction} is not in (0, 1]; using 1.0");
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fraction_scales_the_capacity() {
        assert_eq!(PersistentCount::Fraction(0.5).resolve(100, 1000), 50);
    }

    #[test]
    fn invalid_fraction_uses_the_full_capacity() {
        assert_eq!(PersistentCount::Fraction(0.0).resolve(100, 1000), 100);
        assert_eq!(PersistentCount::Fraction(1.5).resolve(100, 1000), 100);
    }

    #[test]
    fn count_is_at_least_one_and_at_most_max() {
        assert_eq!(PersistentCount::Fraction(0.001).resolve(100, 1000), 1);
        assert_eq!(PersistentCount::AtMost(500).resolve(400, 300), 300);
    }

    #[test]
    fn at_most_is_bounded_by_the_capacity() {
        assert_eq!(PersistentCount::AtMost(8).resolve(100, 1000), 8);
        assert_eq!(PersistentCount::AtMost(800).resolve(100, 1000), 100);
    }

    #[test]
    fn tuning_picks_the_fastest_measured_candidate() {
        let mut tuning = Tuning::default();
        for millis in [Some(5), Some(3), None, Some(4), Some(9)]
            .map(|m: Option<u64>| m.map(Duration::from_millis))
        {
            assert_eq!(tuning.best(), None);
            tuning.record(millis);
        }
        assert_eq!(tuning.next_candidate(), None);
        assert_eq!(tuning.best(), Some(AutotunedCapacity::CANDIDATES[1]));
    }

    #[test]
    fn exact_is_not_changed() {
        assert_eq!(PersistentCount::Exact(5000).resolve(100, 1000), 5000);
    }
}
