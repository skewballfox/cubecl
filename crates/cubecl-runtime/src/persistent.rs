//! The capacity is how many cubes of one compiled kernel the device runs at the same time.

use crate::{
    client::Client,
    config::{CubeClRuntimeConfig, RuntimeConfig},
    kernel::CubeKernel,
};
use cubecl_environment::sync::LazyLock;

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

/// Gives the capacity on a runtime that cannot query it.
///
/// A runtime that can query its capacity (CUDA, HIP) uses its own answer instead.
pub trait CapacityHint {
    /// How many cubes of `kernel` the device runs at the same time.
    fn capacity(client: &Client, kernel: &dyn CubeKernel) -> u32;
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
    fn exact_is_not_changed() {
        assert_eq!(PersistentCount::Exact(5000).resolve(100, 1000), 5000);
    }
}
