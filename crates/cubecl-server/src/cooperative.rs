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
use cubecl_ir::{
    PciAddress,
    features::{Features, GridSync, GridSyncEmulation},
    settings::Persistence,
};
use cubecl_runtime::{
    id::KernelId,
    persistent::PersistentCount,
    server::{CubeCount, LaunchError, ResourceLimitError},
};

/// What a runtime knows about other work on a device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceSharing {
    /// The device is part of the CPU package. It also draws the display.
    pub integrated: bool,
    /// The driver stops a kernel that runs too long, because a display uses the device.
    pub watchdog: bool,
    /// Whether a display is connected to the device, or `None` if the runtime cannot tell.
    pub display: Option<bool>,
}

impl DeviceSharing {
    /// Whether other work can use the compute units of the device during a launch. A device that
    /// the runtime cannot check counts as shared.
    pub fn is_shared(&self) -> bool {
        self.integrated || self.watchdog || self.display != Some(false)
    }
}

/// Sets the grid sync of a device that supports a cooperative launch. A shared device reports
/// Split, so only an exclusive launch gets native grid sync.
pub fn set_native_grid_sync(features: &mut Features, sharing: DeviceSharing) {
    features.exclusive_grid_sync = true;
    features.grid_sync = if sharing.is_shared() {
        log::info!("Other work shares the device ({sharing:?}): grid sync uses Split by default");
        GridSync::Emulated(GridSyncEmulation::Split.into())
    } else {
        GridSync::Native
    };
}

/// Whether a display is connected to the device at `pci`, from the DRM connectors in sysfs.
/// `None` if the address is unknown, or off Linux.
pub fn display_connected(pci: Option<PciAddress>) -> Option<bool> {
    #[cfg(all(feature = "std", target_os = "linux"))]
    {
        drm::display_connected(std::path::Path::new("/sys/class/drm"), pci?)
    }
    #[cfg(not(all(feature = "std", target_os = "linux")))]
    {
        let _ = pci;
        None
    }
}

#[cfg(all(feature = "std", target_os = "linux"))]
mod drm {
    use super::PciAddress;
    use alloc::string::ToString;
    use std::{fs, path::Path};

    /// Finds the DRM card of the device at `pci` in `root`, and reads the status of each of its
    /// connectors. A device without a card has no display.
    pub(super) fn display_connected(root: &Path, pci: PciAddress) -> Option<bool> {
        let pci = pci.to_string();
        for card in fs::read_dir(root).ok()?.flatten() {
            let name = card.file_name();
            let name = name.to_string_lossy();
            // `card1` is a card; `card1-eDP-1` is one of its connectors.
            if !name.starts_with("card") || name.contains('-') {
                continue;
            }
            let device = fs::canonicalize(card.path().join("device")).ok()?;
            if device
                .file_name()
                .is_some_and(|id| id.to_string_lossy() == pci)
            {
                return Some(connected(&card.path(), &name));
            }
        }
        Some(false)
    }

    fn connected(card: &Path, name: &str) -> bool {
        let Ok(entries) = fs::read_dir(card) else {
            return false;
        };
        entries.flatten().any(|connector| {
            connector
                .file_name()
                .to_string_lossy()
                .starts_with(&alloc::format!("{name}-"))
                && fs::read_to_string(connector.path().join("status"))
                    .is_ok_and(|status| status.trim() == "connected")
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::{os::unix::fs::symlink, path::PathBuf};

        /// A fake `/sys/class/drm` with one card at `0000:07:00.0` and the given connectors.
        fn sysfs(test: &str, connectors: &[(&str, &str)]) -> PathBuf {
            let root = std::env::temp_dir().join(alloc::format!("cubecl-drm-{test}"));
            let _ = fs::remove_dir_all(&root);
            let device = root.join("devices/0000:07:00.0");
            let card = root.join("class/card1");
            fs::create_dir_all(&device).unwrap();
            fs::create_dir_all(&card).unwrap();
            symlink(&device, card.join("device")).unwrap();
            for (connector, status) in connectors {
                let dir = card.join(alloc::format!("card1-{connector}"));
                fs::create_dir_all(&dir).unwrap();
                fs::write(dir.join("status"), alloc::format!("{status}\n")).unwrap();
            }
            root.join("class")
        }

        fn pci(text: &str) -> PciAddress {
            text.parse().unwrap()
        }

        #[test]
        fn a_connected_connector_is_a_display() {
            let root = sysfs(
                "connected",
                &[("DP-1", "disconnected"), ("eDP-1", "connected")],
            );
            assert_eq!(display_connected(&root, pci("0000:07:00.0")), Some(true));
        }

        #[test]
        fn disconnected_connectors_are_no_display() {
            let root = sysfs("disconnected", &[("HDMI-A-1", "disconnected")]);
            assert_eq!(display_connected(&root, pci("0000:07:00.0")), Some(false));
        }

        #[test]
        fn a_device_without_a_card_has_no_display() {
            let root = sysfs("other", &[("eDP-1", "connected")]);
            assert_eq!(display_connected(&root, pci("0000:08:00.0")), Some(false));
        }
    }
}

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
    fn a_device_is_shared_unless_every_signal_says_otherwise() {
        let headless = DeviceSharing {
            display: Some(false),
            ..Default::default()
        };
        assert!(!headless.is_shared());
        for shared in [
            DeviceSharing {
                integrated: true,
                ..headless
            },
            DeviceSharing {
                watchdog: true,
                ..headless
            },
            DeviceSharing {
                display: Some(true),
                ..headless
            },
            DeviceSharing {
                display: None,
                ..headless
            },
        ] {
            assert!(shared.is_shared(), "{shared:?}");
        }
    }

    #[test]
    fn a_shared_device_splits_by_default() {
        let mut features = Features::default();
        set_native_grid_sync(&mut features, DeviceSharing::default());
        assert_eq!(
            features.grid_sync,
            GridSync::Emulated(GridSyncEmulation::Split.into())
        );
        assert!(features.exclusive_grid_sync);

        let headless = DeviceSharing {
            display: Some(false),
            ..Default::default()
        };
        set_native_grid_sync(&mut features, headless);
        assert_eq!(features.grid_sync, GridSync::Native);
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
