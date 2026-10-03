//! A device for compiler tests that run without one.

use alloc::string::ToString;
use cubecl_ir::{
    DeviceIdentity, DeviceProperties, HardwareProperties, MemoryDeviceProperties, VectorSize,
    features::Features,
};

use crate::profile::TimingMethod;

/// The properties of a device with planes of `plane_dim` units and no type registered. A compiler
/// test registers the types its target supports.
#[must_use]
pub fn offline_device_properties(plane_dim: u32) -> DeviceProperties {
    let hardware = HardwareProperties {
        load_width: 128,
        vector_register_count: None,
        plane_size_min: plane_dim,
        plane_size_max: plane_dim,
        max_bindings: 32,
        max_shared_memory_size: 65536,
        max_cube_count: (u32::MAX, u32::from(u16::MAX), u32::from(u16::MAX)),
        max_units_per_cube: 1024,
        max_cube_dim: (1024, 1024, 1024),
        num_streaming_multiprocessors: None,
        num_tensor_cores: None,
        min_tensor_cores_dim: None,
        num_cpu_cores: None,
        last_level_cache_size: None,
        max_vector_size: VectorSize::MAX,
        cube_mma_reserved_shared_memory: 0,
    };
    DeviceProperties::new(
        Features::default(),
        MemoryDeviceProperties::new(u64::MAX, 256),
        hardware,
        TimingMethod::Device,
        DeviceIdentity {
            name: "offline".to_string(),
            fingerprint: "offline".to_string(),
            physical: None,
        },
    )
}
