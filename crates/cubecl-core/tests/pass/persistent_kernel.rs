//! The launch functions that `persistent` and `cooperative` generate, and their signatures.
use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube(launch, launch_unchecked, persistent)]
fn persistent_kernel<F: Float>(input: &[F], output: &mut [F]) {
    for item in persistent_range(input.len()) {
        output[item] = input[item];
    }
}

#[cube(launch, cooperative)]
fn cooperative_kernel(output: &mut [u32]) {
    sync_grid();
    output[0] = 1;
}

fn main() {
    let _ = persistent_kernel::launch::<f32>;
    let _ = persistent_kernel::launch_persistent::<f32>;
    let _ = persistent_kernel::launch_persistent_with::<f32, DefaultCapacity>;
    let _ = persistent_kernel::launch_persistent_unchecked::<f32>;
    let _ = persistent_kernel::capacity::<f32>;
    let _ = cooperative_kernel::launch_persistent;
    let _ = cooperative_kernel::capacity;
}
