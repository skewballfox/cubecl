use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube(launch, persistent, grid_sync_emulation = "spin")]
fn persistent_kernel(output: &mut [u32]) {
    output[0] = 1;
}

fn main() {}
