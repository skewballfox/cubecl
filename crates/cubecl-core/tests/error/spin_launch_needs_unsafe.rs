use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube(launch, cooperative, grid_sync_emulation = "spin")]
fn spin_kernel(output: &mut [u32]) {
    sync_grid();
    output[0] = 1;
}

fn launch(client: &Client, output: BufferArg) {
    spin_kernel::launch_persistent(client, PersistentCount::Fill, CubeDim::new_1d(1), output);
}

fn main() {}
