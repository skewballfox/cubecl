use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube(launch, cooperative)]
fn cooperative_kernel(output: &mut [u32]) {
    sync_grid();
    output[0] = 1;
}

fn launch(client: &Client, output: BufferArg) {
    cooperative_kernel::launch_persistent_exclusive(
        client,
        PersistentCount::Fill,
        CubeDim::new_1d(1),
        output,
    );
}

fn main() {}
