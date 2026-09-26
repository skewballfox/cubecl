use crate as cubecl;
use alloc::{string::String, vec, vec::Vec};

use cubecl::prelude::*;
use cubecl_ir::features::GridSync;
use cubecl_runtime::persistent::split::split;

/// Adds each input item to the output one time per visit, so an output equal to the input
/// proves that every item was visited exactly one time.
#[cube(launch, persistent)]
fn kernel_persistent_range(input: &[u32], output: &mut [u32], cube_count: &mut [u32]) {
    for item in persistent_range(input.len()) {
        if UNIT_POS == 0 {
            output[item] += input[item];
        }
    }
    if ABSOLUTE_POS == 0 {
        cube_count[0] = CUBE_COUNT as u32;
    }
}

#[cube(launch, persistent)]
fn kernel_persistent_range_units(input: &[u32], output: &mut [u32]) {
    for item in persistent_range_units(input.len()) {
        output[item] += input[item];
    }
}

const ITEMS: usize = 1000;

fn items() -> Vec<u32> {
    (1..=ITEMS as u32).collect()
}

/// Returns the output and the cube count that the kernel saw.
fn launch_persistent_range(client: &Client, count: PersistentCount) -> (Vec<u32>, u32) {
    let input = client.create_from_slice(u32::as_bytes(&items()));
    let output = client.create_from_slice(u32::as_bytes(&vec![0u32; ITEMS]));
    let cube_count = client.create_from_slice(u32::as_bytes(&[0u32]));

    kernel_persistent_range::launch_persistent(
        client,
        count,
        CubeDim::new_1d(4),
        unsafe { BufferArg::from_raw_parts(input, ITEMS) },
        unsafe { BufferArg::from_raw_parts(output.clone(), ITEMS) },
        unsafe { BufferArg::from_raw_parts(cube_count.clone(), 1) },
    );

    let output = client.read_one_unchecked(output);
    let cube_count = client.read_one_unchecked(cube_count);
    (
        u32::from_bytes(&output).to_vec(),
        u32::from_bytes(&cube_count)[0],
    )
}

pub fn test_persistent_range_visits_each_item_once(client: Client) {
    for count in [
        PersistentCount::Fill,
        PersistentCount::Fraction(0.5),
        PersistentCount::AtMost(1),
        PersistentCount::Exact(3),
    ] {
        let (output, _) = launch_persistent_range(&client, count);
        assert_eq!(output, items(), "{count:?}");
    }
}

pub fn test_persistent_count_reaches_the_kernel(client: Client) {
    assert_eq!(
        launch_persistent_range(&client, PersistentCount::AtMost(1)).1,
        1
    );
    assert_eq!(
        launch_persistent_range(&client, PersistentCount::Exact(3)).1,
        3
    );
}

pub fn test_persistent_range_units_visits_each_item_once(client: Client) {
    let input = client.create_from_slice(u32::as_bytes(&items()));
    let output = client.create_from_slice(u32::as_bytes(&vec![0u32; ITEMS]));

    kernel_persistent_range_units::launch_persistent(
        &client,
        PersistentCount::Exact(3),
        CubeDim::new_1d(4),
        unsafe { BufferArg::from_raw_parts(input, ITEMS) },
        unsafe { BufferArg::from_raw_parts(output.clone(), ITEMS) },
    );

    let output = client.read_one_unchecked(output);
    assert_eq!(u32::from_bytes(&output), items());
}

pub fn test_persistent_capacity(client: Client) {
    let buffer = || unsafe { BufferArg::from_raw_parts(client.empty(4), 1) };
    let capacity = kernel_persistent_range::capacity(
        &client,
        CubeDim::new_1d(4),
        buffer(),
        buffer(),
        buffer(),
    );

    // `None` is a valid answer: the runtime cannot query its capacity.
    assert!(!matches!(capacity, Ok(Some(0)) | Err(_)), "{capacity:?}");
}

#[cube(launch, persistent, create_dummy_kernel)]
fn kernel_sync_grid_not_cooperative(output: &mut [u32]) {
    sync_grid();
    output[0] = 1;
}

#[cube(launch, cooperative, create_dummy_kernel)]
fn kernel_sync_grid(output: &mut [u32]) {
    sync_grid();
    output[0] = 1;
}

/// Phase 0 writes every item, and phase 1 reads the item that another cube wrote.
#[cube(launch, cooperative)]
fn kernel_sync_grid_orders_phases(written: &mut [u32], output: &mut [u32]) {
    let len = written.len();
    for item in persistent_range_units(len) {
        written[item] = item as u32 + 1;
    }
    sync_grid();
    for item in persistent_range_units(len) {
        output[item] = written[(item + len / 2) % len];
    }
}

#[cube(launch, cooperative, create_dummy_kernel)]
fn kernel_value_in_registers_across_sync_grid(input: &[u32], output: &mut [u32]) {
    let loaded = input[0];
    sync_grid();
    output[0] = loaded;
}

#[cube(launch, cooperative, create_dummy_kernel)]
fn kernel_sync_grid_in_loop(output: &mut [u32]) {
    for _ in 0..CUBE_COUNT {
        sync_grid();
    }
    output[0] = 1;
}

pub fn test_sync_grid_orders_phases(client: Client) {
    if client.properties().features.grid_sync == GridSync::None {
        std::println!("grid sync not supported - skipped");
        return;
    }
    let written = client.empty(ITEMS * core::mem::size_of::<u32>());
    let output = client.empty(ITEMS * core::mem::size_of::<u32>());

    kernel_sync_grid_orders_phases::launch_persistent(
        &client,
        PersistentCount::Exact(4),
        CubeDim::new_1d(8),
        unsafe { BufferArg::from_raw_parts(written, ITEMS) },
        unsafe { BufferArg::from_raw_parts(output.clone(), ITEMS) },
    );

    let expected: Vec<u32> = (0..ITEMS)
        .map(|i| ((i + ITEMS / 2) % ITEMS) as u32 + 1)
        .collect();
    let output = client.read_one_unchecked(output);
    assert_eq!(u32::from_bytes(&output), expected);
}

const SHARED_UNITS: usize = 8;

/// Each unit reads, after the grid sync, the shared value that its neighbour wrote before it.
#[cube(launch, cooperative)]
fn kernel_shared_memory_survives_sync_grid(output: &mut [u32]) {
    let mut shared = Shared::<[u32]>::new_slice(SHARED_UNITS);
    let unit = UNIT_POS as usize;
    shared[unit] = CUBE_POS as u32 * 100 + UNIT_POS;
    sync_grid();
    output[ABSOLUTE_POS] = shared[(unit + 1) % SHARED_UNITS];
}

pub fn test_shared_memory_survives_sync_grid(client: Client) {
    if client.properties().features.grid_sync == GridSync::None {
        std::println!("grid sync not supported - skipped");
        return;
    }
    let cubes = 3;
    let len = cubes * SHARED_UNITS;
    let output = client.empty(len * core::mem::size_of::<u32>());

    kernel_shared_memory_survives_sync_grid::launch_persistent(
        &client,
        PersistentCount::Exact(cubes as u32),
        CubeDim::new_1d(SHARED_UNITS as u32),
        unsafe { BufferArg::from_raw_parts(output.clone(), len) },
    );

    let expected: Vec<u32> = (0..len)
        .map(|i| ((i / SHARED_UNITS) * 100 + (i + 1) % SHARED_UNITS) as u32)
        .collect();
    let output = client.read_one_unchecked(output);
    assert_eq!(u32::from_bytes(&output), expected);
}

#[cube(
    launch,
    cooperative,
    shared_after_grid_sync = "discard",
    create_dummy_kernel
)]
fn kernel_shared_memory_discarded(output: &mut [u32]) {
    let mut shared = Shared::<[u32]>::new_slice(SHARED_UNITS);
    shared[UNIT_POS as usize] = UNIT_POS;
    sync_grid();
    output[ABSOLUTE_POS] = shared[UNIT_POS as usize];
}

pub fn test_discard_needs_no_spill_buffer(client: Client) {
    let output = unsafe { BufferArg::from_raw_parts(client.empty(4), 1) };
    let kernel = kernel_shared_memory_discarded::create_dummy_kernel(
        client.properties_shared(),
        client.target_properties_shared(),
        CubeCount::new_single(),
        CubeDim::new_1d(SHARED_UNITS as u32),
        output,
    );
    let plan = split(&mut kernel.define(), 1).expect("the kernel splits");
    assert_eq!(plan.phases, 2);
    assert!(plan.spill_bytes_per_cube.is_empty());
}

/// The errors of splitting `kernel` at phase `phase`.
fn split_error(kernel: impl CubeKernel, phase: usize) -> String {
    let mut definition = kernel.define();
    split(&mut definition, phase)
        .map(|_| ())
        .expect_err("the split must fail")
}

pub fn test_split_refuses_a_value_in_registers(client: Client) {
    let buffer = || unsafe { BufferArg::from_raw_parts(client.empty(4), 1) };
    let kernel = kernel_value_in_registers_across_sync_grid::create_dummy_kernel(
        client.properties_shared(),
        client.target_properties_shared(),
        CubeCount::new_single(),
        CubeDim::new_1d(1),
        buffer(),
        buffer(),
    );
    assert!(split_error(kernel, 1).contains("cannot be computed again"));
}

pub fn test_split_refuses_sync_grid_in_a_loop(client: Client) {
    let output = unsafe { BufferArg::from_raw_parts(client.empty(4), 1) };
    let kernel = kernel_sync_grid_in_loop::create_dummy_kernel(
        client.properties_shared(),
        client.target_properties_shared(),
        CubeCount::new_single(),
        CubeDim::new_1d(1),
        output,
    );
    assert!(split_error(kernel, 0).contains("inside a branch or a loop"));
}

/// The errors that expanding `kernel` pushes to its scope. A compiler refuses a kernel with any.
fn expansion_errors(kernel: impl CubeKernel) -> Vec<String> {
    kernel.define().body.pop_errors()
}

pub fn test_sync_grid_needs_a_cooperative_kernel(client: Client) {
    let output = unsafe { BufferArg::from_raw_parts(client.empty(4), 1) };
    let kernel = kernel_sync_grid_not_cooperative::create_dummy_kernel(
        client.properties_shared(),
        client.target_properties_shared(),
        CubeCount::new_single(),
        CubeDim::new_1d(1),
        output,
    );

    let errors = expansion_errors(kernel);
    assert!(
        errors.iter().any(|e| e.contains("`cooperative`")),
        "{errors:?}"
    );
}

pub fn test_sync_grid_needs_runtime_support(client: Client) {
    let output = unsafe { BufferArg::from_raw_parts(client.empty(4), 1) };
    let kernel = kernel_sync_grid::create_dummy_kernel(
        client.properties_shared(),
        client.target_properties_shared(),
        CubeCount::new_single(),
        CubeDim::new_1d(1),
        output,
    );

    let errors = expansion_errors(kernel);
    let supported = client.properties().features.grid_sync != GridSync::None;
    let refused = errors
        .iter()
        .any(|e| e.contains("not supported on this runtime"));
    assert_eq!(refused, !supported, "{errors:?}");
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_persistent {
    () => {
        mod persistent {
            use super::*;

            $crate::testgen_persistent!(
                @tests
                test_persistent_range_visits_each_item_once,
                test_persistent_count_reaches_the_kernel,
                test_persistent_range_units_visits_each_item_once,
                test_persistent_capacity,
                test_sync_grid_orders_phases,
                test_shared_memory_survives_sync_grid,
                test_discard_needs_no_spill_buffer,
                test_split_refuses_a_value_in_registers,
                test_split_refuses_sync_grid_in_a_loop,
                test_sync_grid_needs_a_cooperative_kernel,
                test_sync_grid_needs_runtime_support
            );
        }
    };
    (@tests $($name:ident),*) => {
        $(
            #[$crate::runtime_tests::test_log::test]
            fn $name() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::persistent::$name(client);
            }
        )*
    };
}
