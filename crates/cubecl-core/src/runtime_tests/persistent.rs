use crate as cubecl;
use alloc::{string::String, vec, vec::Vec};

use cubecl::prelude::*;
use cubecl_ir::features::GridSync;

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

            #[$crate::runtime_tests::test_log::test]
            fn test_persistent_range_visits_each_item_once() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::persistent::test_persistent_range_visits_each_item_once(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_persistent_count_reaches_the_kernel() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::persistent::test_persistent_count_reaches_the_kernel(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_persistent_range_units_visits_each_item_once() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::persistent::test_persistent_range_units_visits_each_item_once(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_sync_grid_needs_a_cooperative_kernel() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::persistent::test_sync_grid_needs_a_cooperative_kernel(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_sync_grid_needs_runtime_support() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::persistent::test_sync_grid_needs_runtime_support(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_persistent_capacity() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::persistent::test_persistent_capacity(client);
            }
        }
    };
}
