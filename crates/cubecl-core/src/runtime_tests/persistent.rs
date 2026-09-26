use crate as cubecl;
use alloc::{string::String, vec, vec::Vec};

use cubecl::prelude::*;
use cubecl_common::bytes::Bytes;
use cubecl_ir::features::{GridSync, GridSyncEmulation};
use cubecl_runtime::{
    persistent::split::split,
    server::{Handle, ResourceLimitError},
};

use super::launch::resource_error;

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

/// Every tuning launch does the caller's work, and the launches after the tuning use one of the
/// candidates.
pub fn test_autotuned_capacity(client: Client) {
    let input = client.create_from_slice(u32::as_bytes(&items()));
    let output = client.create_from_slice(u32::as_bytes(&vec![0u32; ITEMS]));
    let cube_count = client.create_from_slice(u32::as_bytes(&[0u32]));
    let launches = AutotunedCapacity::CANDIDATES.len() as u32 + 1;
    for _ in 0..launches {
        kernel_persistent_range::launch_persistent_with::<AutotunedCapacity>(
            &client,
            PersistentCount::Fill,
            CubeDim::new_1d(4),
            unsafe { BufferArg::from_raw_parts(input.clone(), ITEMS) },
            unsafe { BufferArg::from_raw_parts(output.clone(), ITEMS) },
            unsafe { BufferArg::from_raw_parts(cube_count.clone(), 1) },
        );
    }

    let expected: Vec<u32> = items().iter().map(|item| item * launches).collect();
    assert_eq!(
        u32::from_bytes(&client.read_one_unchecked(output)),
        expected
    );
    let max = client.properties().hardware.max_cube_count.0;
    let tuned = u32::from_bytes(&client.read_one_unchecked(cube_count))[0];
    let buffer = || unsafe { BufferArg::from_raw_parts(client.empty(4), 1) };
    let queried = kernel_persistent_range::capacity(
        &client,
        CubeDim::new_1d(4),
        buffer(),
        buffer(),
        buffer(),
    )
    .expect("the kernel compiles");
    match queried {
        // A runtime that can query its capacity ignores the hint.
        Some(capacity) => assert_eq!(tuned, capacity.min(max)),
        None => {
            let candidates = AutotunedCapacity::CANDIDATES.map(|c| c.min(max));
            assert!(candidates.contains(&tuned), "{tuned}");
        }
    }
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

/// Takes items from the launch's work queue. Cube 0 exits early when `skip_cube_0` is set.
#[cube(launch, persistent)]
fn kernel_work_queue(input: &[u32], output: &mut [u32], #[comptime] skip_cube_0: bool) {
    if skip_cube_0 && CUBE_POS == 0 {
        terminate!();
    }
    let total = input.len() as u32;
    let mut item = next_work_item();
    while item < total {
        if UNIT_POS == 0 {
            output[item as usize] += input[item as usize];
        }
        item = next_work_item();
    }
}

/// The runtime resets the queue before each launch, so every launch adds each item one time.
pub fn test_work_queue_resets_every_launch(client: Client) {
    let launches = 20;
    for skip_cube_0 in [false, true] {
        let input = client.create_from_slice(u32::as_bytes(&items()));
        let output = client.create_from_slice(u32::as_bytes(&vec![0u32; ITEMS]));
        for _ in 0..launches {
            kernel_work_queue::launch_persistent(
                &client,
                PersistentCount::Exact(4),
                CubeDim::new_1d(4),
                unsafe { BufferArg::from_raw_parts(input.clone(), ITEMS) },
                unsafe { BufferArg::from_raw_parts(output.clone(), ITEMS) },
                skip_cube_0,
            );
        }

        let expected: Vec<u32> = items().iter().map(|item| item * launches).collect();
        let output = client.read_one_unchecked(output);
        assert_eq!(
            u32::from_bytes(&output),
            expected,
            "skip_cube_0 = {skip_cube_0}"
        );
    }
}

/// In each round, every cube increments its own value, then reads its neighbour's. A loop with
/// `sync_grid` inside needs a spin barrier: a split cannot express it.
#[cube(launch, cooperative, grid_sync_emulation = "spin")]
fn kernel_spin_rounds(values: &mut [u32], seen: &mut [u32], rounds: u32) {
    let cube = CUBE_POS;
    for _ in 0..rounds {
        if UNIT_POS == 0 {
            values[cube] += 1;
        }
        sync_grid();
        if UNIT_POS == 0 {
            seen[cube] = values[(cube + 1) % CUBE_COUNT];
        }
        sync_grid();
    }
}

pub fn test_spin_grid_sync_in_a_loop(client: Client) {
    let spin = match client.properties().features.grid_sync {
        GridSync::Emulated(emulations) => emulations.contains(GridSyncEmulation::Spin),
        _ => false,
    };
    if !spin {
        std::println!("spin grid sync not supported - skipped");
        return;
    }
    let cubes = 2;
    let rounds = 50;
    let values = client.create_from_slice(u32::as_bytes(&vec![0u32; cubes]));
    let seen = client.create_from_slice(u32::as_bytes(&vec![0u32; cubes]));

    // Two cubes of four units run at the same time on any device this test runs on.
    unsafe {
        kernel_spin_rounds::launch_persistent(
            &client,
            PersistentCount::Exact(cubes as u32),
            CubeDim::new_1d(4),
            BufferArg::from_raw_parts(values.clone(), cubes),
            BufferArg::from_raw_parts(seen.clone(), cubes),
            rounds,
        )
    };

    let expected = vec![rounds; cubes];
    assert_eq!(
        u32::from_bytes(&client.read_one_unchecked(values)),
        expected
    );
    assert_eq!(u32::from_bytes(&client.read_one_unchecked(seen)), expected);
}

/// Phase 1: each cube sums its items into `partials[CUBE_POS]`. Phase 2: each cube sums the
/// partials of all cubes into `sums[CUBE_POS]`. `cubes` gets the cube count.
#[allow(clippy::needless_range_loop)]
#[cube(launch, cooperative)]
fn kernel_two_phase_sum(input: &[u32], partials: &mut [u32], sums: &mut [u32], cubes: &mut [u32]) {
    if UNIT_POS == 0 {
        let mut sum = 0u32;
        for item in persistent_range(input.len()) {
            sum += input[item];
        }
        partials[CUBE_POS] = sum;
    }
    sync_grid();
    if UNIT_POS == 0 {
        let mut sum = 0u32;
        for cube in 0..CUBE_COUNT {
            sum += partials[cube];
        }
        sums[CUBE_POS] = sum;
    }
    if ABSOLUTE_POS == 0 {
        cubes[0] = CUBE_COUNT as u32;
    }
}

const SUM_UNITS: u32 = 32;
/// The largest cube count of the sum tests that do not query the capacity.
const SUM_CUBES: u32 = 64;
const TOTAL: u32 = (ITEMS * (ITEMS + 1) / 2) as u32;

/// The output buffers of one launch of [`kernel_two_phase_sum`].
struct TwoPhaseSum {
    input: Handle,
    sums: Handle,
    cubes: Handle,
}

impl TwoPhaseSum {
    /// Buffers for at most `max_cubes` cubes. The outputs start at zero, so a launch that does
    /// not run gives no sums.
    fn new(client: &Client, max_cubes: u32) -> Self {
        Self {
            input: client.create_from_slice(u32::as_bytes(&items())),
            sums: client.create_from_slice(u32::as_bytes(&vec![0u32; max_cubes as usize])),
            cubes: client.create_from_slice(u32::as_bytes(&[0u32])),
        }
    }

    fn launch(&self, client: &Client, count: PersistentCount, partials: &Handle) {
        let len = |handle: &Handle| handle.size() as usize / core::mem::size_of::<u32>();
        kernel_two_phase_sum::launch_persistent(
            client,
            count,
            CubeDim::new_1d(SUM_UNITS),
            unsafe { BufferArg::from_raw_parts(self.input.clone(), ITEMS) },
            unsafe { BufferArg::from_raw_parts(partials.clone(), len(partials)) },
            unsafe { BufferArg::from_raw_parts(self.sums.clone(), len(&self.sums)) },
            unsafe { BufferArg::from_raw_parts(self.cubes.clone(), 1) },
        );
    }

    /// The sum that each cube computed.
    fn read(&self, client: &Client) -> Vec<u32> {
        let cubes = client.read_one_unchecked(self.cubes.clone());
        let cubes = u32::from_bytes(&cubes)[0] as usize;
        let sums = client.read_one_unchecked(self.sums.clone());
        u32::from_bytes(&sums)[..cubes].to_vec()
    }
}

/// Launches [`kernel_two_phase_sum`] with at most `max_cubes` cubes, and returns the sum that
/// each cube computed.
fn two_phase_sum(client: &Client, count: PersistentCount, max_cubes: u32) -> Vec<u32> {
    let sum = TwoPhaseSum::new(client, max_cubes);
    let partials = client.empty(max_cubes as usize * core::mem::size_of::<u32>());
    sum.launch(client, count, &partials);
    sum.read(client)
}

fn two_phase_sum_capacity(client: &Client) -> Option<u32> {
    let buffer = || unsafe { BufferArg::from_raw_parts(client.empty(4), 1) };
    kernel_two_phase_sum::capacity(
        client,
        CubeDim::new_1d(SUM_UNITS),
        buffer(),
        buffer(),
        buffer(),
        buffer(),
    )
    .expect("the kernel compiles")
}

fn assert_every_cube_sums_all_items(sums: &[u32]) {
    assert!(!sums.is_empty(), "the kernel did not run");
    assert!(sums.iter().all(|sum| *sum == TOTAL), "{sums:?}");
}

/// Whether the runtime has native grid sync. Prints why the calling test is skipped if not.
fn native_grid_sync(client: &Client) -> bool {
    let native = client.properties().features.grid_sync == GridSync::Native;
    if !native {
        std::println!("native grid sync not supported - skipped");
    }
    native
}

pub fn test_sync_grid_two_phase_sum(client: Client) {
    if client.properties().features.grid_sync == GridSync::None {
        std::println!("grid sync not supported - skipped");
        return;
    }
    let sums = two_phase_sum(&client, PersistentCount::AtMost(SUM_CUBES), SUM_CUBES);
    assert_every_cube_sums_all_items(&sums);
}

/// Detects a deadlock, or state that one launch leaves for the next (P1, P5).
pub fn test_sync_grid_two_phase_sum_repeated(client: Client) {
    if client.properties().features.grid_sync == GridSync::None {
        std::println!("grid sync not supported - skipped");
        return;
    }
    for launch in 0..1000 {
        let sums = two_phase_sum(&client, PersistentCount::AtMost(SUM_CUBES), SUM_CUBES);
        assert!(
            sums.iter().all(|sum| *sum == TOTAL),
            "launch {launch}: {sums:?}"
        );
        assert!(!sums.is_empty(), "launch {launch} did not run");
    }
}

const LOOP_CUBES: u32 = 4;
const LOOP_ROUNDS: u32 = 10;

/// In each round, every cube takes the value of the next cube and adds one. The first grid sync
/// separates the reads of a round from its writes. The second separates the writes from the
/// reads of the next round.
#[cube(launch, cooperative)]
fn kernel_sync_grid_in_uniform_loop(values: &mut [u32], rounds: u32) {
    let next = (CUBE_POS + 1) % CUBE_COUNT;
    for _ in 0..rounds {
        let mut value = 0u32;
        if UNIT_POS == 0 {
            value = values[next];
        }
        sync_grid();
        if UNIT_POS == 0 {
            values[CUBE_POS] = value + 1;
        }
        sync_grid();
    }
}

/// Native grid sync permits a grid sync in a `for` loop with uniform bounds. Split does not (S1).
pub fn test_sync_grid_in_uniform_loop(client: Client) {
    if !native_grid_sync(&client) {
        return;
    }
    let initial: Vec<u32> = (0..LOOP_CUBES).map(|cube| cube * 1000).collect();
    let values = client.create_from_slice(u32::as_bytes(&initial));

    kernel_sync_grid_in_uniform_loop::launch_persistent(
        &client,
        PersistentCount::Exact(LOOP_CUBES),
        CubeDim::new_1d(8),
        unsafe { BufferArg::from_raw_parts(values.clone(), LOOP_CUBES as usize) },
        LOOP_ROUNDS,
    );

    let expected: Vec<u32> = (0..LOOP_CUBES)
        .map(|cube| initial[((cube + LOOP_ROUNDS) % LOOP_CUBES) as usize] + LOOP_ROUNDS)
        .collect();
    let values = client.read_one_unchecked(values);
    assert_eq!(u32::from_bytes(&values), expected);
}

/// V4: the launch fails before the driver call. The capacity is the driver occupancy per SM
/// times the SM count, and the driver runs exactly that many cubes of the kernel.
pub fn test_sync_grid_exact_above_capacity_fails(client: Client) {
    if !native_grid_sync(&client) {
        return;
    }
    let capacity = two_phase_sum_capacity(&client).expect("native grid sync has a capacity");
    let sms = client
        .properties()
        .hardware
        .num_streaming_multiprocessors
        .expect("native grid sync runs on a GPU");
    assert!(
        capacity > 0 && capacity.is_multiple_of(sms),
        "{capacity} for {sms} SMs"
    );

    let sums = two_phase_sum(&client, PersistentCount::Exact(capacity), capacity);
    assert_eq!(sums.len(), capacity as usize);
    assert_every_cube_sums_all_items(&sums);

    let sum = TwoPhaseSum::new(&client, capacity + 1);
    let partials = client.empty((capacity as usize + 1) * core::mem::size_of::<u32>());
    sum.launch(&client, PersistentCount::Exact(capacity + 1), &partials);
    let error = resource_error(&client, sum.sums);
    assert!(
        matches!(
            error,
            ResourceLimitError::CooperativeGrid { requested, max, .. }
                if requested == capacity + 1 && max == capacity
        ),
        "{error:?}"
    );
}

/// With the default `fill_fraction` of `1.0`, `Fill` launches exactly the capacity.
pub fn test_fill_gives_the_capacity(client: Client) {
    if client.properties().features.grid_sync == GridSync::None {
        std::println!("grid sync not supported - skipped");
        return;
    }
    let Some(capacity) = two_phase_sum_capacity(&client) else {
        std::println!("capacity query not supported - skipped");
        return;
    };
    let sums = two_phase_sum(&client, PersistentCount::Fill, capacity);
    assert_eq!(sums.len(), capacity as usize);
    assert_every_cube_sums_all_items(&sums);
}

/// Replays a captured cooperative launch (P9).
///
/// Not in `testgen_persistent!`: while a graph capture records, the driver refuses allocations
/// on every other stream of the process. `cubecl-cuda/tests/graph.rs` runs this test alone.
pub fn test_sync_grid_graph_replay(client: Client) {
    if !native_grid_sync(&client) {
        return;
    }
    if let Err(error) = client.graph_prepare() {
        std::println!("graph capture not supported ({error}) - skipped");
        return;
    }
    let sum = TwoPhaseSum::new(&client, SUM_CUBES);
    let partials = client.empty(SUM_CUBES as usize * core::mem::size_of::<u32>());
    let count = PersistentCount::AtMost(SUM_CUBES);

    // The warmup run compiles the kernel and allocates, so the capture does neither.
    sum.launch(&client, count, &partials);
    assert_every_cube_sums_all_items(&sum.read(&client));

    client.start_capture().expect("start_capture");
    sum.launch(&client, count, &partials);
    let graph = client.stop_capture().expect("stop_capture");

    // New input between the replays, so each replay must run to give the new sums.
    for scale in [2, 3] {
        let input: Vec<u32> = items().iter().map(|item| item * scale).collect();
        client.write(
            &sum.input,
            Bytes::from_bytes_vec(u32::as_bytes(&input).to_vec()),
        );
        unsafe { graph.replay() }.expect("replay");
        let sums = sum.read(&client);
        assert!(!sums.is_empty());
        assert!(sums.iter().all(|s| *s == TOTAL * scale), "{sums:?}");
    }
}

#[cube(launch, cooperative)]
fn kernel_sync_grid_in_unit_branch(output: &mut [u32]) {
    if UNIT_POS == 0 {
        sync_grid();
    }
    output[ABSOLUTE_POS] = 1;
}

/// V2: the compiler refuses a grid sync that only some units reach.
pub fn test_sync_grid_in_unit_branch_fails_to_compile(client: Client) {
    if !native_grid_sync(&client) {
        return;
    }
    let output = client.empty(8 * core::mem::size_of::<u32>());

    kernel_sync_grid_in_unit_branch::launch_persistent(
        &client,
        PersistentCount::Exact(1),
        CubeDim::new_1d(8),
        unsafe { BufferArg::from_raw_parts(output.clone(), 8) },
    );

    let error = client
        .read_one(output)
        .expect_err("the kernel must not compile");
    let error = std::format!("{error:?}");
    assert!(error.contains("must be reached by every unit"), "{error}");
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
                test_autotuned_capacity,
                test_work_queue_resets_every_launch,
                test_spin_grid_sync_in_a_loop,
                test_sync_grid_orders_phases,
                test_sync_grid_two_phase_sum,
                test_sync_grid_two_phase_sum_repeated,
                test_sync_grid_in_uniform_loop,
                test_sync_grid_exact_above_capacity_fails,
                test_fill_gives_the_capacity,
                test_sync_grid_in_unit_branch_fails_to_compile,
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
