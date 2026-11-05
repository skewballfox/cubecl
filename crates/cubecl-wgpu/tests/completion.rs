//! A deferred mark on wgpu leaves the queued work in the encoder, so the token
//! cannot be done until something submits it. This makes "not done before the
//! wait" a fact here, where on other runtimes it is a race.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_server::runtime::Runtime;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

#[cube(launch)]
fn add_one(values: &mut [u32]) {
    values[ABSOLUTE_POS] += 1;
}

fn launch(client: &Client, values: &cubecl_server::server::Handle) {
    add_one::launch(client, CubeCount::new_1d(1), CubeDim::new_1d(4), unsafe {
        BufferArg::from_raw_parts(values.clone(), 4)
    });
}

#[test]
fn a_deferred_mark_is_done_only_after_a_submission() {
    let client = <WgpuRuntime>::client(&WgpuDevice::default());
    let values = client.create_from_slice(u32::as_bytes(&[0; 4]));

    // Compile the kernel and drain the queue first, so the launch below is the
    // only task in the encoder.
    launch(&client, &values);
    cubecl_environment::future::block_on(client.sync()).unwrap();

    launch(&client, &values);
    let token = client.mark_deferred();
    assert!(!token.is_done(), "nothing submitted the launch yet");

    // The wait flushes the stream of the token, then waits.
    cubecl_environment::future::block_on(client.wait(&token)).unwrap();
    assert!(token.is_done());

    let values = u32::from_bytes(&client.read_one_unchecked(values)).to_vec();
    assert_eq!(values, [2; 4]);
}

#[test]
fn a_mark_now_submits() {
    let client = <WgpuRuntime>::client(&WgpuDevice::default());
    let values = client.create_from_slice(u32::as_bytes(&[0; 4]));

    launch(&client, &values);
    let token = client.mark();
    // No flush and no wait: the mark submitted the launch, and the poll thread
    // completes the token.
    let start = std::time::Instant::now();
    while !token.is_done() {
        assert!(
            start.elapsed() < std::time::Duration::from_secs(30),
            "the token did not complete without a wait"
        );
        std::thread::yield_now();
    }
}
