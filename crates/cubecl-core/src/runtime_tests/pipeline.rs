use crate::{self as cubecl, as_bytes, Feature};
use cubecl::prelude::*;
use pipeline::Pipeline;

#[cube(launch)]
/// Calculate the sum of an array, using pipelining
fn pipelined_sum<F: Float>(
    input: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] batch_len: u32,
) {
    let smem_size = 2 * batch_len;
    let num_batches = input.len() / batch_len;
    let mut shared_memory = SharedMemory::<F>::new_lined(smem_size, input.line_size());
    let pipeline = Pipeline::new(2);

    let mut sum = Line::<F>::empty(input.line_size()).fill(F::new(0.));

    // Copy the first batch to shared memory
    pipeline.producer_acquire();
    pipeline.memcpy_async(
        input.slice(0, batch_len),
        shared_memory.slice_mut(0, batch_len),
    );
    pipeline.producer_commit();

    for input_batch in 1..num_batches {
        // Copy and compute index always alternate
        let copy_index = input_batch % 2;
        let compute_index = (input_batch + 1) % 2;

        // Copy the next batch to shared memory
        pipeline.producer_acquire();
        pipeline.memcpy_async(
            input.slice(batch_len * input_batch, batch_len * (input_batch + 1)),
            shared_memory.slice_mut(batch_len * copy_index, batch_len * (copy_index + 1)),
        );
        pipeline.producer_commit();

        // Compute the batch that is ready
        pipeline.consumer_wait();
        let compute_slice =
            shared_memory.slice(batch_len * compute_index, batch_len * (compute_index + 1));
        for i in 0..batch_len {
            sum += compute_slice[i];
        }
        pipeline.consumer_release();
    }

    // Compute the last batch
    pipeline.consumer_wait();
    let compute_slice = shared_memory.slice(
        batch_len * ((num_batches + 1) % 2),
        batch_len * ((num_batches + 1) % 2 + 1),
    );
    for i in 0..batch_len {
        sum += compute_slice[i];
    }
    pipeline.consumer_release();

    output[0] = sum;
}

#[cube(launch)]
pub fn async_copy_test<F: Float>(input: &Array<Line<F>>, output: &mut Array<Line<F>>) {
    let pipeline = pipeline::Pipeline::<F>::new(2);
    let mut smem = SharedMemory::<F>::new_lined(1u32, 1u32);

    if UNIT_POS == 0 {
        let source = input.slice(2, 3);
        let destination = smem.slice_mut(0, 1);

        pipeline.producer_acquire();
        pipeline.memcpy_async(source, destination);
        pipeline.producer_commit();

        pipeline.consumer_wait();
        output[0] = smem[0];
        pipeline.consumer_release();
    }
}

pub fn test_async_copy<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    if !client.properties().feature_enabled(Feature::Pipeline) {
        // We can't execute the test, skip.
        return;
    }

    let input = client.create(as_bytes![F: 0.0, 1.0, 2.0, 3.0, 4.0]);
    let output = client.empty(core::mem::size_of::<F>());

    unsafe {
        async_copy_test::launch::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<F>(&input, 5, 1),
            ArrayArg::from_raw_parts::<F>(&output, 1, 1),
        )
    };

    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(2.0));
}

pub fn test_pipelined_sum<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    if !client.properties().feature_enabled(Feature::Pipeline) {
        // We can't execute the test, skip.
        return;
    }

    let input = client.create(as_bytes![F: 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let output = client.empty(core::mem::size_of::<F>());

    unsafe {
        pipelined_sum::launch::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<F>(&input, 8, 1),
            ArrayArg::from_raw_parts::<F>(&output, 1, 1),
            2,
        )
    };

    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(36.0));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_pipeline {
    () => {
        use super::*;

        #[test]
        fn test_pipeline_async_copy() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::pipeline::test_async_copy::<TestRuntime, FloatType>(client);
        }

        #[test]
        fn test_pipeline_sum() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::pipeline::test_pipelined_sum::<TestRuntime, FloatType>(
                client,
            );
        }
    };
}
