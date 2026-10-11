use crate::compiler::{CudaBackend, CudaCompilationOptions};
use crate::compute::artifact::CudaArtifactCompiler;
use crate::compute::events::{EventProfiler, driver_error, poisons_device};
use crate::compute::modules::{CudaCompiledKernel, CudaModules};
use crate::compute::storage::gpu::GpuResource;
use crate::compute::stream::Stream;
use cubecl_core::{ir::DeviceProperties, prelude::*};
use cubecl_cpp::cuda::arch::CudaArchitecture;
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::compiler::{ArtifactId, KernelLoader};
use cubecl_server::cooperative::{is_cooperative, too_many_cubes};
use cubecl_server::kernel::CubeKernel;
use cubecl_server::logging::ServerLogger;
use cudarc::driver::DriverError;
use cudarc::driver::sys::{
    CUctx_st, CUfunc_st, CUfunction_attribute, CUlaunchAttribute, CUlaunchAttributeID,
    CUlaunchConfig, CUresult, CUstream, cuLaunchKernelEx,
};
use std::os::raw::c_void;

#[derive(Debug)]
pub(crate) struct CudaContext {
    pub context: *mut CUctx_st,
    /// The stream collectives run on. Kept on the context so a relocation —
    /// which reaches the context, not the server — can wait on it.
    pub comm_stream: CUstream,
    /// The stream transfers between devices run on, apart from the collectives', so neither
    /// queues behind the other.
    pub transfer_stream: CUstream,
    /// One element for a communicator's first `all_reduce`, allocated with the context so that
    /// setting a group up never fails to allocate on one device after joining on the others.
    pub connect_element: GpuResource,
    /// The modules loaded on the device, and how to load another.
    kernels: KernelLoader<CudaModules>,
    /// The options kernels are compiled with.
    compilation_options: CudaCompilationOptions,
    /// The number of streaming multiprocessors, which the capacity of a kernel scales with.
    streaming_multiprocessors: u32,
    pub profiler: EventProfiler,
}

impl CudaContext {
    /// `backend` is which one compiles here; see [`CudaModules::new`].
    ///
    /// An environment switch drops every loaded kernel, so the new
    /// environment's PTX store is filled rather than bypassed. The modules
    /// those kernels named stay resident: nothing calls `cuModuleUnload`, here
    /// or anywhere else in this context, and unloading one a stream still has
    /// queued work against would be unsound. A process that switches
    /// environments a handful of times at startup pays a bounded price; one
    /// that switches repeatedly grows its resident modules without bound — see
    /// [`cubecl_environment::environment::activate`].
    pub fn new(
        compilation_options: CudaCompilationOptions,
        properties: DeviceProperties,
        context: *mut CUctx_st,
        arch: CudaArchitecture,
        backend: CudaBackend,
        comm_stream: CUstream,
        transfer_stream: CUstream,
    ) -> Self {
        let streaming_multiprocessors = properties
            .hardware
            .num_streaming_multiprocessors
            .unwrap_or(1);
        let compiler = CudaArtifactCompiler::new(properties, compilation_options.clone(), arch);

        Self {
            context,
            comm_stream,
            transfer_stream,
            // SAFETY: the context is current. The element is only ever reduced into itself, and
            // nothing reads what that leaves in it.
            connect_element: unsafe { cudarc::driver::result::malloc_sync(4) }
                .map(|ptr| GpuResource::new(ptr, core::ptr::null_mut(), 4))
                .expect("Can allocate the element communicators connect over."),
            kernels: KernelLoader::new(CudaModules::new(compiler, backend)),
            compilation_options,
            streaming_multiprocessors,
            profiler: EventProfiler::default(),
        }
    }

    /// The options kernels are compiled with, which the launch path reads
    /// to pass arguments the way the compiled code expects them.
    pub fn compilation_options(&self) -> &CudaCompilationOptions {
        &self.compilation_options
    }

    /// Switches the current CUDA context to this context.
    pub fn unsafe_set_current(&self) -> Result<(), DriverError> {
        // SAFETY: `self.context` is a valid CUDA context obtained from `primary_ctx::retain`
        // during server initialization and remains valid for the server's lifetime.
        unsafe { cudarc::driver::result::ctx::set_current(self.context) }
    }

    /// Loads `kernel`, whose id is `id`, on the device, compiling it first
    /// when no store holds it. A kernel already loaded costs a map lookup.
    pub fn load_kernel(
        &mut self,
        kernel: &dyn CubeKernel,
        id: &ArtifactId<()>,
        logger: &ServerLogger,
    ) -> Result<CudaCompiledKernel, LaunchError> {
        self.kernels.load(kernel, id, logger)
    }

    /// Compiles every queued kernel now, rather than inside the next launch.
    pub fn compile_queued(&mut self, logger: &ServerLogger) {
        self.kernels.compile_queued(logger);
    }

    /// Queues `kernel` to be compiled with others, by the next kernel
    /// loaded for a launch.
    pub fn queue_kernel(&mut self, kernel: Box<dyn CubeKernel>) {
        self.kernels.enqueue(kernel, ());
    }

    pub fn execute_task(
        &mut self,
        stream: &mut Stream,
        id: &KernelId,
        kernel: &CudaCompiledKernel,
        dispatch_count: (u32, u32, u32),
        resources: &mut [*mut c_void],
    ) -> Result<(), LaunchError> {
        let func = kernel.func;
        let cube_dim = (kernel.cube_dim.x, kernel.cube_dim.y, kernel.cube_dim.z);
        // Shared memory is collected into a single buffer, with each shared memory being
        // an offset pointer
        let shared_mem_bytes = kernel.shared_mem_bytes;
        let launch_shared_mem = u32::try_from(shared_mem_bytes).unwrap_or(u32::MAX);
        // SAFETY: `func` is a valid function handle from a loaded module.
        // `stream.sys` is a valid CUDA stream. `bindings` contains valid device pointers
        // for all kernel arguments. The dispatch and cube dimensions are validated by
        // the caller.
        let launched = unsafe {
            allow_dynamic_shared(func, shared_mem_bytes)
                .map_err(|err| launch_failed("cuFuncSetAttribute", err))?;
            if is_cooperative(id) {
                launch_cooperative(
                    func,
                    dispatch_count,
                    cube_dim,
                    launch_shared_mem,
                    stream.sys,
                    resources,
                )
                .map_err(|err| ("cuLaunchKernelEx", err))
            } else {
                cudarc::driver::result::launch_kernel(
                    func,
                    dispatch_count,
                    cube_dim,
                    launch_shared_mem,
                    stream.sys,
                    resources,
                )
                .map_err(|err| ("cuLaunchKernel", err))
            }
        };

        match launched {
            Err((_, err)) if err.0 == CUresult::CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE => {
                let requested = dispatch_count.0 * dispatch_count.1 * dispatch_count.2;
                Err(too_many_cubes(requested, self.capacity(kernel)?).into())
            }
            launched => launched.map_err(|(op, err)| launch_failed(op, err)),
        }
    }

    /// How many cubes of the loaded `kernel` the device runs at the same time: the driver
    /// occupancy per SM, with the real cube dim and dynamic shared memory, times the SM count.
    pub fn capacity(&self, kernel: &CudaCompiledKernel) -> Result<u32, LaunchError> {
        let (func, shared_mem_bytes) = (kernel.func, kernel.shared_mem_bytes);
        let units = kernel.cube_dim.num_elems().cast_signed();
        // SAFETY: `func` is a valid function handle from a loaded module.
        let per_sm = unsafe {
            // The occupancy counts only as much dynamic shared memory as the launch permits.
            allow_dynamic_shared(func, shared_mem_bytes)
                .map_err(|err| launch_failed("cuFuncSetAttribute", err))?;
            cudarc::driver::result::occupancy::max_active_block_per_multiprocessor(
                func,
                units,
                shared_mem_bytes,
            )
            .map_err(|err| launch_failed("cuOccupancyMaxActiveBlocksPerMultiprocessor", err))?
        };
        Ok(per_sm.cast_unsigned() * self.streaming_multiprocessors)
    }
}

/// Lets `func` use `shared_mem_bytes` of dynamic shared memory.
///
/// # Safety
///
/// `func` must be a valid function handle from a loaded module.
unsafe fn allow_dynamic_shared(
    func: *mut CUfunc_st,
    shared_mem_bytes: usize,
) -> Result<(), DriverError> {
    unsafe {
        cudarc::driver::result::function::set_function_attribute(
            func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            i32::try_from(shared_mem_bytes).unwrap_or(i32::MAX),
        )
    }
}

/// Launches `func` so that all its cubes run at the same time. `cuLaunchKernelEx` with the
/// cooperative attribute, because a graph capture keeps the attribute in the kernel node.
///
/// # Safety
///
/// The same as [`cudarc::driver::result::launch_kernel`].
unsafe fn launch_cooperative(
    func: *mut CUfunc_st,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    stream: CUstream,
    params: &mut [*mut c_void],
) -> Result<(), DriverError> {
    // SAFETY: an all-zero attribute is a valid value of this plain C struct.
    let mut attribute: CUlaunchAttribute = unsafe { std::mem::zeroed() };
    attribute.id = CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
    attribute.value.cooperative = 1;
    let config = CUlaunchConfig {
        gridDimX: grid_dim.0,
        gridDimY: grid_dim.1,
        gridDimZ: grid_dim.2,
        blockDimX: block_dim.0,
        blockDimY: block_dim.1,
        blockDimZ: block_dim.2,
        sharedMemBytes: shared_mem_bytes,
        hStream: stream,
        attrs: &raw mut attribute,
        numAttrs: 1,
    };
    unsafe {
        cuLaunchKernelEx(
            &raw const config,
            func,
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
        .result()
    }
}

/// A refused launch error : maps to a poisoned-device error if that call happened right after
/// a fault.
fn launch_failed(op: &'static str, err: cudarc::driver::DriverError) -> LaunchError {
    match poisons_device(err.0) {
        true => driver_error(op, err).into(),
        false => LaunchError::Unknown {
            reason: format!("{err}"),
            backtrace: BackTrace::capture(),
        },
    }
}
