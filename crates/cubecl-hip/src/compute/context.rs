//! The device context: its loaded kernels, and its open profiles.
//!
//! Compilation is memoized here rather than per stream, because a module is
//! loaded into the context and every stream sharing it can launch from the
//! same one. What a compiled kernel answers for beyond its entry point is
//! which of its bindings it writes, which is what a launch stages its write
//! scope from.

use super::storage::gpu::GpuResource;
use crate::compiler::{HipBackend, HipCompilationOptions};
use crate::compute::artifact::HipArtifactCompiler;
use crate::compute::events::EventProfiler;
use crate::compute::modules::{HipCompiledKernel, HipModules};
use crate::compute::status::checked;
use crate::compute::stream::Stream;
use cubecl_core::{ir::DeviceProperties, prelude::*};
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::compiler::{ArtifactId, KernelLoader};
use cubecl_server::cooperative::{is_cooperative, too_many_cubes};
use cubecl_server::kernel::CubeKernel;
use cubecl_server::logging::ServerLogger;

#[derive(Debug)]
pub(crate) struct HipContext {
    /// The modules loaded on the device, and how to load another.
    kernels: KernelLoader<HipModules>,
    /// The number of compute units, which the capacity of a kernel scales with.
    compute_units: u32,
    pub profiler: EventProfiler,
}

impl HipContext {
    /// `fingerprint` and `backend` name where compiled kernels are stored; see
    /// [`HipModules::new`].
    ///
    /// An environment switch drops every loaded kernel, so the new
    /// environment's store is filled rather than bypassed. The modules those
    /// kernels named stay resident: nothing calls `hipModuleUnload`, here or
    /// anywhere else in this context, and unloading one a stream still has
    /// queued work against would be unsound. A process that switches
    /// environments a handful of times at startup pays a bounded price; one
    /// that switches repeatedly grows its resident modules without bound — see
    /// [`cubecl_environment::environment::activate`].
    pub fn new(
        compilation_options: HipCompilationOptions,
        properties: DeviceProperties,
        fingerprint: String,
        backend: HipBackend,
    ) -> Self {
        let compute_units = properties
            .hardware
            .num_streaming_multiprocessors
            .unwrap_or(1);
        let compiler = HipArtifactCompiler::new(properties, compilation_options);
        Self {
            kernels: KernelLoader::new(HipModules::new(compiler, fingerprint, backend)),
            compute_units,
            profiler: EventProfiler::default(),
        }
    }

    /// Loads `kernel`, whose id is `id`, on the device, compiling it first
    /// when no store holds it. A kernel already loaded costs a map lookup.
    pub fn load_kernel(
        &mut self,
        kernel: &dyn CubeKernel,
        id: &ArtifactId<()>,
        logger: &ServerLogger,
    ) -> Result<HipCompiledKernel, LaunchError> {
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

    /// Executes a task on the given stream.
    pub fn execute_task(
        &mut self,
        stream: &mut Stream,
        kernel_id: &KernelId,
        kernel: &HipCompiledKernel,
        dispatch_count: (u32, u32, u32),
        resources: &[GpuResource],
    ) -> Result<(), LaunchError> {
        let mut bindings = resources
            .iter()
            .map(|memory| memory.binding)
            .collect::<Vec<_>>();

        let func = kernel.func;
        let cube_dim = kernel.cube_dim;
        // Shared memory is collected into a single buffer, with each shared memory being
        // an offset pointer
        let shared_mem_bytes = u32::try_from(kernel.shared_mem_bytes).unwrap_or(u32::MAX);

        // SAFETY: `func` is a valid function handle from a loaded module.
        // `stream.sys` is a valid HIP stream. `bindings` contains valid device pointers
        // for all kernel arguments. The dispatch and cube dimensions are validated by
        // the caller.
        let (name, status) = unsafe {
            if is_cooperative(kernel_id) {
                let status = cubecl_hip_sys::hipModuleLaunchCooperativeKernel(
                    func,
                    dispatch_count.0,
                    dispatch_count.1,
                    dispatch_count.2,
                    cube_dim.x,
                    cube_dim.y,
                    cube_dim.z,
                    shared_mem_bytes,
                    stream.sys,
                    bindings.as_mut_ptr(),
                );
                ("hipModuleLaunchCooperativeKernel", status)
            } else {
                let status = cubecl_hip_sys::hipModuleLaunchKernel(
                    func,
                    dispatch_count.0,
                    dispatch_count.1,
                    dispatch_count.2,
                    cube_dim.x,
                    cube_dim.y,
                    cube_dim.z,
                    shared_mem_bytes,
                    stream.sys,
                    bindings.as_mut_ptr(),
                    std::ptr::null_mut(),
                );
                ("hipModuleLaunchKernel", status)
            }
        };

        // Out of memory is told apart from the rest because the caller
        // can act on it — reclaim and relaunch — where nothing else here
        // is worth retrying.
        match checked(name, status) {
            Ok(()) => Ok(()),
            Err(_) if status == cubecl_hip_sys::hipError_t_hipErrorOutOfMemory => {
                Err(LaunchError::OutOfMemory {
                    reason: format!("out of memory launching kernel {kernel_id:?}"),
                    backtrace: BackTrace::capture(),
                })
            }
            Err(_) if status == cubecl_hip_sys::hipError_t_hipErrorCooperativeLaunchTooLarge => {
                let requested = dispatch_count.0 * dispatch_count.1 * dispatch_count.2;
                Err(too_many_cubes(requested, self.capacity(kernel_id, kernel)?).into())
            }
            Err(err) => Err(LaunchError::Unknown {
                reason: format!("{err}, launching kernel {kernel_id:?}"),
                backtrace: BackTrace::capture(),
            }),
        }
    }

    /// How many cubes of the loaded `kernel` the device runs at the same time: the driver
    /// occupancy per compute unit, with the real cube dim and dynamic shared memory, times the
    /// compute unit count.
    pub fn capacity(
        &self,
        kernel_id: &KernelId,
        kernel: &HipCompiledKernel,
    ) -> Result<u32, LaunchError> {
        let units = kernel.cube_dim.num_elems().cast_signed();
        let mut per_cu = 0;
        // SAFETY: `kernel.func` is a valid function handle from a loaded module, and
        // `per_cu` outlives the call.
        let status = unsafe {
            cubecl_hip_sys::hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                &raw mut per_cu,
                kernel.func,
                units,
                kernel.shared_mem_bytes,
            )
        };
        checked("hipModuleOccupancyMaxActiveBlocksPerMultiprocessor", status).map_err(|err| {
            LaunchError::Unknown {
                reason: format!("{err}, querying the capacity of kernel {kernel_id:?}"),
                backtrace: BackTrace::capture(),
            }
        })?;
        Ok(per_cu.cast_unsigned() * self.compute_units)
    }
}
