# Persistent kernels: implementation plan

- **Base commit:** [`a1bb768ce919260eea56dbd0b59c70e55236e22d`](https://github.com/skewballfox/cubecl/commit/a1bb768ce919260eea56dbd0b59c70e55236e22d) (`main`, 2026-09-25).
- **Language:** ASD-STE100 and Attempto Controlled English. Domain terms are permitted.
- **Open decisions:** see [`DECISIONS.md`](./DECISIONS.md). This plan refers to a decision as `Dn`.
- **Reverification:** All permalinks point to the base commit. Before you implement a step against a later commit, open each permalink of that step. Compare it with the same path on the later commit. If the code changed, verify the step again.

## 0. Terms

| Term | Meaning |
|---|---|
| Persistent kernel | A kernel whose cube count is set by the device capacity, not by the problem size. Each cube loops over work items. |
| Co-residency | All cubes of a launch execute on the device at the same time. |
| Grid sync | A barrier that all units of all cubes of a launch must reach. It is a device-scope release before and a device-scope acquire after. |
| Capacity | The maximum number of co-resident cubes for one compiled kernel and one cube dim. |
| Tier A | A persistent kernel without grid sync. It is correct on every runtime. |
| Tier B | A persistent kernel with grid sync. It needs co-residency or emulation. |

## 1. Research summary

### 1.1 CUDA (native)

- The driver gives co-residency only through a cooperative launch: `cuLaunchCooperativeKernel`, or `cuLaunchKernelEx` with `CU_LAUNCH_ATTRIBUTE_COOPERATIVE`. The device attribute `CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH` tells if the device supports it.
- If the grid is larger than the capacity, the driver rejects the launch (`CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`).
- The kernel syncs the grid with `cooperative_groups::this_grid().sync()`. Since CUDA 11, this does not need `-rdc`.
- The capacity is `cuOccupancyMaxActiveBlocksPerMultiprocessor(func, units_per_cube, dyn_smem) × SM count`.
- HIP has the same model: `hipModuleLaunchCooperativeKernel`, `hipDeviceAttributeCooperativeLaunch`, `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`. The pinned `cubecl-hip-sys` 7.14 and `cudarc` 0.19.9 expose all of these functions.

**Advantages**

1. One launch replaces many launches. The launch overhead occurs one time.
2. Registers and shared memory keep state between work items.
3. Dynamic work distribution (atomic queue, stream-K) removes tail effects.
4. A grid sync replaces a kernel boundary.

**Disadvantages and pitfalls.** Each row has an ID. Section 3 refers to these IDs.

| ID | Pitfall | Cause | Mitigation in this plan |
|---|---|---|---|
| P1 | Deadlock | A spin barrier runs without co-residency. | `sync_grid` compiles only in a `persistent` kernel (3.4 V1). The safe API never spins without a guarantee. |
| P2 | Deadlock | Some units skip a grid sync (divergent control flow). | Compile error if the block is not device-uniform (3.4 V2). |
| P3 | Launch failure or deadlock | The capacity is computed from wrong values (hard-coded SM count, stale register count, wrong dynamic shared memory, MPS/MIG). | Capacity comes from the compiled function, at launch time (3.5). |
| P4 | Stale reads | A cube reads data from another cube without release/acquire. The compiler hoists a load out of a spin loop. | Grid sync includes device fences. Work-queue helpers use atomics only. |
| P5 | Wrong work distribution on the next launch | A work counter is not reset between launches. | See D5. |
| P6 | Hang of the display or a driver reset | A long kernel hits the watchdog (TDR, WDDM). | Documentation only. No enforcement. |
| P7 | Less throughput | Too many grid syncs. A grid sync costs microseconds. | Documentation. Grid sync is explicit, never implicit. |
| P8 | Lost concurrency | A kernel that fills the device blocks other streams. | `PersistentCount::AtMost`. See D3. |
| P9 | Capture failure | A cooperative launch in a CUDA graph. | Use `cuLaunchKernelEx` + attribute. The attribute is kept in captured kernel nodes. |
| P10 | Divergent result between backends | Shared memory or registers survive a grid sync on CUDA, but not in an emulation. | See D2. |

### 1.2 wgpu (emulation)

- WebGPU has no inter-workgroup barrier, no forward-progress guarantee between workgroups, and no occupancy query ([Sorensen et al., OOPSLA 2016](https://www.doc.ic.ac.uk/~afd/papers/2016/OOPSLA.pdf); [OOPSLA 2021](https://arxiv.org/pdf/2109.06132)).
- WGSL `storageBarrier()` has workgroup scope only. cubecl already models this with `Features::device_memory_scope`. The wgpu Vulkan and Metal backends set it to `true`. WebGPU (WGSL) does not.
- WebGPU orders consecutive dispatches in one pass, and makes the storage writes of one dispatch visible to the next.
- Three emulation techniques exist:
  - **E1 Dispatch split.** Each grid sync becomes a dispatch boundary. This is always correct. State in registers and shared memory is lost. Grid syncs in runtime-bounded loops need host-side iteration.
  - **E2 Occupancy discovery + spin barrier.** At kernel start, cubes register under a lock. Cubes that arrive after the poll closes exit. The rest use an atomic barrier ([reference implementation](https://github.com/mc-imperial/gpu_discovery_barrier)). This relies on the occupancy-bound execution model. The WebGPU specification does not guarantee this model. E2 also needs `device_memory_scope`.
  - **E3 No emulation.** Tier A needs no co-residency. A heuristic cube count is sufficient.

## 2. State of cubecl at the base commit

| Area | Current state | Link |
|---|---|---|
| Sync scopes | `SyncScope { Unit, Plane, Cube, Device }`. `Device` is a cube barrier + device fences. There is no grid barrier. | [`dialect/synchronization.rs` L23-L41](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-ir/src/dialect/synchronization.rs#L23-L41) |
| Frontend sync | `sync_cube`, `sync_plane`, `sync_storage`. | [`frontend/synchronization.rs` L14-L70](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-core/src/frontend/synchronization.rs#L14-L70) |
| Cross-cube test | The last-cube pattern exists. It does not spin. | [`runtime_tests/synchronization.rs` L353-L448](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-core/src/runtime_tests/synchronization.rs#L353-L448) |
| Atomics | No memory-order operand. All atomics are relaxed. | [`dialect/atomic.rs` L1-L58](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-ir/src/dialect/atomic.rs#L1-L58) |
| Features | `device_memory_scope` exists. No cooperative feature. `Features` derives `Default`. | [`features.rs` L8-L38](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-ir/src/features.rs#L8-L38) |
| Hardware properties | `num_streaming_multiprocessors` exists. It is `None` on wgpu. The struct is built by struct literal in 7 places. | [`properties.rs` L30-L50](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-ir/src/properties.rs#L30-L50), [wgpu L475](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-wgpu/src/runtime.rs#L470-L480) |
| Kernel settings | `KernelSettings` has `cluster_dim`, no persistent field. | [`settings.rs` L78-L105](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-ir/src/settings.rs#L78-L105) |
| Kernel id | `KernelId` hashes cube dim, address type, mode, info. | [`id.rs` L90-L101](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/id.rs#L90-L101) |
| Cube count | `CubeCount { Static, Dynamic }`. It is matched exhaustively in many places. | [`server/base.rs` L1495-L1500](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/server/base.rs#L1495-L1500) |
| Server launch | `Server::launch(kernel, count, bindings, stream, mode)`. | [`server/base.rs` L608-L629](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/server/base.rs#L608-L629) |
| Client launch | `launch` → `launch_inner` (profiling, dry run, logging). | [`client.rs` L1073-L1262](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/client.rs#L1073-L1262) |
| Errors | `ResourceLimitError` and `LaunchError` are not `#[non_exhaustive]`. | [`server/base.rs` L215-L280](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/server/base.rs#L215-L280) |
| Macro flags | `launch`, `launch_unchecked`, `cluster_dim`, … | [`parse/kernel.rs` L21-L42](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/parse/kernel.rs#L21-L42) |
| Macro launch fns | `launch` / `launch_unchecked` generation. | [`generate/launch.rs` L56-L156](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/launch.rs#L56-L156) |
| Macro settings / id | `cluster_dim` goes into settings. `id()` builds the `KernelId`. | [`generate/kernel.rs` L417-L470](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/kernel.rs#L417-L470) |
| Launcher | `KernelLauncher::launch` calls `Client::launch`. | [`compute/launcher.rs` L66-L73](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-core/src/compute/launcher.rs#L66-L73) |
| Uniformity | Block uniformity analysis has a `Device` level. | [`cubecl-opt passes/uniformity.rs` L31-L53](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-opt/src/passes/uniformity.rs#L31-L53) |
| CUDA launch | `cuLaunchKernel` only. | [`cuda context.rs` L517-L557](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cuda/src/compute/context.rs#L517-L557) |
| CUDA compiled kernel | Stores `cube_dim`, `shared_mem_bytes`, `func`, `io`. | [`cuda context.rs` L66-L78](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cuda/src/compute/context.rs#L66-L78) |
| CUDA features | SM count read; `device_memory_scope = true`. | [`cuda runtime.rs` L170-L172](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cuda/src/runtime.rs#L170-L172), [L357-L359](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cuda/src/runtime.rs#L357-L359) |
| CUDA C++ sync lowering | `SyncOp` → `__syncthreads` / `__threadfence`. | [`cpp cuda/dialect.rs` L67-L78](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cpp/src/cuda/dialect.rs#L67-L78) |
| CUDA includes | Op-driven `cooperative_groups.h` include exists for clusters. | [`cpp cuda/builtin.rs` L20-L32](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cpp/src/cuda/builtin.rs#L20-L32) |
| HIP | `hipModuleLaunchKernel` only. | [`hip context.rs` L439-L490](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-hip/src/compute/context.rs#L439-L490), [`cpp hip/dialect.rs` L30-L50](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cpp/src/hip/dialect.rs#L30-L50) |
| Metal (native) | `dispatchThreadgroups`; MSL sync lowering. | [`metal server.rs` L510-L540](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-metal/src/compute/server.rs#L510-L540), [`cpp metal/dialect.rs` L48-L70](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cpp/src/metal/dialect.rs#L48-L70) |
| wgpu launch | One pipeline per kernel. One `ScheduleTask::Execute` per launch. | [`wgpu server.rs` L511-L610](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-wgpu/src/compute/server.rs#L511-L610), [`stream.rs` L1056-L1128](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-wgpu/src/compute/stream.rs#L1056-L1128) |
| wgpu sync lowering | WGSL and SPIR-V. | [`wgsl ops/sync.rs` L5-L13](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-wgpu/src/compiler/wgsl/ops/sync.rs#L5-L13), [`spirv ops/sync.rs` L10-L44](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-spirv/src/ops/sync.rs#L10-L44) |
| LLVM targets | `SyncScope::Device` panics on CPU. GPU targets lower to a cube barrier. | [`llvm polyfill/synchronization.rs` L15-L33](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/shared/polyfill/synchronization.rs#L15-L33) |
| CPU runtime | Thread pool. One worker per unit only for barrier kernels. | [`dispatcher.rs` L43-L70](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cpu/src/compute/threadpool/scheduler/dispatcher.rs#L43-L70) |

**Findings.** cubecl has no cooperative launch, no grid sync, and no occupancy query. The existing hooks are sufficient for an additive design: a new op, a new macro flag, new default-implemented trait methods, and a new `Features` field.

## 3. Design

### 3.1 Principles

1. Do not change existing signatures. Add new items only. Exception: new error variants (3.6).
2. The safe API must not deadlock (P1, P2). The unsafe API permits more control.
3. Tier A works on every runtime from the first milestone.
4. Grid sync is explicit. The runtime never inserts it.

### 3.2 User API (default path)

```rust
#[cube(launch, persistent)]
fn scale(input: &[f32], output: &mut [f32], #[comptime] steps: u32) {
    #[unroll]
    for _ in 0..steps {
        for i in persistent_range(input.len()) {   // cube-strided work loop
            output[i] = input[i] * 2.0;
        }
        sync_grid();                                // Tier B only
    }
}

scale::launch_persistent(&client, PersistentCount::Fill, CubeDim::new_1d(256), input, output, 4);
```

- The flag `persistent` generates `launch_persistent`. If the kernel also has `launch_unchecked`, the flag generates `launch_persistent_unchecked`. The arguments are equal to `launch`, but `PersistentCount` replaces `CubeCount`.
- `PersistentCount` (new, in `cubecl-runtime/src/server/base.rs`, next to `CubeCount`):
  - `Fill` — the runtime uses the capacity (see D3).
  - `AtMost(u32)` — the runtime uses `min(n, capacity)`.
  - `Exact(u32)` — the runtime uses `n`. If a Tier B kernel has `n > capacity`, the launch fails (3.4 V4).
- The kernel reads the chosen count from the existing `CUBE_COUNT` builtin.
- See D8 for one flag against two flags.

### 3.3 Kernel primitives

| Item | Crate / file | Behaviour |
|---|---|---|
| `GridSyncOp` (`sync.grid`) | `cubecl-ir/src/dialect/synchronization.rs`, after `SyncAsyncProxyOp` | A new op. Do **not** add a `SyncScope::Grid` variant. Reason: 7 lowerings match `SyncScope` exhaustively, and `SyncScope` permits a weaker fallback. A grid sync has no weaker fallback. Implement `Synchronizes` with minimum and maximum `SyncScope::Device`. Implement `SideEffects` = `true`. |
| `sync_grid()` | `cubecl-core/src/frontend/synchronization.rs`, after `sync_storage` | Safe. Registers `GridSyncOp`. |
| `unsafe sync_grid_unchecked()` | same file | Registers `GridSyncOp` with a unit attribute `unchecked`. Validation V1 does not apply. The caller guarantees co-residency. |
| `persistent_range(len)` | new `cubecl-std/src/persistent.rs` | Returns a cube-uniform range `CUBE_POS, CUBE_POS + CUBE_COUNT, …`. Cube-uniform indices keep `sync_cube` legal inside the loop. |
| `persistent_range_units(len)` | same file | Unit-strided variant: `ABSOLUTE_POS` stepped by `CUBE_COUNT × CUBE_DIM`. |
| `WorkQueue` | same file | Atomic ticket counter for dynamic distribution. `next()` returns `Option<u32>`, taken by unit 0 and broadcast through shared memory. The reset strategy is D5. |

### 3.4 Validation (trap avoidance)

| ID | Rule | Where | Pitfall |
|---|---|---|---|
| V1 | If a kernel contains a checked `GridSyncOp` and `KernelSettings::persistent` is `false`, compilation fails. | New verifier in `cubecl-opt/src/passes/grid_sync.rs`. Call it from each compiler before lowering. | P1 |
| V2 | If the block of a `GridSyncOp` is not `Uniformity::Device`, compilation fails. | Same pass. Use `uniformity_solver` and `BlockUniformityAnalysis` ([L31-L40](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-opt/src/passes/uniformity.rs#L31-L40)). | P2 |
| V3 | If the runtime reports `GridSync::None` and the kernel contains `GridSyncOp`, compilation fails with a named error. | Each compiler, when it meets an op without a lowering. | P1 |
| V4 | If `Exact(n)` exceeds the capacity for a Tier B kernel, the launch fails with `ResourceLimitError::CooperativeGrid { requested, max }`. | Server, before the driver call. | P3 |
| V5 | The capacity comes from the compiled function, with the real dynamic shared memory. It is never cached across `ExecutionMode`s. | `Server::capacity` (3.5). | P3 |
| V6 | The count is clamped to `max_cube_count.0`. | Client, in `launch_persistent`. | — |

### 3.5 Runtime layer (`cubecl-ir`, `cubecl-runtime`, `cubecl-core`, `cubecl-macros`)

1. `cubecl-ir/src/features.rs`: add `pub grid_sync: GridSync` to `Features`. Add `enum GridSync { #[default] None, Native, Emulated(EnumSet<GridSyncEmulation>) }` and `enum GridSyncEmulation { Split, Spin }`. `Default` keeps all current constructors valid.
2. `cubecl-ir/src/settings.rs`: add `pub persistent: bool` to `KernelSettings` (default `false`) and a builder `persistent()`, next to `cluster_dim()` ([L133-L136](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-ir/src/settings.rs#L133-L136)).
3. `cubecl-runtime/src/id.rs`: add `persistent: bool` to `KernelId` and a builder. The compile caches then keep persistent and plain variants apart.
4. `cubecl-runtime/src/server/base.rs`:
   - Add `PersistentCount` and `Capacity { cubes: u32, exact: bool }`.
   - Add to `trait Server` ([L543](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/server/base.rs#L543)) two methods with default bodies:
     - `fn capacity(&mut self, kernel: &dyn CubeKernel, stream_id) -> Result<Capacity, ServerError>`. The default returns an estimate (D6), with `exact: false`.
     - `unsafe fn launch_persistent(&mut self, kernel, count: PersistentCount, bindings, stream_id, launch_mode)`. The default resolves the count with `capacity` and calls `launch` with `CubeCount::Static(n, 1, 1)`. This default is correct for Tier A on every backend. A Tier B kernel on a backend without an override fails at V3.
5. `cubecl-runtime/src/client.rs`: add `Client::launch_persistent` and `Client::capacity`. Do not copy `launch_inner`. Change the private `launch_inner` to take a private `enum LaunchShape { Count(CubeCount), Persistent(PersistentCount) }`, and dispatch to `state.launch` or `state.launch_persistent` at the three call sites ([L1130](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/client.rs#L1128-L1132), [L1161](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/client.rs#L1159-L1163), [L1186](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/client.rs#L1184-L1192)). The public `launch` stays unchanged.
6. `cubecl-core/src/compute/launcher.rs`: add `KernelLauncher::launch_persistent(self, count, kernel, client)` next to `launch` ([L66-L73](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-core/src/compute/launcher.rs#L66-L73)).
7. `cubecl-macros`:
   - `parse/kernel.rs` `KernelArgs`: add `pub persistent: Flag`. `is_launch()` also returns `true` for it.
   - `generate/kernel.rs` at [L421-L423](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/kernel.rs#L421-L423) and [L512-L514](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/kernel.rs#L512-L514): emit `.persistent()` when the flag is present. In `id()` ([L458-L470](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/kernel.rs#L458-L470)): chain `.persistent(self.settings.persistent)`.
   - `generate/launch.rs`: add `fn launch_persistent()` and `fn launch_persistent_unchecked()` beside [L56-L137](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/launch.rs#L56-L137). Reuse `launch_body`. Add `fn capacity(__client, __cube_dim, args…) -> Result<Capacity, ServerError>` for granular control.

### 3.6 Errors

Add `ResourceLimitError::CooperativeGrid { requested: u32, max: u32, backtrace }` and `CompilationError` text for V1–V3. `ResourceLimitError` is not `#[non_exhaustive]`, so a new variant breaks exhaustive `match` in downstream crates. Accept this, or mark the enum `#[non_exhaustive]` in the same release.

### 3.7 Backends

#### CUDA — native (`cubecl-cuda`, `cubecl-cpp`)

1. `runtime.rs`, after [L357-L359](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cuda/src/runtime.rs#L357-L359): read `CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH`. If it is `1`, set `features.grid_sync = GridSync::Native`.
2. `compute/context.rs` `CudaCompiledKernel` ([L66-L78](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cuda/src/compute/context.rs#L66-L78)): add `cooperative: bool`. Set it to `true` if the IR contains `GridSyncOp`. Add the field to `PtxCacheEntry` and change the cache version, because old entries do not have it.
3. `compute/context.rs` `execute_task` ([L540-L553](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cuda/src/compute/context.rs#L540-L553)): if `cooperative`, call `cuLaunchKernelEx` with `CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 1` (P9). Else keep `launch_kernel`.
4. `compute/context.rs`: add `fn capacity(&mut self, kernel_id) -> u32`. Compute `occupancy::max_active_block_per_multiprocessor(func, cube_dim.num_elems(), shared_mem_bytes) × SM count`. Set `MAX_DYNAMIC_SHARED_SIZE_BYTES` before the query, as `execute_task` does.
5. `compute/server.rs`: override `capacity` (compile, then query) and `launch_persistent` (resolve the count, check V4, then run the normal launch path).
6. `cubecl-cpp/src/cuda/dialect.rs`: add `cuda_op!(GridSyncOp, …)` → `cooperative_groups::this_grid().sync();`. Declare `cooperative_groups.h` with the same `includes` mechanism as [`builtin.rs` L20-L32](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cpp/src/cuda/builtin.rs#L20-L32).
7. LLVM `nvptx` target: see D7. Until D7 is closed, report `GridSync::None` when the LLVM compiler is active.

#### HIP — native (`cubecl-hip`, `cubecl-cpp`)

Apply CUDA steps 1–6 with `hipDeviceAttributeCooperativeLaunch`, `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`, and `hipModuleLaunchCooperativeKernel` in `execute_task` ([L459](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-hip/src/compute/context.rs#L455-L480)). Lower `GridSyncOp` in `cubecl-cpp/src/hip/dialect.rs` to `cooperative_groups::this_grid().sync();` with `hip/hip_cooperative_groups.h`.

#### wgpu — emulated (`cubecl-wgpu`, `cubecl-spirv`, `cubecl-opt`)

Tier A needs no change. The default `launch_persistent` is sufficient. For Tier B, implement two emulations. D1 selects the default.

**E1 Split** (`GridSyncEmulation::Split`)
1. New pass `cubecl-opt/src/passes/split_grid_sync.rs`. Input: the entry function after `#[unroll]` expansion. Output: `N + 1` entry functions for `N` grid syncs.
2. Constraint S1: every `GridSyncOp` must be in the top-level block of the entry function. Else compilation fails with the text "sync_grid inside a runtime loop needs native or Spin grid sync".
3. Constraint S2: an SSA value that is defined before a `GridSyncOp` and used after it must be rematerializable (constant, builtin, argument, scalar, or a pure op on these). The pass copies it into the next phase. Else compilation fails and names the value.
4. Shared memory after a split: see D2.
5. `cubecl-wgpu/src/compute/server.rs` `pipeline()`: return `Vec<Arc<ComputePipeline>>` for a split kernel. `launch_persistent` registers one `ScheduleTask::Execute` per phase, with the same resources and count, on the same stream. WebGPU dispatch ordering gives the barrier.
6. Set `Emulated(Split)` on all wgpu backends.

**E2 Spin** (`GridSyncEmulation::Spin`)
1. Available only if `device_memory_scope` is `true` (Vulkan, Metal via wgpu). Never on WebGPU.
2. New pass `cubecl-opt/src/passes/lower_grid_sync_spin.rs`. It adds a hidden `&mut [Atomic<u32>]` workspace argument, a discovery prologue (cubes after the poll close exit), and replaces each `GridSyncOp` with: `sync_storage`; unit 0 does a sense-reversal barrier on the workspace with atomics; `sync_storage`. Only unit 0 spins.
3. The count that the kernel reads as `CUBE_COUNT` must be the discovered count. Store it in the workspace and redirect `CUBE_COUNT` reads in the pass.
4. `cubecl-wgpu` server: allocate one workspace per stream. Zero it with a buffer write before each Spin launch. Append it as the last binding in `prepare_bindings`.

#### Metal native (`cubecl-metal`), CPU (`cubecl-cpu`)

- Tier A: default `launch_persistent`. Capacity: Metal uses D6. CPU uses `num_cpu_cores`.
- Tier B: Metal reuses E1 and E2 through the `cubecl-opt` passes. The server change mirrors wgpu step E1.5 in [`metal server.rs` L322](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-metal/src/compute/server.rs#L322-L330). CPU: report `GridSync::None` in the first milestones.

## 4. Milestones

Each milestone compiles, passes `cargo xtask` checks, and is releasable alone.

| M | Content | Depends on |
|---|---|---|
| M1 | 3.5 steps 1–7, `PersistentCount`, default `launch_persistent`, `persistent_range*`. Tier A on all runtimes. | — |
| M2 | `GridSyncOp`, `sync_grid*`, V1–V3, errors 3.6. All backends report `GridSync::None`. | M1 |
| M3 | CUDA and HIP native Tier B, V4, V5. | M2 |
| M4 | E1 Split on wgpu and Metal. | M2, D1, D2 |
| M5 | E2 Spin on wgpu (Vulkan, Metal) and Metal. | M2, D1 |
| M6 | `WorkQueue`. | M1, D5 |

## 5. Tests

Add `crates/cubecl-core/src/runtime_tests/persistent.rs`. Register it in `runtime_tests/mod.rs` next to `synchronization` ([L37](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-core/src/runtime_tests/mod.rs#L37)) and in `testgen_all!`.

1. Tier A: each item is processed exactly one time, for `Fill`, `AtMost(1)`, and `Exact(3)`.
2. Tier B: a two-phase reduction across cubes. Skip with a message if `grid_sync == None`, as the existing cross-cube test does.
3. Repeat test 2 for 1000 launches, to detect a deadlock or a stale counter (P1, P5).
4. Compile errors: V1, V2, V3, S1, S2. Use `create_dummy_kernel` to compile without a launch.
5. CUDA and HIP only: `Exact(capacity + 1)` returns `CooperativeGrid`. The capacity equals the value from the driver occupancy API.
6. CUDA only: capture a Tier B launch in a graph and replay it (P9).

## 6. Documentation

Add a chapter `cubecl-book/src/advanced-usage/persistent-kernels.md` and list it in `cubecl-book/src/SUMMARY.md`. Include the pitfall table (1.1), the tier model, the per-backend support matrix, and the watchdog warning (P6).
