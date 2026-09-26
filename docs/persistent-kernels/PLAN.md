# Persistent kernels: implementation plan

- **Base commit:** [`a1bb768ce919260eea56dbd0b59c70e55236e22d`](https://github.com/skewballfox/cubecl/commit/a1bb768ce919260eea56dbd0b59c70e55236e22d) (`main`, 2026-09-25).
- **Language:** ASD-STE100 and Attempto Controlled English. Domain terms are permitted.
- **Decisions:** closed decisions are in 3.2. Open decisions are in [`DECISIONS.md`](./DECISIONS.md). This plan refers to a decision as `Dn`.
- **Status:** ✅ done, 🟡 partly done, ⬜ not started. "Code:" links point to the commit that implements the item: [`39b8e82`](https://github.com/skewballfox/cubecl/commit/39b8e82288fd8856b00bb87ee8240c016b540f62) for M1 and M2, [`7a065a0`](https://github.com/skewballfox/cubecl/commit/7a065a02f6ef4da039892a5d86d63daef0557f27) for M4 to M7. Items without a mark are design text. The branch `claude/vibrant-wright-7zpf9j` holds the implemented parts. Where the code differs from the first draft of this plan, the plan now describes the code.
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
| P1 | Deadlock | A spin barrier runs without co-residency. | `sync_grid` compiles only in a `cooperative` kernel (3.5 V1). Spin emulation makes the launch `unsafe` (3.3). |
| P2 | Deadlock | Some units skip a grid sync (divergent control flow). | Compile error if the block is not device-uniform (3.5 V2). |
| P3 | Launch failure or deadlock | The capacity is computed from wrong values (hard-coded SM count, stale register count, wrong dynamic shared memory, MPS/MIG). | Capacity comes from the compiled function, at launch time (3.6). If the driver still rejects the count, the runtime lowers it and warns (3.8). |
| P4 | Stale reads | A cube reads data from another cube without release/acquire. The compiler hoists a load out of a spin loop. | Grid sync includes device fences. Work-queue helpers use atomics only. |
| P5 | Wrong work distribution on the next launch | A work counter is not reset between launches. | The server zeroes the `WorkQueue` counter before each launch (3.8). The user-owned counter is `unsafe`. |
| P6 | Hang of the display or a driver reset | A long kernel hits the watchdog (TDR, WDDM). | Documentation only. No enforcement. |
| P7 | Less throughput | Too many grid syncs. A grid sync costs microseconds. | Documentation. Grid sync is explicit, never implicit. |
| P8 | Lost concurrency | A kernel that fills the device blocks other streams. | `fill_fraction` configuration value, `PersistentCount::{Fraction, AtMost}` (3.3). |
| P9 | Capture failure | A cooperative launch in a CUDA graph. | Use `cuLaunchKernelEx` + attribute. The attribute is kept in captured kernel nodes. |
| P10 | Divergent result between backends | Shared memory or registers survive a grid sync on CUDA, but not in an emulation. | Shared memory survives by default on every backend: native, Spin, and Split with `Spill` (3.8 E1). `Discard` is explicit. |

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

1. Do not change existing signatures. Add new items only. Exceptions: new error variants (3.7), and new public fields in `Features`, `KernelSettings` and `GlobalState`. Code that builds these structs with a struct literal must add the field; `Features` has `Default`, so `..Default::default()` keeps working.
2. The safe API must not deadlock (P1, P2). An option that can deadlock is `unsafe`.
3. Tier A works on every runtime from the first milestone.
4. Grid sync is explicit. The runtime never inserts it.
5. A kernel gives the same result on every backend by default (P10). An option that trades this for speed is explicit.

### 3.2 Decisions taken

These decisions were in `DECISIONS.md`. They are closed. The table gives the section that implements each decision.

| ID | Decision | Implemented in |
|---|---|---|
| D1 | Split is the default emulation. Spin is an opt-in, and the opt-in makes the launch functions `unsafe`. | 3.3, 3.8 |
| D2 | Shared memory survives a grid sync on native backends and with Spin. With Split, the kernel selects `spill` (default, survives) or `discard` (undefined). | 3.3, 3.8 E1 |
| D3 | `Fill` uses `capacity × fill_fraction`. `fill_fraction` is a configuration value. A launch can override it. | 3.3, 3.6 step 8 |
| D4 | For `Fill`, `Fraction` and `AtMost`, the runtime clamps the count to the capacity. If the driver still rejects the count, the runtime lowers it and logs a warning. For `Exact`, the launch fails. | 3.5 V4, 3.8 CUDA step 5 |
| D5 | The server zeroes the `WorkQueue` counter before each launch. An `unsafe` constructor lets the user supply and reset the counter. | 3.4, 3.8 |
| D6 | Without an occupancy query, a constant gives the capacity. A generic `CapacityHint` overrides it. `AutotunedCapacity` is one such hint. | 3.6 step 9 |
| D8 | Two macro flags: `persistent` (Tier A) and `cooperative` (Tier B). | 3.3, 3.6 step 7 |

### 3.3 User API (default path) ✅

Code: [`cubecl-runtime/src/persistent.rs` L10-L35](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-runtime/src/persistent.rs#L10-L35), [`cubecl-macros/src/parse/kernel.rs` L34-L41](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-macros/src/parse/kernel.rs#L34-L41), [`cubecl-macros/src/parse/kernel.rs` L51-L63](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-macros/src/parse/kernel.rs#L51-L63), [`cubecl-macros/src/parse/kernel.rs` L86-L115](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-macros/src/parse/kernel.rs#L86-L115), [`cubecl-macros/src/generate/launch.rs` L53-L233](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-macros/src/generate/launch.rs#L53-L233), [`cubecl-macros/src/lib.rs` L42-L48](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-macros/src/lib.rs#L42-L48), [`cubecl-core/src/prelude.rs` L20](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-core/src/prelude.rs#L20-L20).

```rust
// Tier A: no grid sync. Correct on every runtime.
#[cube(launch, persistent)]
fn scale(input: &[f32], output: &mut [f32]) {
    for i in persistent_range(input.len()) {   // cube-strided work loop
        output[i] = input[i] * 2.0;
    }
}

// Tier B: grid sync. Native on CUDA/HIP, Split emulation elsewhere.
#[cube(launch, cooperative)]
fn two_phase(input: &[f32], partial: &mut [f32], output: &mut [f32]) {
    /* phase 1 */
    sync_grid();
    /* phase 2 */
}

scale::launch_persistent(&client, PersistentCount::Fill, CubeDim::new_1d(256), input, output);
```

**Macro flags** (`#[cube(...)]`):

| Flag | Meaning |
|---|---|
| `persistent` | Tier A. Generates `launch_persistent` and `launch_persistent_with`. `sync_grid` is a compile error (V1). |
| `cooperative` | Tier B. Implies `persistent`. Permits `sync_grid`. The runtime uses a cooperative launch or an emulation. |
| `grid_sync_emulation = "spin"` | Valid only with `cooperative`. Selects Spin where the runtime supports it (3.8 E2). The generated launch functions become `unsafe fn` with a `# Safety` section: "The device must run all cubes of the launch at the same time. Else the launch can hang." |
| `shared_after_grid_sync = "discard"` | Valid only with `cooperative`. With Split, shared memory is undefined after `sync_grid`. The default is `"spill"`: shared memory survives. It has no effect on native backends and on Spin. |

If a key is given without `cooperative`, the macro fails with a compile error.

**Generated functions** (arguments equal to `launch`; `PersistentCount` replaces `CubeCount`):

- `launch_persistent(client, count, cube_dim, args…)`. Uses `DefaultCapacity`.
- `launch_persistent_with::<H: CapacityHint>(client, count, cube_dim, args…)`. Uses the hint `H`. Rust does not permit a default for a generic parameter of a function, so this is a separate function.
- `launch_persistent_unchecked` and `launch_persistent_with_unchecked`, if the kernel also has `launch_unchecked`.
- `capacity(client, cube_dim, args…) -> Result<Option<u32>, ServerError>`, and `capacity_unchecked` if the kernel has `launch_unchecked`. `None` means that the runtime cannot query its capacity.

**`PersistentCount`** (new, in `cubecl-runtime/src/server/base.rs`, next to `CubeCount`):

| Variant | Count | If the driver rejects the count of a cooperative launch (D4) |
|---|---|---|
| `Fill` | `capacity × config.persistent.fill_fraction` | Halve the count, launch again, warn one time. |
| `Fraction(f32)` | `capacity × f`, `0 < f ≤ 1`. Overrides the configuration for one launch. | Halve the count, launch again, warn one time. |
| `AtMost(u32)` | `min(n, capacity)`. This clamp is documented and does not warn. | Halve the count, launch again, warn one time. |
| `Exact(u32)` | `n`. If `n > capacity`, the launch fails before the driver call (V4). | Fail with `ResourceLimitError::CooperativeGrid`. |

Every count is at least `1` and at most `max_cube_count.0`. The kernel reads the chosen count from the existing `CUBE_COUNT` builtin.

### 3.4 Kernel primitives

| Item | Crate / file | Behaviour |
|---|---|---|
| ✅ `GridSyncOp` (`sync.sync_grid`) | `cubecl-ir/src/dialect/synchronization.rs`, after `SyncAsyncProxyOp` | A new op with one attribute, `checked: BoolAttr`. Do **not** add a `SyncScope::Grid` variant. Reason: 7 lowerings match `SyncScope` exhaustively, and `SyncScope` permits a weaker fallback. A grid sync has no weaker fallback. Implement `Synchronizes` with minimum and maximum `SyncScope::Device`. Implement `SideEffects` = `true`. Code: [`cubecl-ir/src/dialect/synchronization.rs` L68-L93](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-ir/src/dialect/synchronization.rs#L68-L93). |
| ✅ `sync_grid()` | `cubecl-core/src/frontend/synchronization.rs`, after `sync_storage` | Safe. Registers `GridSyncOp` with `checked = true`. Applies V1 and V3. Code: [`cubecl-core/src/frontend/synchronization.rs` L82-L91](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-core/src/frontend/synchronization.rs#L82-L91), [`cubecl-core/src/frontend/synchronization.rs` L111-L121](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-core/src/frontend/synchronization.rs#L111-L121). |
| ✅ `unsafe sync_grid_unchecked()` | same file | Registers `GridSyncOp` with `checked = false`. V1 and V2 do not apply. V3 applies. The caller guarantees co-residency. Code: [`cubecl-core/src/frontend/synchronization.rs` L74-L121](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-core/src/frontend/synchronization.rs#L74-L121). |
| ✅ `persistent_range(len)` | `cubecl-core/src/frontend/ranges.rs`, after `range_stepped` | Returns a cube-uniform range `CUBE_POS, CUBE_POS + CUBE_COUNT, …`. Cube-uniform indices keep `sync_cube` legal inside the loop. It is in `cubecl-core`, because it builds on the private fields of `SteppedRangeExpand`. Code: [`cubecl-core/src/frontend/ranges.rs` L562-L600](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-core/src/frontend/ranges.rs#L562-L600). |
| ✅ `persistent_range_units(len)` | same file | Unit-strided variant: `ABSOLUTE_POS` stepped by `CUBE_COUNT × CUBE_DIM`. Code: [`cubecl-core/src/frontend/ranges.rs` L562-L600](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-core/src/frontend/ranges.rs#L562-L600). |
| ✅ `next_work_item()` | `cubecl-core/src/frontend/workspace.rs` (new) | Takes the next item of a queue in the launch workspace. Unit 0 takes the item, and `sync_cube` shares it with the cube. The runtime zeroes the queue before each launch (D5). `unsafe next_work_item_from(counter: &Atomic<u32>)` uses a counter that the caller owns and resets. The `# Safety` section names P5. Functions replace the `WorkQueue` type of the first draft, because the frontend has no reference-holding cube types. Code: [`cubecl-core/src/frontend/workspace.rs` L1-L103](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-core/src/frontend/workspace.rs#L1-L103). |

### 3.5 Validation (trap avoidance)

| ID | Rule | Where | Pitfall |
|---|---|---|---|
| ✅ V1 | If a kernel contains a checked `GridSyncOp` and its persistence is not `Cooperative`, compilation fails. | `sync_grid::expand` pushes a scope error. Every compiler refuses a kernel with scope errors (for example [`cpp base.rs` L136-L148](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cpp/src/shared/base.rs#L136-L148)). So no compiler needs a change. The scope gets the persistence from the new field `GlobalState::persistence` in `cubecl-ir/src/scope.rs`. Code: [`cubecl-core/src/frontend/synchronization.rs` L82-L91](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-core/src/frontend/synchronization.rs#L82-L91), [`cubecl-ir/src/scope.rs` L160](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-ir/src/scope.rs#L160-L160). | P1 |
| 🟡 V2 | If control flow that can differ between units decides if a checked `GridSyncOp` runs, compilation fails. | `VerifyGridSyncPass` in `cubecl-opt/src/passes/verify_grid_sync.rs`. It walks from the op to the entry function. Every operand of each enclosing op, and every `branch.condition` in its regions, must be `Uniformity::Device` in the dynamic uniformity analysis ([L31-L40](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-opt/src/passes/uniformity.rs#L31-L40)). The analysis tracks loop-carried values only to cube scope, so the pass refuses `while` loops and accepts `for` loops with uniform bounds. The pass must run after `Mem2RegPass`. Done: the pass and its tests. Not done: calls from the compiler pipelines (M3), because no runtime supports grid sync yet. Code: [`cubecl-opt/src/passes/verify_grid_sync.rs` L1-L124](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-opt/src/passes/verify_grid_sync.rs#L1-L124). | P2 |
| ✅ V3 | If the runtime reports `GridSync::None`, a kernel that calls `sync_grid` or `sync_grid_unchecked` does not compile. | In the same place as V1. Code: [`cubecl-core/src/frontend/synchronization.rs` L111-L121](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-core/src/frontend/synchronization.rs#L111-L121). | P1 |
| V4 | `Exact(n)` with `n > capacity` on a cooperative launch fails with `ResourceLimitError::CooperativeGrid { requested, max }`. Other variants clamp (D4). | Server, before the driver call. | P3 |
| V5 | The capacity comes from the compiled function, with the real dynamic shared memory. It is never cached across `ExecutionMode`s. | `Server::capacity` (3.6). | P3 |
| ✅ V6 | `AutotunedCapacity` never runs the kernel twice on the same data. Each tuning launch is a launch that the caller asked for, so read-write buffers stay correct. This replaces the first draft, which refused kernels with read-write buffers. | `cubecl-runtime/src/persistent.rs`. Code: [`cubecl-runtime/src/persistent.rs` L126-L203](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-runtime/src/persistent.rs#L126-L203). | — |

### 3.6 Runtime layer (`cubecl-ir`, `cubecl-runtime`, `cubecl-core`, `cubecl-macros`)

1. ✅ `cubecl-ir/src/features.rs`: add `pub grid_sync: GridSync` to `Features`. Add `enum GridSync { #[default] None, Native, Emulated(EnumSet<GridSyncEmulation>) }` and `enum GridSyncEmulation { Split, Spin }`. `Default` keeps all current constructors valid. Code: [`cubecl-ir/src/features.rs` L38-L39](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-ir/src/features.rs#L38-L39), [`cubecl-ir/src/features.rs` L152-L173](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-ir/src/features.rs#L152-L173).
2. ✅ `cubecl-ir/src/settings.rs`: add `pub persistence: Persistence` to `KernelSettings` and a builder `persistence(…)`, next to `cluster_dim()` ([L133-L136](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-ir/src/settings.rs#L133-L136)). Types:
   - `enum Persistence { #[default] None, Persistent, Cooperative(CooperativeOptions) }`
   - `struct CooperativeOptions { emulation: GridSyncEmulation /* default Split */, shared: SharedAfterGridSync /* default Spill */ }`
   - `enum SharedAfterGridSync { Spill, Discard }`
   - Code: [`cubecl-ir/src/settings.rs` L78-L111](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-ir/src/settings.rs#L78-L111), [`cubecl-ir/src/settings.rs` L127-L128](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-ir/src/settings.rs#L127-L128), [`cubecl-ir/src/settings.rs` L176-L180](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-ir/src/settings.rs#L176-L180).
3. ✅ `cubecl-runtime/src/id.rs`: add `pub persistence: Persistence` to `KernelId` and a builder. The compile caches then keep the variants apart. The hash and the stable key include the persistence only when it is not `None`, so the cache keys of existing kernels do not change. The launch path reads the cooperative bit from the id, so the CUDA PTX cache format does not change. Code: [`cubecl-runtime/src/id.rs` L100-L101](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-runtime/src/id.rs#L100-L101), [`cubecl-runtime/src/id.rs` L228-L232](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-runtime/src/id.rs#L228-L232), [`cubecl-runtime/src/id.rs` L234-L240](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-runtime/src/id.rs#L234-L240), [`cubecl-runtime/src/id.rs` L176-L188](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-runtime/src/id.rs#L176-L188).
4. ✅ `cubecl-runtime/src/persistent.rs` (new module) holds `PersistentCount` (3.3) with `PersistentCount::resolve(capacity, max)`. `cubecl-runtime/src/server/base.rs`:
   - Add to `trait Server` ([L543](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/server/base.rs#L543)) two methods with default bodies:
     - `fn capacity(&mut self, kernel: Box<dyn CubeKernel>, stream_id) -> Result<Option<u32>, ServerError>`. The default returns `Ok(None)`: the runtime cannot query its capacity.
     - `unsafe fn launch_persistent(&mut self, kernel, count: PersistentCount, estimate: u32, bindings, stream_id, launch_mode)`. `estimate` is the count that the client resolved from the `CapacityHint` (step 9). The default body calls `launch` with `CubeCount::new_1d(estimate)`. A runtime with a capacity query resolves `count` again. This default is correct for Tier A on every backend. A `Cooperative` kernel on a backend without an override fails at V3.
   - Code: [`cubecl-runtime/src/persistent.rs` L10-L35](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-runtime/src/persistent.rs#L10-L35), [`cubecl-runtime/src/server/base.rs` L632-L663](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-runtime/src/server/base.rs#L632-L663).
5. ✅ `cubecl-runtime/src/client.rs`: add `Client::launch_persistent(kernel, count, capacity, bindings)` and `Client::capacity(kernel)`. Do not copy `launch_inner`. Change the private `launch_inner` to take a private `enum LaunchShape { Count(CubeCount), Persistent { count, estimate } }`. Dispatch to `state.launch` or `state.launch_persistent` at the three call sites ([L1130](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/client.rs#L1128-L1132), [L1161](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/client.rs#L1159-L1163), [L1186](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/client.rs#L1184-L1192)). The public `launch` stays unchanged. Code: [`cubecl-runtime/src/client.rs` L1266-L1291](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-runtime/src/client.rs#L1266-L1291), [`cubecl-runtime/src/client.rs` L1075-L1088](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-runtime/src/client.rs#L1075-L1088), [`cubecl-runtime/src/client.rs` L1806-L1814](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-runtime/src/client.rs#L1806-L1814), [`cubecl-runtime/src/client.rs` L1816-L1837](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-runtime/src/client.rs#L1816-L1837).
6. ✅ `cubecl-core/src/compute/launcher.rs`: add `KernelLauncher::launch_persistent::<H, K>(self, count, kernel, client)` and `KernelLauncher::capacity(self, kernel, client)` next to `launch` ([L66-L73](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-core/src/compute/launcher.rs#L66-L73)). It calls `H::capacity` on the calling thread, then `Client::launch_persistent`. Code: [`cubecl-core/src/compute/launcher.rs` L78-L100](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-core/src/compute/launcher.rs#L78-L100).
7. ✅ `cubecl-macros`:
   - `parse/kernel.rs` `KernelArgs` ([L21-L42](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/parse/kernel.rs#L21-L42)): add `persistent: Flag`, `cooperative: Flag`, `grid_sync_emulation: Option<GridSyncEmulation>`, `shared_after_grid_sync: Option<SharedAfterGridSync>` (darling enums, so unknown values fail). `persistent` and `cooperative` need `launch` or `launch_unchecked`. Reject the last two keys without `cooperative`.
   - `generate/kernel.rs`: move the settings builder into one method `settings()`, and remove its unused copy in `info_ty`. At [L421-L423](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/kernel.rs#L421-L423) and [L512-L514](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/kernel.rs#L512-L514): emit `.persistence(…)`. In `id()` ([L458-L470](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/kernel.rs#L458-L470)): chain `.persistence(self.settings.persistence)`.
   - `generate/launch.rs`: one method `launch_fn` generates every launch function, and `capacity_fn` the capacity functions. A `cooperative` kernel does not get the plain `launch`, because a plain count can exceed the capacity. Add the functions of 3.3 beside [L56-L137](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/launch.rs#L56-L137). Reuse `launch_body`. If `grid_sync_emulation = "spin"`, emit `unsafe fn` and the `# Safety` text of 3.3.
   - Code: [`cubecl-macros/src/parse/kernel.rs` L34-L41](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-macros/src/parse/kernel.rs#L34-L41), [`cubecl-macros/src/parse/kernel.rs` L51-L63](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-macros/src/parse/kernel.rs#L51-L63), [`cubecl-macros/src/parse/kernel.rs` L86-L115](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-macros/src/parse/kernel.rs#L86-L115), [`cubecl-macros/src/parse/kernel.rs` L478](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-macros/src/parse/kernel.rs#L478-L478), [`cubecl-macros/src/generate/kernel.rs` L482-L521](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-macros/src/generate/kernel.rs#L482-L521), [`cubecl-macros/src/generate/kernel.rs` L459](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-macros/src/generate/kernel.rs#L459-L459), [`cubecl-macros/src/generate/launch.rs` L53-L233](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-macros/src/generate/launch.rs#L53-L233), [`cubecl-macros/src/lib.rs` L42-L48](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-macros/src/lib.rs#L42-L48).
8. ✅ **Configuration (D3).** Add `crates/cubecl-runtime/src/config/persistent.rs` with `struct PersistentConfig { fill_fraction: f32 /* default 1.0 */ }`. Add `#[serde(default)] pub persistent: PersistentConfig` to `CubeClRuntimeConfig` ([base.rs L19-L46](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/config/base.rs#L19-L46)), and `pub mod persistent;` to `config/mod.rs`. If the value is not in `(0, 1]`, log a warning and use `1.0`. The default `1.0` gives the full capacity. A user with concurrent streams (P8) sets a lower value. Code: [`cubecl-runtime/src/config/persistent.rs` L1-L23](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-runtime/src/config/persistent.rs#L1-L23), [`cubecl-runtime/src/config/base.rs` L45-L47](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-runtime/src/config/base.rs#L45-L47), [`cubecl-runtime/src/persistent.rs` L61-L72](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-runtime/src/persistent.rs#L61-L72).
9. ✅ **Capacity hint (D6).** Add `crates/cubecl-runtime/src/persistent.rs`:
   - `pub trait CapacityHint { fn capacity(client: &Client, kernel: &dyn CubeKernel) -> u32; }`
   - `pub const DEFAULT_PERSISTENT_CUBES: u32 = 256;`
   - `pub struct DefaultCapacity;` returns `hardware.num_cpu_cores` if the runtime has it (one cube per core), else `DEFAULT_PERSISTENT_CUBES`. It does not call the server.
   - `pub struct AutotunedCapacity;` tunes online: the first launches of a kernel each use one candidate of `{64, 128, 256, 512, 1024}` (at most `max_cube_count.0`), and `Client::profile` times them. Later launches use the fastest. `CapacityHint::launch` (a provided method) lets a hint wrap the launch. It replaces the `LocalTuner` of the first draft, which needs to run the kernel more than one time on the same inputs. Code: [`cubecl-runtime/src/persistent.rs` L91-L108](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-runtime/src/persistent.rs#L91-L108), [`cubecl-runtime/src/persistent.rs` L126-L203](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-runtime/src/persistent.rs#L126-L203).
   - A backend whose `Server::capacity` returns `Some` ignores the hint for `Fill`, `Fraction` and `AtMost`. Exception: `AutotunedCapacity` on a native backend tunes over `{capacity / 8, capacity / 4, capacity / 2, capacity}`.
   - Code: [`cubecl-runtime/src/persistent.rs` L37-L59](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-runtime/src/persistent.rs#L37-L59).

### 3.7 Errors

⬜ Moved to M3, the first milestone that can return these errors. Add `ResourceLimitError::CooperativeGrid { requested: u32, max: u32, backtrace }` and `CompilationError` text for V1–V3, S1 and S2. `ResourceLimitError` is not `#[non_exhaustive]`, so a new variant breaks exhaustive `match` in downstream crates. Accept this, or mark the enum `#[non_exhaustive]` in the same release.

### 3.8 Backends

#### Launch workspace ✅ (used by E2 and `next_work_item`)

`Scope::launch_workspace()` declares 16 atomic `u32` counters as a buffer after the kernel's own buffers, the first time a kernel asks for them. The buffer has no metadata, so the info layout of the kernel does not change. The client expands a persistent kernel one time (`LaunchPlan`, cached by `KernelId`), sees if it declared the workspace, and then binds a zeroed buffer after the kernel's own buffers on each launch. Spill buffers follow it. The first draft let the server own the workspace; the client does it, so no backend changes. For the same reason, a `persistent` kernel no longer gets the plain `launch` function: every launch goes through `launch_persistent`. Code: [`cubecl-ir/src/scope.rs` L166-L167](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-ir/src/scope.rs#L166-L167), [`cubecl-ir/src/scope.rs` L762-L792](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-ir/src/scope.rs#L762-L792), [`cubecl-runtime/src/persistent.rs` L48-L89](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-runtime/src/persistent.rs#L48-L89), [`cubecl-runtime/src/server/base.rs` L1340-L1345](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-runtime/src/server/base.rs#L1340-L1345), [`cubecl-runtime/src/client.rs` L1266-L1288](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-runtime/src/client.rs#L1266-L1288), [`cubecl-macros/src/generate/launch.rs` L64-L98](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-macros/src/generate/launch.rs#L64-L98).

#### CUDA — native (`cubecl-cuda`, `cubecl-cpp`)

1. `runtime.rs`, after [L357-L359](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cuda/src/runtime.rs#L357-L359): read `CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH`. If it is `1`, set `features.grid_sync = GridSync::Native`.
2. `compute/context.rs` `execute_task` ([L540-L553](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cuda/src/compute/context.rs#L540-L553)): if `kernel_id.persistence` is `Cooperative`, call `cuLaunchKernelEx` with `CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 1` (P9). Else keep `launch_kernel`. The emulation options have no effect on CUDA.
3. `compute/context.rs`: add `fn capacity(&mut self, kernel_id) -> u32`. Compute `occupancy::max_active_block_per_multiprocessor(func, cube_dim.num_elems(), shared_mem_bytes) × SM count`. Set `MAX_DYNAMIC_SHARED_SIZE_BYTES` before the query, as `execute_task` does.
4. `compute/server.rs`: override `capacity` (compile, then query; `exact: true`).
5. `compute/server.rs`: override `launch_persistent`. Resolve the count (3.3). If the driver returns `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE` for `Fill`, `Fraction` or `AtMost` (for example under MPS), halve the count and launch again, until the launch succeeds or the count is `1`. Log `log::warn!` one time per `KernelId`, with the requested and the used count (D4). For `Exact`, do not retry.
6. `cubecl-cpp/src/cuda/dialect.rs`: add `cuda_op!(GridSyncOp, …)` → `cooperative_groups::this_grid().sync();`. Declare `cooperative_groups.h` with the same `includes` mechanism as [`builtin.rs` L20-L32](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cpp/src/cuda/builtin.rs#L20-L32).
7. LLVM `nvptx` target: see D7. LLVM is the default CUDA compiler; steps 1–6 apply only with the `cpp` feature. Until D7 is closed, report `GridSync::None` when `CudaBackend::default()` is `Llvm` (next to `restrict_features`, [`runtime.rs` L373-L376](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cuda/src/runtime.rs#L373-L376)).

#### HIP — native (`cubecl-hip`, `cubecl-cpp`)

Apply CUDA steps 1–7 with `hipDeviceAttributeCooperativeLaunch`, `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`, and `hipModuleLaunchCooperativeKernel` in `execute_task` ([L459](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-hip/src/compute/context.rs#L455-L480)). Lower `GridSyncOp` in `cubecl-cpp/src/hip/dialect.rs` to `cooperative_groups::this_grid().sync();` with `hip/hip_cooperative_groups.h`.

#### Emulation for wgpu, the CPU runtime and native Metal ✅

Tier A needs no change. Tier B uses E1 by default and E2 by opt-in (D1). Both work in the client and the frontend, so no backend server changes. wgpu reports `Emulated(Split)`, and adds `Spin` where `device_memory_scope` is `true` (SPIR-V, MSL). The CPU runtime reports `Emulated(Split)`. Native Metal reports both; it is not compiled or tested here, because it needs macOS. Code: [`cubecl-wgpu/src/runtime.rs` L900-L907](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-wgpu/src/runtime.rs#L900-L907), [`cubecl-wgpu/src/runtime.rs` L535](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-wgpu/src/runtime.rs#L535), [`cubecl-cpu/src/runtime.rs` L176](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-cpu/src/runtime.rs#L176), [`cubecl-metal/src/runtime.rs` L190-L191](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-metal/src/runtime.rs#L190-L191).

**E1 Split** (default) ✅
1. `split()` in `cubecl-runtime/src/persistent/split.rs` works on the expanded kernel. Phase `k` keeps the ops of its part and erases the rest. `SplitPhase` is phase `k` as a kernel of its own, with its own `KernelId`. `Client::launch_persistent` launches the phases on one stream, one after the other: the launch boundary is the barrier. The first draft split in a `cubecl-opt` pass and changed the wgpu server; this design works for every runtime without server changes. Code: [`cubecl-runtime/src/persistent/split.rs` L47-L105](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-runtime/src/persistent/split.rs#L47-L105), [`cubecl-runtime/src/persistent/split.rs` L107-L118](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-runtime/src/persistent/split.rs#L107-L118), [`cubecl-runtime/src/persistent/split.rs` L120-L209](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-runtime/src/persistent/split.rs#L120-L209).
2. Constraint S1: every grid sync must be in the top-level block of the entry function. Else the kernel does not compile, and the error names `grid_sync_emulation = "spin"`. Code: [`cubecl-runtime/src/persistent/split.rs` L351-L381](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-runtime/src/persistent/split.rs#L351-L381).
3. Constraint S2: a phase keeps the ops of earlier phases that compute values it uses, if they are pure. A value that an earlier phase loaded or stored (it was in registers) is a compile error. A local variable that an earlier phase used is a compile error. Code: [`cubecl-runtime/src/persistent/split.rs` L383-L415](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-runtime/src/persistent/split.rs#L383-L415).
4. Shared memory (D2):
   - `Spill` (default): a shared variable used in more than one phase gets a spill buffer after the launch workspace, sized by the client from the plan. A phase saves the variable at its end and restores it at its start, between two `sync_cube`. Atomic shared memory is a compile error; use `discard`. Code: [`cubecl-runtime/src/persistent/split.rs` L211-L292](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-runtime/src/persistent/split.rs#L211-L292), [`cubecl-runtime/src/persistent/split.rs` L294-L349](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-runtime/src/persistent/split.rs#L294-L349).
   - `Discard`: no copy. ⬜ Not done: the poison pattern in `ExecutionMode::Validate`.
5. Split does not need co-residency. Any count is correct.

**E2 Spin** (opt-in: `grid_sync_emulation = "spin"`, `unsafe` launch) 🟡
1. Available where the runtime reports `Spin`. Elsewhere the kernel splits.
2. `sync_grid` expands to a ticket barrier on counter 1 of the launch workspace: `sync_storage`; unit 0 takes a ticket and spins until the counter reaches the end of its round; `sync_storage`. The counter is never reset in a launch, so there is no ordering problem between two counters. Code: [`cubecl-core/src/frontend/synchronization.rs` L115-L138](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-core/src/frontend/synchronization.rs#L115-L138), [`cubecl-core/src/frontend/workspace.rs` L1-L103](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-core/src/frontend/workspace.rs#L1-L103).
3. Shared memory and registers survive, and `sync_grid` in loops works, because the kernel does not split.
4. ⬜ Not done: the discovery prologue. Without it, the caller must choose a count that the device runs at the same time (the `# Safety` section says so). Discovery needs a way to replace `CUBE_POS` and `CUBE_COUNT` in the whole kernel with the discovered values, and to end the cubes that arrive late. That is a larger frontend change.

#### Metal native (`cubecl-metal`), CPU (`cubecl-cpu`)

- ✅ Tier A: default `launch_persistent`. Capacity: both use the `CapacityHint`. `DefaultCapacity` gives one cube per core on the CPU. Code: [`cubecl-runtime/src/persistent.rs` L37-L59](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-runtime/src/persistent.rs#L37-L59).
- ✅ Tier B: see the section above. No Metal server change is needed. The CPU runtime reports `Emulated(Split)`: it has no device-scope memory ordering, so it has no spin barrier.

## 4. Milestones

Each milestone compiles, passes `cargo xtask` checks, and is releasable alone.

| M | Status | Content | Depends on |
|---|---|---|---|
| M1 | ✅ | 3.6 steps 1–9 without `AutotunedCapacity`, `PersistentCount`, default `launch_persistent`, `persistent_range*`. Tier A on all runtimes. | — |
| M2 | ✅ | `GridSyncOp`, `sync_grid*`, V1, V3, the V2 pass. All backends report `GridSync::None`. | M1 |
| M3 | ⬜ | CUDA and HIP native Tier B (C++ path), V4, V5, clamp and warn, errors 3.7. Call `VerifyGridSyncPass` after `Mem2RegPass` in the `cubecl-cpp` pipeline ([`base.rs` L282](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cpp/src/shared/base.rs#L280-L284)). Needs a CUDA or ROCm GPU to test. The LLVM path waits for D7. | M2 |
| M4 | ✅ | Launch workspace. E1 Split with `Spill` and `Discard` on wgpu, the CPU runtime and Metal. Not done: the `Discard` poison pattern in `Validate` mode. | M2 |
| M5 | 🟡 | E2 Spin on wgpu (SPIR-V, MSL) and Metal. Not done: discovery (E2.4). | M4 |
| M6 | ✅ | `next_work_item` (runtime-owned counter) and `unsafe next_work_item_from`. | M4 |
| M7 | ✅ | `AutotunedCapacity`, V6. | M1 |

## 5. Tests

Add `crates/cubecl-core/src/runtime_tests/persistent.rs`. Register it in `runtime_tests/mod.rs` next to `synchronization` ([L37](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-core/src/runtime_tests/mod.rs#L37)) and in `testgen_all!`.

1. ✅ Tier A: each item is processed exactly one time, for `Fill`, `Fraction(0.5)`, `AtMost(1)`, and `Exact(3)`, with `persistent_range` and `persistent_range_units`. The kernel sees the chosen `CUBE_COUNT`. `capacity()` does not fail. Passes on the CPU runtime. Code: [`cubecl-core/src/runtime_tests/persistent.rs` L7-L108](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-core/src/runtime_tests/persistent.rs#L7-L108), [`cubecl-core/src/runtime_tests/mod.rs` L159](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-core/src/runtime_tests/mod.rs#L159-L159).
2. ✅ Tier B: phase 1 reads what other cubes wrote in phase 0. Code: [`cubecl-core/src/runtime_tests/persistent.rs` L152-L163](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-core/src/runtime_tests/persistent.rs#L152-L163). Original text: a two-phase reduction across cubes. Skip with a message if `grid_sync == None`, as the existing cross-cube test does.
3. 🟡 A spin barrier in a loop of 50 rounds, and 20 launches of a work queue kernel (P1, P5). Code: [`cubecl-core/src/runtime_tests/persistent.rs` L246-L295](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-core/src/runtime_tests/persistent.rs#L246-L295), [`cubecl-core/src/runtime_tests/persistent.rs` L203-L244](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-core/src/runtime_tests/persistent.rs#L203-L244). Not done: 1000 launches. Original text: Repeat test 2 for 1000 launches, to detect a deadlock or a stale counter (P1, P5).
4. 🟡 D2. Done: `Spill` keeps shared memory across `sync_grid` on every runtime; `Discard` needs no spill buffer. Code: [`cubecl-core/src/runtime_tests/persistent.rs` L297-L357](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-core/src/runtime_tests/persistent.rs#L297-L357). Not done: the poison pattern. Original text: D2: a kernel writes shared memory, calls `sync_grid`, and reads the shared memory. With `Spill`, the result is equal on every backend. With `Discard` in `Validate` mode on Split, the result contains the poison pattern.
5. 🟡 Compile errors. Done: V1 and V3 (runtime tests with `create_dummy_kernel`); V2 (`crates/cubecl-opt/tests/grid_sync.rs`, on expanded `#[cube]` functions); a macro key without `cooperative`, and a `spin` launch without `unsafe` (`trybuild`, `crates/cubecl-core/tests/error/`); the generated signatures (`crates/cubecl-core/tests/pass/persistent_kernel.rs`). S1 and S2 are done too: [`cubecl-core/src/runtime_tests/persistent.rs` L359-L390](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-core/src/runtime_tests/persistent.rs#L359-L390). Code: [`cubecl-core/src/runtime_tests/persistent.rs` L110-L160](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-core/src/runtime_tests/persistent.rs#L110-L160), [`cubecl-opt/tests/grid_sync.rs` L1-L138](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-opt/tests/grid_sync.rs#L1-L138), [`cubecl-core/tests/error/persistent_keys_need_cooperative.rs` L1-L9](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-core/tests/error/persistent_keys_need_cooperative.rs#L1-L9), [`cubecl-core/tests/error/spin_launch_needs_unsafe.rs` L1-L14](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-core/tests/error/spin_launch_needs_unsafe.rs#L1-L14), [`cubecl-core/tests/pass/persistent_kernel.rs` L1-L26](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-core/tests/pass/persistent_kernel.rs#L1-L26).
6. CUDA and HIP only: `Exact(capacity + 1)` returns `CooperativeGrid`. The capacity equals the value from the driver occupancy API.
7. CUDA only: capture a Tier B launch in a graph and replay it (P9).
8. 🟡 Configuration. Done: unit tests of `PersistentCount::resolve` in `cubecl-runtime/src/persistent.rs` (fraction, invalid fraction, bounds, `Exact`). Not done: a test that reads `fill_fraction` from a configuration file. Code: [`cubecl-runtime/src/persistent.rs` L74-L105](https://github.com/skewballfox/cubecl/blob/39b8e82288fd8856b00bb87ee8240c016b540f62/crates/cubecl-runtime/src/persistent.rs#L74-L105).
9. ✅ `AutotunedCapacity`: every tuning launch does the caller's work, and the launches after the tuning use a candidate. Code: [`cubecl-core/src/runtime_tests/persistent.rs` L97-L124](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-core/src/runtime_tests/persistent.rs#L97-L124), [`cubecl-runtime/src/persistent.rs` L245-L256](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-runtime/src/persistent.rs#L245-L256).
10. ✅ `next_work_item`: 20 launches, each processes each item exactly one time, also when cube 0 exits early. Code: [`cubecl-core/src/runtime_tests/persistent.rs` L203-L244](https://github.com/skewballfox/cubecl/blob/7a065a02f6ef4da039892a5d86d63daef0557f27/crates/cubecl-core/src/runtime_tests/persistent.rs#L203-L244). Original text: launch a kernel 1000 times; every launch processes each item exactly one time. Include a kernel where some cubes exit early (P5).

## 6. Documentation

Add a chapter `cubecl-book/src/advanced-usage/persistent-kernels.md` and list it in `cubecl-book/src/SUMMARY.md`. Include the pitfall table (1.1), the tier model, the macro keys (3.3), the per-backend support matrix, the configuration value (3.6 step 8), and the watchdog warning (P6).
