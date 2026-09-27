# Persistent kernels: open decisions

- **Base commit:** [`a1bb768ce919260eea56dbd0b59c70e55236e22d`](https://github.com/skewballfox/cubecl/commit/a1bb768ce919260eea56dbd0b59c70e55236e22d).
- **Plan:** [`PLAN.md`](./PLAN.md). Pitfall IDs (`Pn`), milestone IDs (`Mn`) and section numbers refer to the plan.
- **Scope:** This document lists only choices where no option is clearly better. A closed decision moves to `PLAN.md` section 3.2 and leaves this document. The IDs do not change. D1, D2, D3, D4, D5, D6 and D8 are closed. D7 is open. D9 is decided, but the code does not exist yet. D9 moves to `PLAN.md` section 3.2 when its code lands.

---

## D7. Grid sync on the LLVM `nvptx` and `amdgpu` targets

Blocks: native Tier B in default builds (part of M3). Until this decision is closed, the plan reports `GridSync::None` when an LLVM target compiles the kernel. M1, M2 and the C++ half of M3 do not wait for it.

### Facts

- The C++ path lowers `GridSyncOp` to `cooperative_groups::this_grid().sync()`. This is header code. The LLVM targets cannot include it.
- CUDA and HIP select their compiler at build time: `Cpp` with the `cpp` feature, else `Llvm` ([CUDA `compiler.rs` L17-L32](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cuda/src/compiler.rs#L17-L32), [HIP `compiler.rs` L16-L31](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-hip/src/compiler.rs#L16-L31)). The `cpp` feature is not a default feature. So **LLVM is the default compiler for CUDA and HIP**, and a default build has no NVRTC or HIPRTC path.
- On NVIDIA, `grid.sync()` uses a barrier word in a grid workspace. The driver gives its address in the registers `%envreg1` and `%envreg2` of a cooperative launch. PTX documents these registers only as "driver-defined".
- On AMD, `grid.sync()` calls `__ockl_grid_sync()` from the ROCm device library `ockl`. It reads the hidden kernel argument `multigrid_sync_arg`. `hipModuleLaunchCooperativeKernel` fills this argument.
- The LLVM `amdgpu` target already links ROCm device libraries on demand ([`codegen.rs` L247-L262](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/amdgpu/codegen.rs#L247-L262)).
- Every option below uses a cooperative launch (plan 3.8, CUDA step 2). So co-residency is guaranteed in every option.

### Options

**A. Software barrier in the launch workspace.** Lower `GridSyncOp` to the E2 barrier without the discovery prologue: `sync_storage`; unit 0 does `fetch_add` on a workspace counter and spins until the counter reaches the next multiple of `CUBE_COUNT`; `sync_storage`. This is the algorithm of `grid.sync()`, with a different barrier word.

**B. Driver barrier.** Use the barrier that the driver prepares:
- `nvptx`: read `%envreg1` and `%envreg2` with inline PTX, and emit the `grid.sync()` sequence on that address.
- `amdgpu`: add `ockl` to `DeviceLibs`, and call `__ockl_grid_sync()`. Keep the hidden argument `multigrid_sync_arg` in the kernel signature.

**C. Compile cooperative kernels with the C++ path.** If the kernel is `Cooperative`, the CUDA and HIP runtimes send it to `cubecl-cpp`. Other kernels stay on LLVM.

**D. Not supported.** Keep `GridSync::None` on LLVM targets. Then native Tier B needs the `cpp` feature.

### Comparison

| Criterion | A. Software barrier | B. Driver barrier | C. C++ path | D. Not supported |
|---|---|---|---|---|
| Works on `nvptx` | Yes | Yes | Only with the `cpp` feature, which is not a default | No |
| Works on `amdgpu` | Yes | Yes | Only with the `cpp` feature, which is not a default | No |
| Relies on undocumented behaviour | No | `nvptx`: yes (`%envreg` contents). `amdgpu`: no (open-source `ockl`). | No | No |
| Risk from a driver update | None | `nvptx`: the barrier can break without an error. `amdgpu`: low. | None | None |
| Speed compared to the C++ path | Equal algorithm. One extra memset per launch, which `WorkQueue` already pays. | Equal | Equal | — |
| Extra runtime work | The CUDA and HIP servers must bind the launch workspace to LLVM kernels. | None | Both compilers in one build. A cooperative kernel does not get LLVM-only features. | None |
| Code shared with other parts of the plan | The E2 barrier and the launch workspace. | None | None | — |
| Same PTX / AMDGCN for the same kernel on both CUDA paths | No | Nearly | Yes | — |

Lean: A. It is one implementation for both targets, it uses only documented behaviour, and it reuses code that E2 needs. B on `amdgpu` is a good second option if A is too slow on AMD.

---

## D9. Native grid sync on a device that other work shares

Status: decided (B + C). Not built. Blocks: M3 on HIP.

### Facts

- **Incident.** GPU: AMD Radeon iGPU of a Ryzen 5 PRO 5650U, gfx90c, 7 CUs, wave64. This GPU also drives the display. ROCm: 7.1.1. A standalone probe (hiprtc + `hipModuleLaunchCooperativeKernel`, no cubecl code) launched a `grid.sync()` kernel. At 256 threads and 32 KiB of dynamic shared memory, the occupancy query gave 2 blocks per CU, so the capacity was 14. The driver accepted a grid of 14. The kernel did not finish. The driver could not preempt the queue. It did 38 full (MODE2) GPU resets in 3 minutes. Each reset turned the display off. The compositor crashed. A reboot stopped the loop.
- The cubecl `persistent` tests passed (20/20) on the same device 20 s before the probe. Their grids are small.
- For every shape that completed, a count of `capacity + 1` or more returned `hipErrorCooperativeLaunchTooLarge`. So V4 and D4 work on HIP.
- On AMD, KFD queues of different processes run on the same CUs at the same time. The occupancy query sees only one kernel. So `occupancy × CU count` is an upper bound, not a guarantee of co-residency. If a cube of a cooperative grid cannot start, the resident cubes wait in `grid.sync()` forever.
- A deadlocked grid sync blocks preemption. The driver then resets the device. A reset destroys every context on the device, including the contexts of other processes and of the display.
- On CUDA without MPS, contexts are time-sliced, so a cooperative grid gets the whole device. This was not tested here. MPS and green contexts share SMs, so the same risk applies there.
- Spin (E2) also needs co-residency, so Spin is not a safe fallback. D1 already makes Spin `unsafe`. Split (E1) needs no co-residency. So Split is the only safe Tier B path on a shared device.
- Detection signals on the gfx90c: `hipDeviceAttributeKernelExecTimeout = 0` while the device drives a display, and `hipDeviceAttributeIntegrated = 1`. So on ROCm, the watchdog attribute does not detect a display. On CUDA, `cudaDevAttrKernelExecTimeout` is `1` when a display uses the device.
- Precedents for `unsafe`: D1 (a Spin opt-in makes the launch functions `unsafe`), `launch_unchecked`, and `next_work_item_from` (D5).

### Options

**A. Headroom.** `Fill`, `Fraction` and `AtMost` resolve against a reduced capacity on a shared device, for example `(CUs − r) × occupancy`.

**B. Gate.** The runtime reports `GridSync::Native` only on a device that no other work shares. On a shared device, it reports `GridSync::Emulated(Split)`.

**C. `unsafe` per-call opt-in.** A new `unsafe` launch function ignores the gate and uses the full capacity. Its `# Safety` section states the co-residency contract.

**D. Runtime policy.** A configuration value or an environment variable selects conservative or full behaviour.

**E. Bounded spin.** A spin barrier with a timeout sets an error flag and exits.

### Comparison

| Criterion | A. Headroom | B. Gate | C. `unsafe` opt-in | D. Policy | E. Bounded spin |
|---|---|---|---|---|---|
| Prevents the incident | Lowers the risk. It does not remove it. | Yes, if the detection finds the display. | Not by itself. | Only if the value is conservative. | Recovers. It does not prevent. |
| Full capacity on a headless farm GPU | Yes, with `r = 0` | Yes, with no configuration | Yes | Yes, with configuration | No, it is slower |
| Explicit at the call site | No | No | Yes | No | No |
| Can bypass a safety contract | No | No | Only in `unsafe` code | Yes, from the environment | No |
| Work | Small | Detection per runtime | New launch functions | Configuration only | Needs E2 discovery (M5), and a timeout on every barrier |

### Decision: B + C

**B. The gate.**
1. The runtime treats a device as shared if one of these is true:
   - `Integrated` is `1`.
   - `KernelExecTimeout` is `1`.
   - On Linux with an AMD device: the DRM card with the device's PCI bus ID has a connector whose `status` is `connected`.
   - The runtime cannot find the facts above (for example, HIP on a platform other than Linux).
2. On a shared device, the runtime reports `GridSync::Emulated(Split)`, not `GridSync::Native`. So `launch_persistent` runs a Tier B kernel with Split. The kernel still runs, but slower.
3. On a device that is not shared, nothing changes.
4. The runtime keeps the hardware fact (cooperative launch is available) in a new field, separate from `grid_sync`. The field name is chosen with the code.

**C. The opt-in.**
1. Add new items. Do not change existing signatures:
   - `unsafe fn launch_persistent_unchecked` on the client (next to `launch_persistent`).
   - The same function on the kernel launcher.
2. The opt-in is per call, not per kernel. A reviewer finds each use at the call site. This is different from D1, which sets Spin per kernel with a macro flag.
3. The function uses native grid sync if the hardware supports a cooperative launch, even on a shared device. It uses the full capacity (`occupancy × CU count`).
4. V4 stays: `Exact(n)` with `n > capacity` fails before the driver call. D4 stays: for `Fill`, `Fraction` and `AtMost`, the runtime halves the count when the driver rejects it.
5. If the hardware has no cooperative launch, the function does the same thing as `launch_persistent`.
6. The `# Safety` text: "No other workload may use the compute units of the device during the launch. This includes a display and other processes. Else some cubes cannot start, the grid deadlocks, and the driver resets the device. A reset destroys every context on the device."

**Rejected.** A: it lowers the risk but does not remove it, and B + C already cover both groups of users. D: an environment variable cannot be `unsafe`, so it would bypass C. A configuration value that can only make the behaviour stricter (for example, force Split everywhere) is acceptable later. E: Spin needs co-residency too, and a timeout does not remove the reset risk.

**Residual risk.** B does not detect every shared device. Examples: a discrete GPU that renders for a display through a different GPU (PRIME), and a headless GPU that other tenants share. For these, a user can force Split later with the stricter configuration value above.

**Tests.**
- A test that needs native grid sync on a shared device must use `launch_persistent_unchecked`. It must skip when the device is shared, because the test cannot guarantee the safety contract. This is a precondition of the test, not a weaker test.
- A new test checks that a shared device reports `Emulated(Split)`, and that the safe Tier B tests pass through Split on that device.

