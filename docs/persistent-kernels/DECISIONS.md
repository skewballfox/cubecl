# Persistent kernels: open decisions

- **Base commit:** [`a1bb768ce919260eea56dbd0b59c70e55236e22d`](https://github.com/skewballfox/cubecl/commit/a1bb768ce919260eea56dbd0b59c70e55236e22d).
- **Plan:** [`PLAN.md`](./PLAN.md). Pitfall IDs (`Pn`), milestone IDs (`Mn`) and section numbers refer to the plan.
- **Scope:** This document lists only choices where no option is clearly better. A closed decision moves to `PLAN.md` section 3.2 and leaves this document. The IDs do not change. D1, D2, D3, D4, D5, D6, D8 and D9 are closed. Only D7 is open. The full text of D9 (facts, options, comparison) is in commit [`ad9c513`](https://github.com/skewballfox/cubecl/blob/ad9c51305/docs/persistent-kernels/DECISIONS.md).

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
