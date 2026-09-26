# Persistent kernels: open decisions

- **Base commit:** [`a1bb768ce919260eea56dbd0b59c70e55236e22d`](https://github.com/skewballfox/cubecl/commit/a1bb768ce919260eea56dbd0b59c70e55236e22d).
- **Plan:** [`PLAN.md`](./PLAN.md). Pitfall IDs (`Pn`) and milestone IDs (`Mn`) refer to the plan.
- **Scope:** This document lists only choices where no option is clearly better. Each decision states the options, their trade-offs, and what it blocks. A "Lean" line is a weak preference, not a recommendation.

---

## D1. Default grid-sync emulation on wgpu and Metal

Blocks: M4, M5.

| Option | For | Against |
|---|---|---|
| **A. Split only** | Always correct under the WebGPU specification. Works in browsers. No hidden buffer. | No `sync_grid` in runtime loops (constraint S1). Registers and shared memory do not survive (P10). One dispatch per phase. The most compiler work. |
| **B. Spin only** (where `device_memory_scope`) | Same semantics as CUDA. Runtime loops work. Low cost per sync. | Relies on occupancy-bound scheduling, which no specification guarantees. A hang is possible on untested drivers. Not available on WebGPU. |
| **C. Split default, Spin opt-in** | The safe API stays deadlock-free. Experts get the fast path. | Two implementations to maintain. Kernels that run on CUDA can fail S1 on wgpu by default. |
| **D. Spin default where available, Split elsewhere** | Best portability of source code. | The safe API can hang on some drivers. This conflicts with principle 3.1.2. |

Lean: C.

## D2. Shared memory contents after `sync_grid`

Blocks: M3 (semantics), M4.

| Option | For | Against |
|---|---|---|
| **A. Undefined on every backend** | One portable rule. Split emulation is simple. | CUDA users lose a real advantage of persistent kernels. Code that works on CUDA can read garbage elsewhere unless `Validate` mode poisons it. |
| **B. Preserved where native; undefined in emulation** | No cost on CUDA and HIP. | Silent divergence between backends (P10). Tests on CUDA do not catch it. |
| **C. Preserved everywhere** (Split spills shared memory to a hidden buffer and restores it) | One rule, full semantics. | Hidden global traffic of `shared_size × cube_count` bytes per sync. More compiler work. |

The same question applies to registers. For registers, constraint S2 already forces an explicit answer at compile time.

## D3. Meaning of `PersistentCount::Fill`

Blocks: M1.

| Option | For | Against |
|---|---|---|
| **A. Full capacity** (occupancy × SMs) | Maximum latency hiding. Matches the CUDA samples. | No room for kernels on other streams (P8). The cube count changes with the register count between compiler versions. |
| **B. One cube per SM** | Stable count. Leaves room for other work. Good for kernels that use a lot of shared memory. | Low occupancy for small cubes. |
| **C. Capacity × fraction** (config value, default < 1) | Tunable. | One more configuration value. The best fraction depends on the workload. |

## D4. Native cooperative launch is not possible at run time

Cases: the device attribute is `0`, MPS or MIG limits the SMs, or `Exact(n)` exceeds the capacity. Blocks: M3.

| Option | For | Against |
|---|---|---|
| **A. Fail the launch** | Explicit. No surprise in performance or semantics. | The user must handle a new error. |
| **B. Fall back to emulation** | The kernel runs. | Different semantics (D2) and performance on the same device. Hard to diagnose. |
| **C. Clamp the count** (for `Fill` and `AtMost` only) and fail only for `Exact` | Correct for well-written kernels. | Hides a capacity drop from the user. |

## D5. Reset of the `WorkQueue` counter

Blocks: M6.

| Option | For | Against |
|---|---|---|
| **A. Host writes zero before each launch** | Simple. Always correct. | One extra operation per launch. The graph replay must include it. |
| **B. Epoch tickets** (the launch passes an epoch; the counter is compared to `epoch × total`) | No extra operation. | Needs 64-bit atomics or wrap handling. More complex code. |
| **C. The last cube resets the counter** | No extra operation. No 64-bit need. | A failed or aborted launch leaves a bad counter. The next launch is then wrong (P5). |

## D6. Capacity estimate on runtimes without an occupancy query

Runtimes: wgpu, Metal. Tier A is correct with any count, so this decision affects only performance. For Spin, the discovery protocol corrects the count at run time. Blocks: M1.

| Option | For | Against |
|---|---|---|
| **A. Fixed constant** (for example 4 × a default core count) | Deterministic. No maintenance. | Too low on large GPUs, too high on small GPUs. |
| **B. Vendor table** from adapter information | Better values on known devices. | Maintenance of the table. Unknown devices need A as fallback. |
| **C. Autotune** with the existing tuner | Measured value. | Extra launches at first use. The result depends on the kernel. |

## D7. Grid sync on the LLVM `nvptx` and `amdgpu` targets

`cooperative_groups` is a C++ header library. The LLVM targets cannot include it. Blocks: none (the plan reports `GridSync::None` until this is closed).

| Option | For | Against |
|---|---|---|
| **A. Software barrier with a hidden workspace**, launched cooperatively | Reuses the E2 pass without discovery. Co-residency comes from the driver. | Hidden binding on a native backend. A little slower than the driver barrier. |
| **B. Emit the same PTX / AMDGCN sequence as `grid.sync()`** (driver grid workspace) | Same cost as the C++ path. | Depends on undocumented driver details. Can break with a driver update. |
| **C. Not supported** | No work. | LLVM users do not get Tier B. |

## D8. Macro surface: one flag or two

Blocks: M1.

| Option | For | Against |
|---|---|---|
| **A. One flag `persistent`.** The server detects `GridSyncOp` in the compiled kernel. | Short. One concept for the user. | The cost of Tier B (cooperative count limit, split compile) is not visible at the definition. |
| **B. Two flags `persistent` and `cooperative`.** Only `cooperative` permits `sync_grid`. | The cost is explicit. V1 is more precise. | More API surface. Users must learn two terms. |
