# Persistent kernels: open decisions

- **Base commit:** [`a1bb768ce919260eea56dbd0b59c70e55236e22d`](https://github.com/skewballfox/cubecl/commit/a1bb768ce919260eea56dbd0b59c70e55236e22d).
- **Plan:** [`PLAN.md`](./PLAN.md). Pitfall IDs (`Pn`), milestone IDs (`Mn`) and section numbers refer to the plan.
- **Scope:** This document lists only choices where no option is clearly better. A closed decision moves to `PLAN.md` section 3.2 and leaves this document. The IDs do not change. D1, D2, D3, D4, D6 and D8 are closed.

---

## D5. Reset of the `WorkQueue` counter

Blocks: M6. Recommendation: **A**, with a runtime-owned counter. Status: waiting for confirmation.

### Problem

A `WorkQueue` gives each cube the next work item with `fetch_add` on a counter in device memory. The counter must be `0` when a launch starts. If it is not `0`, the launch skips work items (P5). This error is silent: the result is wrong, and no error occurs.

### Options

**A. The runtime writes zero before each launch.** The counter is in the launch workspace (plan 3.8). The server zeroes it on the same stream, just before the kernel.

**B. Epoch tickets.** The counter is never reset. The host keeps a running total of the tickets that earlier launches used, and passes it to the kernel as a base. A cube computes `item = fetch_add(1) - base`. Each launch uses exactly `total + cube_count` tickets, because each cube stops after its first failed fetch. So the host can compute the next base.

**C. The last cube resets the counter.** A second counter counts the cubes that finish. The last cube to finish writes `0` to both counters.

### Comparison

| Criterion | A. Runtime writes zero | B. Epoch tickets | C. Last cube resets |
|---|---|---|---|
| Correct after a normal launch | Yes | Yes | Yes |
| Correct if a cube exits early (early `return`, a bounds check) | Yes | No: the ticket count differs from the host total. All later launches are wrong. | No: the last cube never arrives. The next launch is wrong. |
| Correct after a skipped launch (tainted input) | Yes | No, unless the host also knows about the skip. | Yes: the kernel did not run, so the counter is still `0`. |
| Correct after a device fault | Yes | No | No. On CUDA the context is lost anyway. |
| CUDA graph replay | Yes, if the zero write is captured with the kernel. | No: the base is a scalar that the graph records. Each replay uses the same base. | Yes |
| Works on WebGPU | Yes | Only with 32-bit tickets, which wrap after 2³² tickets. WebGPU has no 64-bit atomics. | Yes |
| Extra device work per launch | One small fill (about 1–3 µs on CUDA; one `clear_buffer` in the same wgpu encoder). | None | One more atomic per cube. |
| User can make a mistake | No: the counter is hidden. | Yes: a user who shares a queue between kernels breaks the total. | Yes: any early exit. |
| Implementation cost | Low. It reuses the launch workspace that Spin and Split already need. | Medium: host-side state per queue. | Low |

### Analysis

- **B has the most failure modes.** It fails silently and permanently after one irregular launch, it breaks graph replay, and it has no 64-bit path on WebGPU. It is the worst option for users.
- **C fails on a normal pattern.** An early exit is common in cubecl kernels. The error appears in the *next* launch, not in the launch with the bug. This makes it hard to find.
- **A fails in no case in the table.** Its only cost is one small fill per launch. A persistent kernel exists to replace many launches with one long launch. So the fixed cost of the fill is small compared to the kernel run time. The launch workspace already exists for E1 spill and E2 (plan 3.8), so A adds almost no code.

### Recommendation

Use **A**. It is the least bad option: its cost is small and fixed, and the other options fail silently. Give experts a way out of the fill:

- `WorkQueue::from_buffer(unsafe …)`: the user supplies the counter buffer and resets it. The API documents P5.

If you accept A, move D5 to plan section 3.2. Then add "the server zeroes the `WorkQueue` counters in the launch workspace" to plan 3.8, and remove "D5" from M6.

---

## D7. Grid sync on the LLVM `nvptx` and `amdgpu` targets

`cooperative_groups` is a C++ header library. The LLVM targets cannot include it. Blocks: none. The plan reports `GridSync::None` for these targets until this decision is closed.

| Option | For | Against |
|---|---|---|
| **A. Software barrier in the launch workspace**, with a cooperative launch | Reuses the E2 barrier without the discovery prologue. The driver gives co-residency. | A hidden binding on a native backend. A little slower than the driver barrier. |
| **B. Emit the same PTX / AMDGCN sequence as `grid.sync()`** (driver grid workspace) | Same cost as the C++ path. | Depends on undocumented driver details. A driver update can break it. |
| **C. Not supported** | No work. | LLVM users do not get Tier B. |
