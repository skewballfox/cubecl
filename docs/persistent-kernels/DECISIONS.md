# Persistent kernels: open decisions

- **Base commit:** [`a1bb768ce919260eea56dbd0b59c70e55236e22d`](https://github.com/skewballfox/cubecl/commit/a1bb768ce919260eea56dbd0b59c70e55236e22d).
- **Plan:** [`PLAN.md`](./PLAN.md). Pitfall IDs (`Pn`), milestone IDs (`Mn`) and section numbers refer to the plan.
- **Scope:** This document lists only choices where no option is clearly better. A closed decision moves to `PLAN.md` section 3.2 and leaves this document. The IDs do not change. D1, D2, D3, D4, D5, D6 and D8 are closed. Only D7 is open.

---

## D7. Grid sync on the LLVM `nvptx` and `amdgpu` targets

`cooperative_groups` is a C++ header library. The LLVM targets cannot include it. Blocks: none. The plan reports `GridSync::None` for these targets until this decision is closed.

| Option | For | Against |
|---|---|---|
| **A. Software barrier in the launch workspace**, with a cooperative launch | Reuses the E2 barrier without the discovery prologue. The driver gives co-residency. | A hidden binding on a native backend. A little slower than the driver barrier. |
| **B. Emit the same PTX / AMDGCN sequence as `grid.sync()`** (driver grid workspace) | Same cost as the C++ path. | Depends on undocumented driver details. A driver update can break it. |
| **C. Not supported** | No work. | LLVM users do not get Tier B. |
