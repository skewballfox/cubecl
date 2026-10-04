# Phase 2, step 5: follow-up work in cubecl

| Item | Value |
|---|---|
| Plan step | [PLAN.md §6, step 5](PLAN.md#6-phase-2-keep-locations-through-passes-and-emit-dwarf-llvm-paths--cpu-and-gpu-done) |
| pliron change | Commit [`bb5544b`](https://github.com/skewballfox/pliron/commit/bb5544bd4494617081a8e287b17fac0a4950dbe6), the head of the branch `llvm-debug-info` of the fork. It will be a separate PR, on top of [pliron#290](https://github.com/pliron-org/pliron/pull/290). |
| Status | Done. The patch: [`6b529f4`](https://github.com/skewballfox/cubecl/commit/6b529f418440da900f5802402e5679afa28062ce). The cubecl part: [`630d6e7`](https://github.com/skewballfox/cubecl/commit/630d6e78e233e774f3ff57070582890613c800e3). |
| Language | ASD-STE100 / Attempto Controlled English. Domain terms are permitted. |

## 1. The pliron change

`pliron-llvm` now has a new field in `DebugInfoOptions`:

- `source_text: HMap<String, SourceText<'a>>`. The key is the path of the file in the location. `SourceText` has `text: &'a str` and `md5: String`, the MD5 of the text in hexadecimal. The `DIFile` of the file gets the text and the MD5.

The caller must give the MD5. The conversion does not calculate it, and it does not examine whether the MD5 agrees with its text. An earlier draft had a separate field `source_md5` and calculated a missing MD5. The commit `bb5544b` does not.

The type is `HMap`, not `BTreeMap` as in `pliron-spirv`. `pliron` does not use `BTreeMap`.

## 2. Facts that the work must obey

1. LLVM writes the text into the object only with DWARF 5. cubecl uses DWARF 4 now (the default of `DebugInfoOptions::dwarf_version`). With DWARF 4, the text is in the LLVM IR but not in the object.
2. **Do not give texts to NVPTX.** With a `DIFile` that has a text, `llc` writes `.file 1 "k.rs" source "…"` into the PTX. This occurs with DWARF 4 too. `ptxas` 12.9 rejects it: `Parsing error near 'source': syntax error`. The test used `ptxas` from the wheel `nvidia-cuda-nvcc-cu12`, in a container. The CUDA driver JIT (`cuModuleLoadDataEx`) was not tested.
3. On the CPU and on AMDGPU (`gfx1100`), `llc` writes the text with DWARF 5, and `llvm-dwarfdump` shows it. This is also correct when only some files have a text. Then LLVM writes no MD5 for any file. The macro gives a text only to `#[cube(debug_symbols)]` functions, or to all functions with `CUBECL_DEBUG=1`.
4. Cost of the MD5, measured with a synthetic module: 300 ops, release build. The cost occurs one time for each file in each module, not for each op.

   | Text of each file | MD5 calculated | MD5 given |
   |---|---|---|
   | 20 KB | approximately 36 µs | approximately 10–13 µs |
   | 200 KB | approximately 368 µs | approximately 118 µs |

   Without texts, the conversion of the module takes approximately 280 µs. The time of the rest of the kernel compilation was not measured.

## 3. Work in cubecl

- [x] **Step 1. Patch revision** ([`6b529f4`](https://github.com/skewballfox/cubecl/commit/6b529f418440da900f5802402e5679afa28062ce)). The `[patch."https://github.com/pliron-org/pliron.git"]` revision is `bb5544b`, the head of `llvm-debug-info`. Before, it was `8b27b0e` on `cubecl-patch`, which also merged `llvm-perf`. Upstream does not take `llvm-perf`, and cubecl does not use it (PLAN §7, option (c)). Later, when the upstream PR is merged, remove the patch (PLAN §6, "Before a merge").
- [x] **Step 2. Target parameter.** [`convert_module`](https://github.com/skewballfox/cubecl/blob/630d6e78e233e774f3ff57070582890613c800e3/crates/cubecl-llvm/src/shared/debug_info.rs#L17-L59) has the parameter `embed_source`. The CPU and AMDGPU give `true`. NVPTX gives `false` (fact 2).
- [x] **Step 3. Fill `source_text`.** At `Full` only, and only with `embed_source`. The pattern is the same as `cubecl-spirv/src/debug_info.rs::builder`.
  - **Verify result, true:** `ctx.try_aux_ty::<DebugState>()` returns the state on the LLVM paths. The CPU and AMDGPU tests find the text in the output.
- [x] **Step 4. DWARF 5.** With a text, `options.dwarf_version` is 5 (fact 1). Without a text, it stays 4.
  - **Verify result:** with `CUBECL_DEBUG=1`, the `cubecl-cpu` suite (812) passes with RuntimeDyld and the gdb listener, and with JITLink and the jitdump. All 1,073 jitdumps of the run have `DEBUG_INFO` and `UNWINDING_INFO` records. The gdb registration tests pass. The offline AMDGPU tests pass.
  - **Checked by hand later** (PLAN §6 step 5, §7 step 5): gdb shows the kernel lines and the inlined frames. The object has DWARF 5 with the MD5 and the source text, but gdb 17 does not read the source text. The hand checks found three errors in the jitdump of the ORC plugin, which [`2fec40f`](https://github.com/skewballfox/cubecl/commit/2fec40f3b2ef6aefcabf3915b3756e61436d7eb2) corrects.
  - **Not checked:** HIP did not run, because no AMD GPU is on the host.
- [x] **Step 5. MD5 at run time.** The commit `bb5544b` does not calculate the MD5, so this step is necessary. [`md5_hex`](https://github.com/skewballfox/cubecl/blob/630d6e78e233e774f3ff57070582890613c800e3/crates/cubecl-llvm/src/shared/debug_info.rs#L61-L73) calculates it at run time with `md-5`: approximately 25 µs for each file of 20 KB (fact 4). `md-5` adds approximately 2 KB to the `gelu` example. Alternative: `cubecl-macros` calculates the MD5 at build time. That needs the macro to read the file, an extra parameter of `debug_source_expand`, and a field in `DebugState`. Do it only if a measurement shows that the run-time cost is important.
- [x] **Step 6. Tests:** [`debug_info.rs#L254-L287`](https://github.com/skewballfox/cubecl/blob/630d6e78e233e774f3ff57070582890613c800e3/crates/cubecl-llvm/src/shared/debug_info.rs#L254-L287), [`offline_tests.rs#L72-L82`](https://github.com/skewballfox/cubecl/blob/630d6e78e233e774f3ff57070582890613c800e3/crates/cubecl-llvm/src/amdgpu/offline_tests.rs#L72-L82), [`offline_tests.rs#L80-L92`](https://github.com/skewballfox/cubecl/blob/630d6e78e233e774f3ff57070582890613c800e3/crates/cubecl-llvm/src/nvptx/offline_tests.rs#L80-L92). The kernel `scale_with_source` has `#[cube(debug_symbols)]`, so the macro embeds its file.
  - At `Full` on the CPU, the `DIFile` of the kernel file has `source:` and `checksumkind: CSK_MD5`. The module has `"Dwarf Version", i32 5`. The module verifies.
  - At `LineTables`, no `DIFile` has `source:`, and the DWARF version is 4.
  - At `Full` on AMDGPU, the `.file` directive of the kernel file has `md5` and `source`.
  - At `Full` on NVPTX, no `.file` directive has `source`. The test fails when NVPTX gets `true`.
- [x] **Step 7. Documentation.** PLAN §6 step 5 is marked done. The book page does not tell the `Full` level, so it does not change.

## 4. Not in this follow-up

The compile directory (`DebugInfoOptions::directory`) is still not set. §6 step 5 gives the reason: a proc macro cannot see `--remap-path-prefix` or `trim-paths`.
