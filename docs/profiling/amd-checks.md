# AMD: checks of the kernel debug data on an integrated GPU

| Item | Value |
|---|---|
| Plan steps | [PLAN.md §6, steps 4–5](PLAN.md#6-phase-2-keep-locations-through-passes-and-emit-dwarf-llvm-paths--cpu-and-gpu-done), PLAN.md §8, steps 2 and 4, [step-5-followup.md](step-5-followup.md) step 4 |
| Branch | `claude/upbeat-fermat-if47lz`. The Vulkan runs used `42474c2`. The HIP runs used `4dc0d1e`, the fix of §3.1. The commits between them change only documents. |
| Baseline | `2d92e72`, the merge base with `origin/main`. The local `main` is older (`a1bb768`), so `git merge-base HEAD main` does not give it. |
| Status | Vulkan: done. HIP: partial. The GPU hung in a test that forces a fault, and the compute queue did not recover (§2.2). The C++ backend and the debugger: not done. |
| Language | ASD-STE100 / Attempto Controlled English. Domain terms are permitted. |

## 1. The host

| Fact | Value |
|---|---|
| CPU and GPU | AMD Ryzen 5 PRO 5650U. The GPU is the integrated Vega GPU (Renoir/Cezanne), 7 compute units. |
| `gcnArchName` | `gfx90c:xnack+`. `HSA_OVERRIDE_GFX_VERSION` is not set. `rocminfo` finds the GPU without it. |
| Kernel | `7.2.8-200.fc44.x86_64` (Fedora 44) |
| Vulkan driver | RADV, Mesa 26.2.3 (`AMD Radeon Graphics (RADV RENOIR)`), Vulkan 1.4.354. llvmpipe is also present. |
| `VK_KHR_shader_non_semantic_info` | Listed by RADV. |
| Vulkan loader, validation layer | 1.4.341, 1.4.341 |
| ROCm | 7.1.1, the Fedora packages (`rocm-hip`, `rocm-runtime`, `rocm-llvm` 20). ROCm does not officially support `gfx90c`. |
| `spirv-val` | SPIRV-Tools v2026.1 |
| rocgdb | 16.3 (`rocm-gdb-16.3.70101-38.el10`) and `rocm-dbgapi-0.77.4`, from `repo.radeon.com/rocm/rhel10/7.1.1`. `rpmkeys --checksig` accepts the two packages with the AMD key `CA8BB472…1A693C5C`. It also needs `libpython3.12`, from the Fedora package `python3.12-libs-3.12.15` (signature OK). The packages were unpacked, not installed. |
| `LD_PRELOAD` | Empty. ptrace: Yama scope 0, `deny_ptrace` off. |

**Adapter:** cubecl requests `PowerPreference::HighPerformance` without a fallback adapter. Thus wgpu selects RADV, not llvmpipe.

## 2. Vulkan / wgpu

### 2.1 The suites

| Run | Branch | Baseline |
|---|---|---|
| `cargo test -p cubecl-wgpu` | 587 pass, 0 fail, 12 ignored | 587 pass, 0 fail, 12 ignored |
| `cargo test -p cubecl-wgpu --features spirv` | 824 pass, 0 fail, 17 ignored | 824 pass, 0 fail, 17 ignored |
| Same, `CUBECL_DEBUG=1`, `CUBECL_SPIRV_DEBUG_FORMAT=non-semantic` | 824 pass, 0 fail | – |
| Same, `CUBECL_DEBUG=1`, `CUBECL_SPIRV_DEBUG_FORMAT=op-line` | 803 of 803 `--lib` tests pass (see below) | – |

- **Checked:** the branch and the baseline run the same tests, and no test fails on either side. On the NVIDIA host, 90 tests failed on both sides. RADV has none of these failures.
- **Checked:** no test changes its result with debug data. A `dev` test build already has `LineTables`, and `auto` selects `NonSemantic` on RADV. Thus the run without `CUBECL_DEBUG` also loads debug data into the driver.
- `CUBECL_DEBUG` is read by `cubecl-macros/build.rs`, so it changes the build. The debug runs used a separate target directory (`CARGO_TARGET_DIR=target-dbg`).
- **Found:** the first `op-line` run stopped with SIGSEGV in the `--lib` binary after 748 passes. The stack is in the Vulkan loader, not in cubecl: `loader_get_icd_and_device` ← `terminator_SetDebugUtilsObjectNameEXT` ← `VkLayer_khronos_validation` ← `wgpu_hal` `begin_encoding` ← `queue_submit`. It names a command buffer. It does not touch a shader. The same binary then passed 11 times: once with one thread (803 pass), and five times with each format with parallel threads. The cause is a race in the loader or in the validation layer (bug 2 in §5).
- The validation layer was active in all runs, because wgpu enables it in a debug build when it is installed (`InstanceFlags::default()` is `from_build_config()`). The baseline has the same code.

### 2.2 The validation layer

Command: the debug build of the `--lib` binary, filter `tests_spirv`, with `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`, `VK_KHRONOS_VALIDATION_REPORT_FLAGS=error,warn`, `VK_KHRONOS_VALIDATION_ENABLE_MESSAGE_LIMIT=false`, and `--nocapture`. The messages go through the `tracing` subscriber of wgpu. libtest captures them for a test that passes, thus `--nocapture` is necessary.

| Run | Messages | Distinct |
|---|---|---|
| Branch, `CUBECL_DEBUG=1`, `non-semantic` | 20 | 3 |
| Branch, `CUBECL_DEBUG=1`, `op-line` | 20 | 3 |
| Branch, no `CUBECL_DEBUG` | 20 | 3 |
| Baseline | 20 | 3 |

- **Checked:** no message names `OpLine`, `OpExtInst`, `NonSemantic` or a `Debug*` instruction.
- **Found:** the three messages are the same on all four runs, thus they are older than this branch. All three are `spirv-val` errors in `vkCreateShaderModule` (bug 3 in §5):
  - `BuiltIn SubgroupLocalInvocationId cannot be used as a member decoration` (18 times).
  - `OpMemberDecorate %_struct_14 1 BuiltIn SubgroupId`: `Operand 4 of MemberDecorate requires one of these capabilities: Kernel GroupNonUniform`.
  - `OpControlBarrier %uint_3 %uint_3 %uint_136`: `Memory Semantics with a non-relaxed memory order … must have at least one Vulkan-supported storage class semantics bit set`.

### 2.3 `cubecl-spirv` with `CI`

- **Checked:** `CI=1 cargo test -p cubecl-spirv` passes on the branch (4 tests). The baseline has no tests in this crate.
- **Checked:** with `CI=1` and without `spirv-val` on the `PATH`, the tests fail at `debug_info.rs:190`. Thus the tests use `spirv-val`.

## 3. HIP, LLVM backend

### 3.1 The architecture table did not know `gfx90c`

**Found:** `GfxArch::parse` gave `AMDArchitecture::Other` for `gfx90c`, and `Other` has no wavefront width. `HipServer::init` asserts that the width of the table is the width of the driver (64). Thus 546 of the 549 `cubecl-hip` tests failed at the assert, on the branch and on the baseline, with the same list. No kernel ran.

**Fix:** [`4dc0d1e`](https://github.com/skewballfox/cubecl/commit/4dc0d1e2f) adds `AMDArchitecture::GFX9` for GCN5: `gfx900`, `gfx902`, `gfx904`, `gfx906`, `gfx909` and `gfx90c`. It has wave64, no WMMA and no MFMA. The names are a list, not a prefix, because `gfx908` and `gfx90a` also start with `gfx90` and have MFMA. The test `gcn5_is_not_cdna` examines this. The fix is in a separate commit because it is not debug work, and the baseline has the same error (bug 1 in §5). The checks below applied the same patch to the baseline worktree, without a commit.

### 3.2 The suites, and the GPU hang

| Run | Result |
|---|---|
| `cargo test -p cubecl-llvm --features amdgpu --lib` (offline) | 45 pass, 0 fail |
| `cargo test -p cubecl-hip`, branch with the fix, `--lib` | 537 pass, 1 fail, 11 ignored |
| Same, the integration test binaries | `device_fault`: GPU hang, then a GPU reset. `empty_read`, `graph`, `relocation`: GPU hang. Stopped. |
| `cargo test -p cubecl-hip`, the baseline with the fix | Not run. The GPU did not recover. |

- **Checked (partial):** the AMD driver loaded and ran the LLVM code objects of 537 tests, at `LineTables` (the `dev` profile). This is the first load of code objects with this debug data on an AMD device. `Full` (DWARF 5, MD5, the source text) did not run.
- The one failure is `test_split_window_closes_on_its_own_stream` (`profiling.rs:203`): "512 launches measured 133.142151ms against 901.92218ms for 32". It is a timing test. It was not run on the baseline, so it is not known if it is a regression.
- **Found:** `tests/device_fault.rs` writes 1 GiB past the end of a buffer on purpose, and it expects an "illegal address" error. On `gfx90c:xnack+`, the write causes a GPU hang (`HW Exception by GPU node-1 … reason: GPU Hang`), and the kernel driver resets the GPU. The GPU also drives the display, thus the display flashed. The test runs by default in `cargo test -p cubecl-hip` (bug 4 in §5).
- **Found:** after this reset, the compute queue did not recover. Each later HIP process caused `qcm fence wait loop timeout` and one more reset, also a process that ran only the `--lib` tests. Nine resets occurred, and each one wrote "GPU reset(n) succeeded" (bug 5 in §5).

**Caution:** on an integrated GPU that drives the display, do not run `cargo test -p cubecl-hip` without `--lib`. Run each HIP command with a guard that stops it at the first `GPU reset begin` in `journalctl -k`.

### 3.3 Debug levels, rocgdb, `llvm-dwarfdump`

- **Not checked:** the runs with `CUBECL_DEBUG=1` and `CUBECL_DEBUG_INFO=none`. The GPU did not recover (§3.2).
- **Not checked:** rocgdb on `profiling_kernels`. rocgdb 16.3 starts on this host (§1). The example now has the feature `hip` (`--no-default-features --features hip`).
- **Not checked:** `llvm-dwarfdump --debug-line` on a code object of the device. `/usr/lib64/rocm/llvm/bin/llvm-dwarfdump` is on this host. The offline AMDGPU tests (§3.2) examine the `.file` directives at `Full`.

## 4. HIP, C++ backend

- **Not checked:** `cargo test -p cubecl-hip --features cpp`, the debug levels, and rocgdb. The GPU did not recover (§3.2). For rocgdb, use `--no-default-features --features hip,cubecl/hip-cpp`.

## 5. Bugs to report upstream

| # | Project | Bug | Evidence |
|---|---|---|---|
| 1 | cubecl | GCN5 (`gfx900`–`gfx90c`) is `AMDArchitecture::Other`, thus HIP refuses each Vega APU. | §3.1. Also on `2d92e72`. Fixed on this branch in `4dc0d1e`. |
| 2 | Vulkan-Loader or the Khronos validation layer (suspected) | SIGSEGV in `loader_get_icd_and_device` in `SetDebugUtilsObjectNameEXT`, with parallel threads. One time in twelve runs. | §2.1. Loader and layer 1.4.341, RADV 26.2.3. |
| 3 | cubecl, SPIR-V | `spirv-val` in the validation layer rejects a member decoration with `SubgroupLocalInvocationId` or `SubgroupId`, and the memory semantics of one `OpControlBarrier`. | §2.2. Also on `2d92e72`. |
| 4 | cubecl | `crates/cubecl-hip/tests/device_fault.rs` forces a GPU fault by default. On an APU, this resets the GPU and the display. It must be opt-in. | §3.2. |
| 5 | amdgpu / KFD (suspected) | After the hang of bug 4, each new HIP process gets `qcm fence wait loop timeout` and a GPU reset, although each reset "succeeded". | §3.2. Kernel 7.2.8, ROCm 7.1.1. ROCm does not support `gfx90c`. Examine if a reboot clears it. |

## 6. Optional checks

- **lldb:** checked on a different host ([cuda-parity.md §5](cuda-parity.md#5-checks-on-the-device)). Not done here.
- **Not checked:** `rocprofv3 --kernel-trace`. The tool is not on this host.
