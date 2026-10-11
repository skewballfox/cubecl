# Assessment: tested tools, Vulkan on this host, and the cost of debug data

| Item | Value |
|---|---|
| Date | 2026-10-05 |
| Branch | `claude/upbeat-fermat-if47lz`. §3 at `49c13b23f` ("docs: check lldb with the source root and the source cache"). §4 at `8192dd442` ("fix(macros): name a variable only when its type has CubeDebug"), because `cubek` does not compile at `49c13b23f` (§5.1). |
| Base | `main` at `2d92e720e` ("fix(throughput): answer a declined key without taking the device (#1710)") |
| cubek | `main` at `a8207f82` ("chore(deps): bump cubecl to 5e521ef"). `5e521ef` is 3 commits before the base. A cargo `[patch]` replaces each cubecl crate with the worktree of the commit under test. |
| Host | AMD Ryzen 7 5700X (8 cores, 16 threads, governor `powersave`), 31 GB RAM, NVIDIA GeForce RTX 4070, driver 580.178.04, Fedora 44 (secureblue), hardened malloc in `/etc/ld.so.preload` |
| Toolchain | rustc 1.99.0, cargo release profile (opt-level 3, no LTO) |
| Status | Not committed. Measured data and scripts: `~/Workspace/bench-target/scripts`. |
| Language | ASD-STE100 / Attempto Controlled English. Domain terms are permitted. |

## 1. Tools

"Works" means that a person or a test checked the tool on this branch. The source column gives the record of the check.

| Tool | Version | Runtime | Status | Result | Limits | Source |
|---|---|---|---|---|---|---|
| gdb | 17.2 | `cpu` | Works | Breakpoints on `#[cube]` lines. Each inlined `#[cube]` function is a frame (`square_third` → `doubled` → kernel → `run_kernel`). Source shows from the workspace, or with `CUBECL_SOURCE_ROOT`, or from the source cache. | gdb does not read the embedded source text (`DW_LNCT_LLVM_source`). | [PLAN §7 step 1](PLAN.md#7-phase-3-cpu-jit-symbols-for-gdb-perf-samply-and-cargo-flamegraph--done), [cuda-parity §5](cuda-parity.md#5-checks-on-the-device) |
| GDB JIT interface | – | `cpu` | Works, test | The tests find the kernel object in `__jit_debug_descriptor`, with RuntimeDyld and with JITLink. | – | `lljit.rs` tests (`1e092e4`) |
| lldb | 23.1.2 (Homebrew) | `cpu` | Works | Inline frames as `[inlined]`. `image lookup -v` gives the blocks. At `Full`, lldb reads the embedded source text, also outside the workspace. | Two kernels with the same relative path: lldb keeps the first file that it did not find. `CUBECL_SOURCE_ROOT` or the source cache removes this limit. | [cuda-parity §5](cuda-parity.md#5-checks-on-the-device) |
| perf (`record`, `report`, `script`, `inject --jit`) | 7.2 | `cpu` | Works, with three corrections in the shim | The perf map names each kernel. The jitdump gives source lines (`--sort srcline`). | Record with `--call-graph fp`. `--call-graph dwarf` cannot unwind a kernel, because perf uses jitdump unwind data only under `/tmp/jitted-`. No inlined frames. Set `addr2line.style = addr2line`. | [PLAN §7 step 5](PLAN.md#7-phase-3-cpu-jit-symbols-for-gdb-perf-samply-and-cargo-flamegraph--done) |
| samply | 0.13.1 | `cpu` | Works | Names the kernel below `run_kernel` from the jitdump. | Needs `CUBECL_JIT_SYMBOLS=perf`. No inlined frames. | PLAN §7 step 5 |
| cargo flamegraph | – | `cpu` | Works | `run_kernel → k_…` with 91–94% of the samples. | Needs `-c "record -F 997 --call-graph fp -g"` and `-C force-frame-pointers=yes`. | PLAN §7 steps 4, 5 |
| inferno | – | `cpu` | Works | `perf script \| inferno-collapse-perf \| inferno-flamegraph` shows the kernel frame. | Same as perf. | PLAN §7 step 5 |
| llvm-dwarfdump | LLVM 23 | `cpu` | Works | The object has DWARF 5, the MD5 and the source text at `Full`. | – | PLAN §6 step 5 |
| LLVM verifier | LLVM 23 (bundle 23.1.0-3) | `cpu` | Works, test | The `cubecl-cpu` suite (812) verifies the debug data of each kernel in debug builds. | – | PLAN §6 |
| Nsight Compute (`ncu`, `ncu-ui`) | 2025.3.1 | `cuda` (LLVM and NVRTC) | Works | Source page with `#[cube]` lines from the absolute path, `--import-source yes`, metrics for each line. The "Inline Functions" table shows `square_third` and `doubled`. | The command line and the Python report interface do not show inline names. NVRTC gives no inline frames. | [cuda-parity §5](cuda-parity.md#5-checks-on-the-device) |
| cuda-gdb | 13.0 | `cuda` (LLVM) | Works | Breakpoint on a `#[cube]` line stops at `main.rs:11 in square_third inlined from main.rs:17`. `list` shows the source. | `bt` shows one device frame. Set a breakpoint after the first launch. `x/i` needs `cuobjdump`. | cuda-parity §5 |
| ptxas, nvdisasm, CUDA driver JIT | 12.9, 13.4, driver 580 | `cuda` (LLVM) | Works | The SASS keeps the inline chain and the names. The cubin keeps the absolute path. | `ptxas` rejects source text in the PTX. | cuda-parity §2 |
| CUDA driver JIT (`cuLinkComplete`) and `nvdisasm -hex` | driver 580, nvdisasm 13.x | `cuda` (LLVM) | Works | The matmul kernel of §4.3 gives bit-identical SASS with and without line data. Script: `jit_cubin.py`. | – | §4.3 |
| Nsight Compute `--clock-control none` | 2025.3.1 | `cuda` | Works | Shows the SM clock of each launch. It found the clock states of §4.3. | – | §4.3 |
| NVRTC (pip wheel) | 12.x | `cuda` (C++) | Works | `#line` directives with absolute paths. The suite with `--features cpp` found a `switch` defect, which is corrected. | One function body, so no inline frames. | PLAN §8 step 1 |
| spirv-val | SDK 1.4 | `wgpu` (SPIR-V) | Works, offline test | `--target-env vulkan1.3` accepts `OpLine`, `NonSemantic.Shader.DebugInfo.100`, and the embedded source. | Before this assessment, no Vulkan device ran (§2). | PLAN §8 step 4 |
| Vulkan device (NVIDIA ICD) | 580.178.04, API 1.4.312 | `wgpu` (SPIR-V, WGSL) | Works from this assessment, with a local driver (§2) | The `cubecl-wgpu` suite with `spirv` passes (803) with `NonSemantic` debug data. The bench runs on SPIR-V and on WGSL, with the same checksum as CUDA. | The host image has no Vulkan driver. | §2 |
| Nsight Systems (`nsys`) | – | `cuda` | Not checked | – | – | – |
| rocprof / rocprofiler, HIP on a device | – | `hip` | Not checked | – | This host has no AMD GPU (`lspci` shows only the RTX 4070). The HIP code paths have offline tests only. | PLAN §6, §8 step 2 |
| RenderDoc | – | `wgpu` (SPIR-V) | Not checked | – | The Vulkan driver of §2 makes a check possible now. | – |
| Tracy (`profile-tracy`) | – | all | Not checked | – | The feature exists on `main`. This work does not change it. | PLAN §2.4 |
| `tracing` layers (Perfetto, `tracing-flame`, OpenTelemetry) | – | all | Not checked | – | The launch span exists on `main`. | PLAN §2.4 |
| Xcode GPU capture (Metal) | – | `metal` | Not checked | – | Needs macOS. | PLAN §8 step 3 |
| Pyroscope | – | `cpu` | Not checked | – | – | PLAN §9 |

## 2. Vulkan on this host

**Symptom:** all Vulkan tests failed. `vulkaninfo` writes "vkCreateInstance: Found no drivers!".

**Cause 1:** the host image has the CUDA part of the NVIDIA driver only (`nvidia-driver-cuda-libs`). It does not have `nvidia-driver-libs`, which contains the Vulkan driver (`libGLX_nvidia.so.0`) and its manifest. Mesa has no Vulkan driver installed either (`mesa-vulkan-drivers` is not installed). Thus `/usr/share/vulkan/icd.d` and `/etc/vulkan/icd.d` are empty. The problem is in the host, not in cubecl.

**Cause 2:** with only the Vulkan manifest, the NVIDIA driver returns no `vkCreateInstance` ("Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr'"). `strace` shows that the driver loads glvnd EGL, and glvnd finds only `50_mesa.json`. The NVIDIA EGL vendor file `10_nvidia.json` is also in `nvidia-driver-libs`. When glvnd gets that file, `vkCreateInstance` succeeds.

**Local fix (no change to the system):**

1. Download `nvidia-driver-libs-580.178.04-5.fc44.x86_64.rpm` from the negativo17 repository `nvidia-580/fedora-44` (the installed driver has the same version and packager). The SHA-256 is the same as in the repository metadata, and `rpmkeys --checksig` accepts the negativo17 key `0C5D 0F47 0484 AE2F C40A 9B65 97F3 0089 93E8 909B`.
2. Unpack it with `rpm2cpio` to `~/.cache/nvtools/vkroot`.
3. Write a Vulkan manifest and an EGL vendor file with the absolute library paths (`~/Workspace/bench-target/scripts/nvidia_icd.json`, `10_nvidia.json`).
4. Set the variables for each run:

   ```sh
   export VK_DRIVER_FILES=~/Workspace/bench-target/scripts/nvidia_icd.json
   export __EGL_VENDOR_LIBRARY_FILENAMES=~/Workspace/bench-target/scripts/10_nvidia.json
   export LD_LIBRARY_PATH=~/.cache/nvtools/vkroot/usr/lib64
   ```

**Result:** `vulkaninfo` shows the RTX 4070, driver 580.178.04, Vulkan 1.4.312.

**Permanent fix (for the user):** add `nvidia-driver-libs` (same version as the kernel module) to the host image. Then remove the variables.

**Test suite:** `cargo test -p cubecl-wgpu --features spirv` on the branch (`49c13b23f`), `dev` profile, with the variables above: 803 passed, 0 failed, 17 ignored. The integration tests (`device`, `device_poisoned`, `graph`, `relocation`) pass. The `dev` profile gives each kernel `LineTables`, and the device has Vulkan 1.3, so `Auto` selects `NonSemantic.Shader.DebugInfo.100` (PLAN §8 step 4). Thus this run is also the first device check of the SPIR-V debug data: the NVIDIA driver accepts it for all test kernels. PLAN §8 says "no device check is claimed". That statement can change now.

## 3. Benchmark: cost of the debug levels

### 3.1 Method

- **Program:** `debug_bench` (`~/Workspace/bench-target/scripts/debug_bench`). It is a member of a detached worktree of each commit, so the branch under test does not change. The kernel calls `step → doubled → square_third` in a loop with a run-time count, and then `variant` more times in a `#[unroll]` loop. Each value of the comptime `variant` is a new kernel.
- **Compile time:** the first launch and `sync` of 32 variants (one cube, one iteration). This is the JIT compile, the backend compile and the launch. The table gives the median for each kernel without the first kernel. The first kernel also has the setup of each process.
- **Run time:** the full grid of variant 8, 5 warm-up launches, then 50 timed launches, each with `sync`. GPU: 4096 × 256 threads, 2000 iterations. CPU: 256 × 256 threads, 2000 iterations (a second pass, 7 repetitions). The first CPU pass (64 cubes, 200 iterations, about 1 ms) had two states, 1 ms and 2 ms, from the thread pool. Thus it is not used for run time.
- **Repetitions:** each configuration and runtime is a new process. The script runs all configurations in turn, 15 times. The tables give the median over the 15 processes. The first repetition was slower on all configurations (cold caches), so the median is necessary.
- **Check of the level:** `CUBECL_DEBUG_PLIRON` dumps of each branch build (§3.2). The JIT checksum is the same on all GPU runtimes.

### 3.2 Configurations

| Config | Commit | Build | Run-time variables | Kernel debug data (checked in the dumps) |
|---|---|---|---|---|
| main | `main` | `debug = 0` | – | None (no location capture on `main`) |
| none | branch | `debug = 0` | – | None: 0 `!DILocation`, 0 `.loc` |
| lines | branch | `debug = "line-tables-only"` | – | `LineTables`: 17–18 `!DILocation`, 8 `inlinedAt`, 80 `.loc` in the PTX |
| lines+env-none | branch | as lines | `CUBECL_DEBUG_INFO=none` | None |
| debug2 | branch | `debug = 2` | – | `LineTables`, the same as lines (§3.1 of PLAN) |
| full | branch | `debug = "line-tables-only"`, `CUBECL_DEBUG=1` at build time | – | `Full`: as lines, plus the source text (CPU). The PTX has no text. |
| lines+jitdump-feature | branch | as lines, feature `jitdump` | – | as lines. The feature is compiled, but off. |
| lines+perfmap | branch | as lines, feature `jitdump` | `CUBECL_JIT_SYMBOLS=perfmap` | as lines, plus `/tmp/perf-<pid>.map` |
| lines+perf | branch | as lines, feature `jitdump` | `CUBECL_JIT_SYMBOLS=perf` | as lines, plus the perf map and the jitdump (JITLink) |
| lines+fp | branch | as lines, `-C force-frame-pointers=yes` | – | as lines, plus frame pointers in the kernel |
| lines+fp+perf | branch | as lines+fp, feature `jitdump` | `CUBECL_JIT_SYMBOLS=perf` | the full recipe of the book for `perf` |

The perf files exist only on the `cpu` runtime, so the last five configurations run only there. The CUDA runtime has no perf files. The C++ (NVRTC) and WGSL backends are separate builds (`cuda-cpp`, `wgpu` without `spirv`) with the configurations main, none, lines, debug2 and full.

### 3.3 Results

Compile = median compile time for each kernel. Run = median run time of the full grid. "vs main" is the change from `main`.

**cpu (LLVM JIT).** Compile: 15 repetitions. Run: the second pass, 7 repetitions.

| Config | Compile (ms) | vs main | Run (ms) | vs main |
|---|---:|---:|---:|---:|
| main | 13.49 | – | 37.44 | – |
| none | 13.08 | −3.0% | 37.31 | −0.3% |
| lines | 14.78 | +9.6% | 37.29 | −0.4% |
| lines+env-none | 13.12 | −2.7% | 37.28 | −0.4% |
| debug2 | 14.71 | +9.1% | 37.27 | −0.4% |
| full | 14.83 | +9.9% | 37.23 | −0.5% |
| lines+jitdump-feature | 14.58 | +8.1% | 37.30 | −0.4% |
| lines+perfmap | 14.45 | +7.2% | 37.31 | −0.4% |
| lines+perf | 15.09 | +11.9% | 37.33 | −0.3% |
| lines+fp | 14.66 | +8.7% | 37.41 | −0.1% |
| lines+fp+perf | 14.98 | +11.1% | 37.16 | −0.7% |

The first kernel of a process takes 29.7 ms on `main` and 30.6 ms at lines. With `CUBECL_JIT_SYMBOLS=perf`, it takes 49.2 ms: the jitdump setup costs approximately 19 ms one time for each process.

**cuda (LLVM, PTX).** 15 repetitions.

| Config | Compile (ms) | vs main | Run (ms) | vs main |
|---|---:|---:|---:|---:|
| main | 10.57 | – | 3.203 | – |
| none | 10.63 | +0.5% | 3.000 | −6.4% |
| lines | 12.08 | +14.2% | 3.000 | −6.4% |
| lines+env-none | 10.58 | +0.1% | 3.000 | −6.3% |
| debug2 | 11.83 | +11.9% | 3.204 | +0.0% |
| full | 11.81 | +11.7% | 3.000 | −6.3% |

The run time has two states, 3.00 ms and 3.20 ms. Each configuration shows both states in different processes, also none, which has no debug data. `main` and debug2 show only 3.20 ms. lines and debug2 have the same debug level and the same number of `.loc` directives, but lines shows mostly 3.00 ms. Thus the difference does not come from the debug level. The cause is not known.

**cuda (C++, NVRTC).** 15 repetitions.

| Config | Compile (ms) | vs main | Run (ms) | vs main |
|---|---:|---:|---:|---:|
| main | 17.00 | – | 3.410 | – |
| none | 17.09 | +0.5% | 3.409 | −0.1% |
| lines | 18.05 | +6.2% | 3.408 | −0.1% |
| debug2 | 18.02 | +6.0% | 3.413 | +0.1% |
| full | 18.17 | +6.8% | 3.407 | −0.1% |

**vulkan (SPIR-V).** 15 repetitions.

| Config | Compile (ms) | vs main | Run (ms) | vs main |
|---|---:|---:|---:|---:|
| main | 12.00 | – | 1.436 | – |
| none | 12.16 | +1.3% | 1.440 | +0.3% |
| lines | 13.86 | +15.5% | 1.446 | +0.7% |
| lines+env-none | 12.13 | +1.1% | 1.444 | +0.5% |
| debug2 | 14.09 | +17.5% | 1.446 | +0.7% |
| full | 13.89 | +15.8% | 1.442 | +0.4% |

The Vulkan run time varies by 7–14% between processes, in all configurations.

**wgpu (WGSL on Vulkan).** 15 repetitions.

| Config | Compile (ms) | vs main | Run (ms) | vs main |
|---|---:|---:|---:|---:|
| main | 3.49 | – | 2.490 | – |
| none | 3.47 | −0.6% | 2.493 | +0.1% |
| lines | 3.97 | +13.6% | 2.497 | +0.3% |
| debug2 | 3.97 | +13.7% | 2.491 | +0.1% |
| full | 3.97 | +13.8% | 2.493 | +0.1% |

### 3.4 Findings

1. **Run time does not change.** On all five backends, no debug level and no perf setting changes the kernel run time by more than the noise between processes (CPU ±0.7%, NVRTC ±0.1%, WGSL ±0.3%, Vulkan ±0.7% with 7–14% noise). This agrees with PLAN §3.1: line data does not change the machine code. Frame pointers also gave no measurable change on this kernel, which has no calls after inlining.
2. **The branch at `debug = 0` costs nothing.** none is within −3.0% to +1.3% of `main` for compile time and run time on all backends. `CUBECL_DEBUG_INFO=none` on a line-table build gives the same result as `debug = 0`.
3. **Debug data costs compile time only.** Compared with none on the same branch, `LineTables` adds approximately 13–14% for each kernel on CUDA (LLVM), Vulkan (SPIR-V), WGSL and the CPU JIT, and approximately 6% on NVRTC. `Full` costs the same as `LineTables`. Cargo `debug = 2` costs the same as `"line-tables-only"`, because cubecl maps both to `LineTables`.
4. **WGSL pays for data that it does not use.** naga writes no source lines in WGSL (PLAN §8 step 5), but WGSL compile time increases by 14%. Thus the cost is in the frontend and in the IR passes (location capture, `CallSite` chains, `inherit_locations`), not in the backend. A backend that cannot use locations could ask for `None`. This is a possible improvement, not a defect.
5. **perf files.** The perf map has no measurable cost. The jitdump (`CUBECL_JIT_SYMBOLS=perf`) adds approximately 2–3% for each kernel and approximately 19 ms one time for each process (the JITLink plugins and the file). The cost occurs only when the variable is set, as PLAN §3.2 requires.
6. **The defaults are acceptable.** The `dev` profile gives `LineTables` by default (PLAN §3.1). In absolute time, this is approximately 0.5–1.7 ms for each kernel compile in this bench, one time for each kernel in a process.

### 3.5 Not measured

- HIP and Metal: no device.
- Compile time with the `persistence` cache: the bench does not turn it on.
- Larger kernels: see §4 (matmul from `cubek`).
- IR dumps (`CUBECL_DEBUG_PLIRON`) and pass timing (`CUBECL_TIME_PASSES`) have a run-time switch and write files or logs. They are tools for compiler work, so they are not in the bench.

### 3.6 How to repeat

The scripts in `~/Workspace/bench-target/scripts` use the scratchpad path of this session in the variable `S`. Change `S` before you run them.

1. `git worktree add --detach ../cubecl-main main`, `git worktree add --detach ../cubecl-bench <commit>`, and copy `debug_bench` to `examples/` in each worktree. On `main`, remove the `jitdump` feature line from its `Cargo.toml`.
2. `build_all.sh`: builds the 13 binaries (approximately 25 minutes).
3. `verify_levels.sh`: shows the debug data of each build from the dumps.
4. `REPS=15 BENCH_SAMPLES=50 run_all.sh`, then `python3 analyze.py results.jsonl`.

## 4. Benchmark: matmul from `cubek`

### 4.1 Method

- **Program:** the `gemm` and `gemm_cpu` benches of `cubek` (`benchmarks/benches`), built with `cargo bench --no-run`. A detached `cubek` worktree for each cubecl tree (`../cubek-main`, `../cubek-bench`) keeps each `Cargo.lock` stable. The patch files are `~/Workspace/bench-target/scripts/cubek-patch-{main,bench}.toml`. The branch file also has the `pliron` patch of the branch.
- **Builds (14):** for each of main, none (`debug = 0`) and lines (`debug = "line-tables-only"`), the runtimes `cuda` (LLVM), `cuda` + `cuda-cpp` (NVRTC), `vulkan` (SPIR-V) and `cpu`. The `cpu` runtime also has lines with the feature `jitdump`, and lines with `jitdump` and frame pointers. `full` is not in this bench, because §3 shows that it costs the same as lines.
- **Rows:** GPU: the strategies `simple_cyclic_cmma` (2 variants), `double_tilewise_cmma` (2 variants) and `simple_unit_max`, on the problems f16 1536³, f16 2×4096³, f32 2×1024³ and f32 4096×(64×1024×64). CPU: `simple_unit_max` and `cpu_gemm_fast_p16` on f32 512³ and f32 2×1024³.
- **Timing:** the `System` timing of the `cubek` harness (launch and sync), 30 samples after the warm-up of the harness. The tables give the median over 5 processes of the median of each process. The process wall time includes the device peak probes, the compile of each kernel, and all samples. The samples are the same for each configuration, so a difference in wall time is mostly compile time.
- **Failures that are not regressions:** the CMMA strategies have no f32 tile on CUDA (LLVM) and Vulkan ("No tile size is available for the problem"). The 8 rows fail on `main` and on the branch in the same way, so the tables leave them out.

### 4.2 Results

Times in milliseconds. "none" and "lines" give the change from `main`.

**cuda (LLVM, PTX)**

| Strategy / problem | main | none | lines |
|---|---:|---:|---:|
| DoubleTilewiseCmma (specialized) / f16 1536³ | 0.214 | +1.3% | +0.1% |
| DoubleTilewiseCmma (specialized) / f16 2×4096³ | 5.648 | +0.1% | +7.4% (§4.3) |
| DoubleTilewiseCmma / f16 1536³ | 0.203 | −2.1% | −1.0% |
| DoubleTilewiseCmma / f16 2×4096³ | 5.148 | −0.2% | +4.5% (§4.3) |
| SimpleCyclicCmma / f16 1536³ | 0.215 | −0.9% | −1.1% |
| SimpleCyclicCmma / f16 2×4096³ | 5.526 | −0.1% | −0.4% |
| SimpleCyclicCmma (multi rows) / f16 1536³ | 0.240 | +0.9% | 0.0% |
| SimpleCyclicCmma (multi rows) / f16 2×4096³ | 6.357 | 0.0% | 0.0% |
| Simple Unit / f16 1536³ | 4.065 | −0.3% | −0.1% |
| Simple Unit / f16 2×4096³ | 143.923 | +0.1% | +0.1% |
| Simple Unit / f32 2×1024³ | 0.496 | −0.6% | +2.4% |
| Simple Unit / f32 4096×(64×1024×64) | 8.189 | +1.8% | −0.3% |
| **Process wall time** | 35.44 s | −0.1% | +5.9% |

**cuda (C++, NVRTC)**, with the complete headers of §5.2

| Strategy / problem | main | none | lines |
|---|---:|---:|---:|
| DoubleTilewiseCmma (specialized) / f16 1536³ | 0.214 | −1.2% | −0.9% |
| DoubleTilewiseCmma (specialized) / f16 2×4096³ | 5.650 | +0.1% | −0.2% |
| DoubleTilewiseCmma (specialized) / f32 2×1024³ | 0.393 | −0.3% | +3.6% |
| DoubleTilewiseCmma (specialized) / f32 4096×(64×1024×64) | 5.484 | +0.1% | 0.0% |
| DoubleTilewiseCmma / f16 1536³ | 0.199 | −1.9% | −0.7% |
| DoubleTilewiseCmma / f16 2×4096³ | 5.128 | +0.2% | −0.4% |
| DoubleTilewiseCmma / f32 2×1024³ | 0.490 | −0.2% | +2.7% |
| DoubleTilewiseCmma / f32 4096×(64×1024×64) | 6.374 | −2.2% | −1.3% |
| SimpleCyclicCmma / f16 1536³ | 0.228 | −0.1% | +0.2% |
| SimpleCyclicCmma / f16 2×4096³ | 5.731 | −0.1% | −0.1% |
| SimpleCyclicCmma / f32 2×1024³ | 0.280 | +0.5% | +0.3% |
| SimpleCyclicCmma / f32 4096×(64×1024×64) | 5.124 | 0.0% | 0.0% |
| SimpleCyclicCmma (multi rows) / f16 1536³ | 0.189 | +0.2% | +0.5% |
| SimpleCyclicCmma (multi rows) / f16 2×4096³ | 5.024 | −0.1% | −0.1% |
| SimpleCyclicCmma (multi rows) / f32 2×1024³ | 0.246 | −0.4% | −0.7% |
| SimpleCyclicCmma (multi rows) / f32 4096×(64×1024×64) | 4.888 | +0.1% | +0.1% |
| Simple Unit / f16 1536³ | 3.977 | +0.1% | 0.0% |
| Simple Unit / f16 2×4096³ | 142.878 | +0.1% | 0.0% |
| Simple Unit / f32 2×1024³ | 0.481 | +0.3% | +0.1% |
| Simple Unit / f32 4096×(64×1024×64) | 8.024 | +0.2% | +0.1% |
| **Process wall time** | 38.03 s | −0.1% | +1.0% |

**vulkan (SPIR-V)**

| Strategy / problem | main | none | lines |
|---|---:|---:|---:|
| DoubleTilewiseCmma (specialized) / f16 1536³ | 0.271 | +2.1% | −1.8% |
| DoubleTilewiseCmma (specialized) / f16 2×4096³ | 7.873 | −0.2% | 0.0% |
| DoubleTilewiseCmma / f16 1536³ | 0.274 | −0.9% | −2.4% |
| DoubleTilewiseCmma / f16 2×4096³ | 5.862 | −0.2% | −0.2% |
| SimpleCyclicCmma / f16 1536³ | 0.345 | +0.5% | −1.8% |
| SimpleCyclicCmma / f16 2×4096³ | 9.137 | −0.1% | 0.0% |
| SimpleCyclicCmma (multi rows) / f16 1536³ | 0.285 | +0.1% | +1.0% |
| SimpleCyclicCmma (multi rows) / f16 2×4096³ | 6.354 | −0.3% | +0.1% |
| Simple Unit / f16 1536³ | 0.828 | +0.8% | +1.3% |
| Simple Unit / f16 2×4096³ | 22.786 | −0.2% | +0.3% |
| Simple Unit / f32 2×1024³ | 0.712 | +0.3% | +2.4% |
| Simple Unit / f32 4096×(64×1024×64) | 12.349 | −0.1% | 0.0% |
| **Process wall time** | 32.30 s | +0.6% | +2.7% |

**cpu (LLVM JIT)**

| Strategy / problem | main | none | lines | lines+jitdump-feature | lines+perf | lines+fp | lines+fp+perf |
|---|---:|---:|---:|---:|---:|---:|---:|
| CpuGemm (fast leaf, 16 threads) / f32 512³ | 0.586 | −3.3% | −5.8% | −3.4% | −2.3% | +9.8% | −2.0% |
| CpuGemm (fast leaf, 16 threads) / f32 2×1024³ | 13.540 | −3.8% | −4.4% | +0.3% | −3.9% | −6.4% | −6.1% |
| Simple Unit / f32 512³ | 1.895 | +0.4% | −0.6% | +0.3% | +0.2% | +0.6% | −0.1% |
| Simple Unit / f32 2×1024³ | 29.698 | −1.1% | −1.0% | −1.3% | −0.4% | −0.8% | −1.0% |
| **Process wall time** | 3.58 s | −0.5% | +6.5% | +7.1% | +7.7% | +6.1% | +6.7% |

The CpuGemm row on 512³ takes 0.6 ms on 16 threads, and its medians vary by up to 10% between processes in all configurations.

### 4.3 The two DoubleTilewiseCmma rows on CUDA (LLVM)

At lines, the f16 2×4096³ rows were 4–7% slower in 4 of 5 processes. This was examined:

1. **The PTX instructions are the same.** The dumps of the specialized kernel at none and at lines have no difference after the removal of `.loc`, `.file`, the labels and the debug sections.
2. **The machine code is bit-identical.** The CUDA driver JIT (`cuLinkComplete`), with and without `CU_JIT_GENERATE_LINE_INFO`, gives the same 4,736 encoded SASS words (`nvdisasm -hex`) and the same resource use (255 registers, 16 bytes of stack) for both PTX files. Only the cubin grows, from 46 KB to 196 KB, because of the line table.
3. **At fixed clocks, the kernel time is the same.** Nsight Compute (base clock): 7.79 ms for none, lines, and lines with `CUBECL_DEBUG_INFO=none`, with the same instruction count, L2 hit rate and DRAM traffic.
4. **The two times are two SM clocks.** Nsight Compute with `--clock-control none` shows two states for none and for lines: 5.60 ms at 2.66 GHz and 6.29 ms at 2.37 GHz, with the same number of cycles (14.9 million) in both states.
5. In 12 alternate runs without a profiler, none was in the slow state 1 time and lines 7 times. The same binary with `CUBECL_DEBUG_INFO=none` was always in the fast state in 5 runs.

**Conclusion:** the debug data does not change the kernel. The GPU selects a lower clock in some processes. Line data makes the slow state more frequent in this bench. A possible cause is the longer JIT compile before the samples, which changes the idle time and the boost state of the GPU, but this was not proved. The `cubek` harness notes that clocks that ramp up distort short benches.

### 4.4 Findings

1. **Run time:** on all four runtimes, line tables do not change the matmul kernels. The only differences above noise are the two rows of §4.3, which are a GPU clock state, not code.
2. **Compile time:** the process wall time increases by 5.9% on CUDA (LLVM), 1.0% on NVRTC, 2.7% on Vulkan and 6.5% on the CPU. For a matmul, the run of the kernels is a larger part of the process than in §3, so the relative compile cost is smaller.
3. **perf files on the CPU:** the jitdump and frame pointers add approximately 1% of wall time to lines. They do not change the kernel time.
4. **none is equal to main** within ±2% in all rows that run, except the 512³ CpuGemm row, which is noisy in all configurations.

## 5. Defects and problems found by the benches

### 5.1 The branch broke `cubek` (fixed in `8192dd442`)

`cubek-tile` (at `a8207f82`) did not compile against the branch: 39 errors such as "the trait bound `InitFrom: CubeDebug` is not satisfied". The macro passes each `let` value and each parameter through the naming function, which needed `CubeDebug`. On `main`, the macro emits that call only with `CUBECL_DEBUG=1`. The branch emits it in all builds, because the level is selected at run time. A value of a plain Rust type, for example a comptime enum that a `#[cube]` function returns, has no `CubeDebug`.

**Fix** (`8192dd442`): the macro names a value only when its type implements `CubeDebug`, with autoref specialization (`DebugVar`, `DebugVarNamed`, `DebugVarUnnamed`). The test `a_plain_variable_needs_no_cube_debug` does not compile without the fix. The `cubecl-cpu` suite (812) passes, and `cubek` compiles. PLAN §5 step 3 records the change (`f451a12fa`).

### 5.2 The NVRTC headers on this host are incomplete

The CMMA kernels include `mma.h`, which includes `crt/mma.h`. The pip wheel `nvidia-cuda-runtime-cu12` does not have the `crt/` directory, so NVRTC rejects each CMMA kernel. A merged include tree (`~/Workspace/bench-target/cuda-include`: the runtime headers, and `crt/` from `nvidia-cuda-nvcc-cu12` 12.9.86, the same version as NVRTC 12.9.86) corrects this. Set `CUDA_PATH` to it. The problem is in this host, not in cubecl.

### 5.3 A failed compile can look like a slow kernel in a `System` bench

With the headers of §5.2 missing, the NVRTC CMMA rows of the first run gave times (30 ms, and 55 ms at lines) for kernels that never compiled. The `System` timing of `cubecl_common::benchmark` (`profile_full`) measures `execute` and `sync`. By the error model of cubecl (ERROR_HANDLING.md, "No device-wide fault"), a failed compile flags the buffers that the launch writes, and a later read of them fails. `sync` does not fail. A bench that does not read its output thus measures the failed launch. With `CUBEK_BENCH_TIMING=device`, the same rows report the compile error. This is not changed here, because it follows from the error model. A `System` bench can read one element of its output after the samples to find the failure. These rows are in `~/Workspace/bench-target/cubek/results-nvrtc-broken`, and §4.2 does not use them.

### 5.4 How to repeat §4

1. `git -C ../cubek worktree add --detach ../cubek-main <rev>`, and the same for `../cubek-bench`.
2. `build_cubek.sh` (approximately 45 minutes; `SKIP_EXISTING=1` keeps the binaries that exist).
3. `REPS=5 run_cubek.sh`, then `python3 analyze_cubek.py`. `ONLY='-cpp$'` runs only the matching binaries.
