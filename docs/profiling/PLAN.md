# Plan: Kernel Profiling and Flamegraph Support

| Item | Value |
|---|---|
| Base commit | [`a1bb768ce919260eea56dbd0b59c70e55236e22d`](https://github.com/skewballfox/cubecl/commit/a1bb768ce919260eea56dbd0b59c70e55236e22d) (`main`, "fix(llvm): the GPU targets read PLANE_POS (#1714)") |
| pliron | `0.18.0`, commit [`3517dc6`](https://github.com/pliron-org/pliron/tree/3517dc6c08370e486297972be8da2b98ed0cd683) |
| pliron-spirv | `0.15.0+sdk-1.4.357.0`, commit [`daa0d7b`](https://github.com/tracel-ai/tracel-rspirv/tree/daa0d7b84e2cfe0786e00e42160303ce7587e726/pliron-spirv) |
| Open decisions | [DECISIONS.md](DECISIONS.md) |
| Status | Phase 0 done ([`86d4822`](https://github.com/skewballfox/cubecl/commit/86d482246b293ea183653e4414290a94e47d0f26)). Phases 1–5 not started. |
| Language | ASD-STE100 / Attempto Controlled English. Domain terms are permitted. |

All links point to the commits above. Before you implement a step on a later commit, do the check in its **Verify** line again.

---

## 1. Goal

1. The user sets `debug` in a cargo profile, as for any Rust program. Then the CPU profilers (`perf`, `samply`) and the GPU profilers (Nsight Compute, `rocprof`, RenderDoc) show kernel samples against the Rust source lines of `#[cube]` functions.
2. The debug data shows each inlined `#[cube]` function as a separate frame. A flamegraph shows the call chain inside a kernel.
3. The data uses open formats and interfaces that existing tools read: DWARF, perf jitdump, the GDB JIT interface, CUDA/HIP line info (CUPTI, rocprofiler), SPIR-V debug lines, and `tracing` spans. cubecl adds no profiler and no output format of its own.
4. The changes add to the API. They do not change existing signatures.
5. The changes are in cubecl. One small hook in `pliron-spirv` is necessary for SPIR-V (§8). Upstream `pliron` does not need changes.

## 2. Findings on the base commit

### 2.1 `CUBECL_DEBUG_PLIRON`

The dump code exists in each backend. A **build-time** switch controls it, not a runtime switch.

- Each `build.rs` enables the `pliron-dump` cfg only if `CUBECL_DEBUG_PLIRON` is set **when cargo compiles the crate**: [`cubecl-llvm/build.rs#L4-L7`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/build.rs#L4-L7), [`cubecl-cpp/build.rs#L4-L9`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cpp/build.rs#L4-L9), [`cubecl-spirv/build.rs#L4-L9`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-spirv/build.rs#L4-L9), [`cubecl-wgpu/build.rs#L30-L35`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-wgpu/build.rs#L30-L35).
- **Result:** if a binary was compiled without the variable, setting it at run time does nothing, and no message tells the user.
- Five copies of the "dump directory" helper exist: [`cubecl-llvm/src/shared/base.rs#L489-L499`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/shared/base.rs#L489-L499), [`cubecl-llvm/src/cpu/jit/engine.rs#L132-L138`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/cpu/jit/engine.rs#L132-L138), [`cubecl-cpp/src/shared/base.rs#L413-L416`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cpp/src/shared/base.rs#L413-L416), [`cubecl-spirv/src/compiler.rs#L414-L416`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-spirv/src/compiler.rs#L414-L416), [`cubecl-wgpu/src/compiler/wgsl/compiler.rs#L198-L200`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-wgpu/src/compiler/wgsl/compiler.rs#L198-L200).
- **Defect:** the LLVM pipeline sets `print_after_all: true` in all builds: [`cubecl-llvm/src/shared/base.rs#L405-L415`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/shared/base.rs#L405-L415). pliron then writes the full IR after each pass to `log::info!` ([`pliron/src/pass.rs#L527-L533`](https://github.com/pliron-org/pliron/blob/3517dc6c08370e486297972be8da2b98ed0cd683/src/pass.rs#L527-L533)). With `RUST_LOG=info`, this makes a very large log.
- **Defect:** the AMDGPU path reads the variable at run time without the cfg: [`cubecl-llvm/src/amdgpu/codegen.rs#L118-L125`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/amdgpu/codegen.rs#L118-L125). If the feature is off, it generates the assembly and then discards it.
- The runtime configuration ([`cubecl-runtime/src/config/compilation.rs#L4-L20`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/config/compilation.rs#L4-L20)) has no field for the dump directory.

### 2.2 Source locations in the frontend

- The macro emits location calls only if the cfg `debug_symbols` is set. `CUBECL_DEBUG=1` or the `debug-symbols` feature sets this cfg **when `cubecl-macros` is compiled**: [`cubecl-macros/build.rs#L5-L16`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/build.rs#L5-L16). The per-function attributes `#[cube(debug_symbols)]` and `#[cube(no_debug_symbols)]` override it.
- The macro emits:
  - `debug_source_expand(scope, name, file!(), source_text, line!(), column!())` at function entry: [`cubecl-macros/src/generate/kernel.rs#L47-L65`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/kernel.rs#L47-L65).
  - `debug_call_expand(scope, line!(), column!(), |scope| call)` around calls: [`cubecl-macros/src/generate/expression.rs#L979-L988`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/expression.rs#L979-L988).
  - Nothing for expressions. The span call is commented out: [`cubecl-macros/src/generate/expression.rs#L968-L977`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/expression.rs#L968-L977).
- The receivers are empty. Their bodies are commented-out code from the IR that came before pliron: [`cubecl-core/src/frontend/debug.rs#L8-L44`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-core/src/frontend/debug.rs#L8-L44).
- **Result:** each pliron `Operation` has `Location::Unknown`.
- The cfg belongs to the build of `cubecl-macros`, not to the profile of the crate that uses the macro. Cargo builds proc macros with the `build-override` profile. Thus `debug = …` in `[profile.release]` has no effect on the cfg.
- `KernelBuilder::new` sets `debug_symbols` at run time from the compilation log level ([`builder.rs#L128-L141`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-core/src/compute/builder.rs#L128-L141)). Without the capture calls, this flag gives no locations.

### 2.5 Toolchain facts (checked with cargo 1.94.1)

- Cargo gives rustc `-C debuginfo=<level>` for the profile of each crate: `line-tables-only`, `1`, `2`, and so on. It gives no flag for `debug = 0`.
- A proc macro runs in the rustc process. `std::env::args()` in the macro shows the flags of the crate that **calls** the macro. `std::env::current_dir()` shows the rustc working directory. When `debug` changes, cargo compiles the crate again, and the macro runs again.
- A build script gets only `DEBUG=true|false`. It gets no level. Cargo compiles the build script of a normal library (for example `cubecl-runtime`) with the target profile. For a proc macro (`cubecl-macros`), cargo uses the `build-override` profile.
- No cubecl build script reads `DEBUG` on the base commit. It also gets `CARGO_ENCODED_RUSTFLAGS`, which contains `-C force-frame-pointers` if the user sets it.

### 2.3 Source locations in the IR and the backends

- pliron keeps one `Location` for each `Operation` ([`pliron/src/operation.rs#L825-L833`](https://github.com/pliron-org/pliron/blob/3517dc6c08370e486297972be8da2b98ed0cd683/src/operation.rs#L825-L833)). `Location` has `SrcPos`, `Named`, `CallSite`, `Fused` and `Unknown` ([`pliron/src/location.rs#L137-L167`](https://github.com/pliron-org/pliron/blob/3517dc6c08370e486297972be8da2b98ed0cd683/src/location.rs#L137-L167)). These are the same as the MLIR location attributes.
- The pliron rewriters do not copy a location to replacement ops. The two rewriters that cubecl uses both record each inserted op in `Recorder::events` ([`pliron/src/irbuild/listener.rs#L83-L107`](https://github.com/pliron-org/pliron/blob/3517dc6c08370e486297972be8da2b98ed0cd683/src/irbuild/listener.rs#L83-L107)). All cubecl passes go through two wrappers: [`cubecl-ir/src/rewrite.rs#L40-L77`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-ir/src/rewrite.rs#L40-L77).
- `pliron-llvm` does not export `Location` to LLVM debug info. `convert_block` ignores it ([`pliron-llvm/src/to_llvm_ir.rs#L2052-L2083`](https://github.com/pliron-org/pliron/blob/3517dc6c08370e486297972be8da2b98ed0cd683/pliron-llvm/src/to_llvm_ir.rs#L2052-L2083)), and `DI*` metadata nodes are not modelled ([`pliron-llvm/src/metadata.rs#L36-L38`](https://github.com/pliron-org/pliron/blob/3517dc6c08370e486297972be8da2b98ed0cd683/pliron-llvm/src/metadata.rs#L36-L38)). But `pliron-llvm` **does** copy generic metadata attachments to LLVM instructions ([`pliron-llvm/src/metadata_conversions.rs#L598-L630`](https://github.com/pliron-org/pliron/blob/3517dc6c08370e486297972be8da2b98ed0cd683/pliron-llvm/src/metadata_conversions.rs#L598-L630)). cubecl already uses these attachments for loop hints ([`cubecl-llvm/src/shared/loop_hints.rs#L24-L42`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/shared/loop_hints.rs#L24-L42)).
- The CPU, NVPTX and AMDGPU paths all parse the LLVM text again through one cubecl wrapper, `LlvmModule::new` ([`cubecl-llvm/src/shared/llvm_module.rs#L54`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/shared/llvm_module.rs#L54)). Callers: [`nvptx/codegen.rs#L95`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/nvptx/codegen.rs#L95), [`amdgpu/codegen.rs#L86`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/amdgpu/codegen.rs#L86), [`cpu/jit/engine.rs#L157-L161`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/cpu/jit/engine.rs#L157-L161). This is the single point where cubecl can add DWARF with `llvm-sys`. `llvm-sys 231` has all the necessary `LLVMDIBuilder*`, `LLVMInstructionSetDebugLoc` and `LLVMSetSubprogram` functions.
- The CPU JIT uses `LLVMLLJIT::new_with_default_builder()` ([`cpu/jit/engine.rs#L80`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/cpu/jit/engine.rs#L80)). This builder does not register a JIT event listener. Thus `perf` cannot symbolize JIT code.
- CUDA NVRTC already gets `-lineinfo` ([`cubecl-cuda/src/compute/context.rs#L420`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cuda/src/compute/context.rs#L420)). But the generated C++ has no `#line` directives, so the line info refers to generated code. The PTX load uses `cuModuleLoadData` with no JIT options ([`context.rs#L480`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cuda/src/compute/context.rs#L480)). HIP gets `-O3` only ([`cubecl-hip/src/compute/context.rs#L569-L575`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-hip/src/compute/context.rs#L569-L575)).
- The SPIR-V emitter converts each op in `op_to_spirv` ([`pliron-spirv/src/lib.rs#L271-L278`](https://github.com/tracel-ai/tracel-rspirv/blob/daa0d7b84e2cfe0786e00e42160303ce7587e726/pliron-spirv/src/lib.rs#L271-L278)). It emits no `OpLine`.

### 2.4 Host-side profiling

- `profile-tracy` exists ([`cubecl-runtime/src/client.rs#L1561-L1580`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/client.rs#L1561-L1580)).
- Each launch already opens a `tracing` span with the fields `kernel.name` and `kernel.id` (feature `tracing`): [`cubecl-runtime/src/client.rs#L1065-L1078`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/client.rs#L1065-L1078). Standard `tracing` layers send this span to Perfetto, Tracy, `tracing-flame`, OpenTelemetry and other tools. No change is necessary.
- CUPTI (Nsight Systems, Nsight Compute) and rocprofiler record each GPU kernel with its entry-point name. The name is the Rust function name plus a suffix for its generics ([`generate/kernel.rs#L303-L306`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/kernel.rs#L303-L306)). No change is necessary.
- **Defect:** `KernelId` does not include `debug_symbols` in its hash: [`cubecl-runtime/src/id.rs#L89-L110`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/id.rs#L89-L110), [`cubecl-macros/src/generate/kernel.rs#L457-L470`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/kernel.rs#L457-L470). With the `persistence` cache, a binary without debug info can be used for a run that requests debug info.

## 3. User-facing API

### 3.0 Design rule

The behavior must be the same as for ordinary Rust code:

- The `debug` key of the cargo profile selects the debug data. cubecl reads it with the cargo build-script variable `DEBUG` (§5 step 0).
- cubecl uses only open formats and interfaces that existing tools read. It adds no profiler and no output format of its own.
- Debug data that has no cost at kernel run time needs no run-time setting.
- A run-time switch is necessary only for a feature that has a run-time cost or a side effect. Such a feature needs the compile-time setting **and** the run-time switch.

### 3.1 Tier 1: the cargo profile only

```toml
[profile.profiling]
inherits = "release"
debug = "line-tables-only"   # or "full"
```

| Cargo `debug` | `DEBUG` in build scripts | cubecl level | Result |
|---|---|---|---|
| `none`, `0`, `false` | `false` | `None` | Kernels have no locations. The kernel output is byte-identical to the base commit. |
| any other value | `true` | `LineTables` | Lines and inlined `#[cube]` frames. |

`DEBUG` gives no level, so `Full` (lines and embedded source) is available only with the existing controls: `#[cube(debug_symbols)]`, the `debug-symbols` feature, or `CUBECL_DEBUG=1`. `DEBUG` is the official cargo interface, and cubecl already uses build scripts for its debug cfg. The rejected alternative read `-C debuginfo=<level>` from the rustc arguments in the macro. It gave the level, but it is not an official interface.

Cargo profiles set `debug = true` for `dev` by default. Thus dev builds get line tables in kernels too, the same as for the host code. This adds JIT compile time only. `CUBECL_DEBUG_INFO=none` removes it (§3.3).

`-C force-frame-pointers=yes` in `RUSTFLAGS` adds frame pointers to CPU JIT code too.

Line data does not change the machine code. It adds only JIT compile time and cache size. Thus it needs no run-time switch.

### 3.2 Features with a run-time cost: run-time switch also necessary

| Feature | Run-time cost or side effect | Run-time switch |
|---|---|---|
| perf jitdump and perf map (CPU runtime) | It writes `jit-<pid>.dump` in `$JITDUMPDIR` or `~/.debug/jit`, and `/tmp/perf-<pid>.map`. The files stay after the process stops. | `CUBECL_JIT_SYMBOLS=perf`. Other triggers: [D10](DECISIONS.md#d10-trigger-for-perf-symbol-files). |
| IR dumps | File I/O for each kernel. | `CUBECL_DEBUG_PLIRON=<dir>` |
| Pass timing | Log output for each kernel. | `CUBECL_TIME_PASSES=1` |

The GDB JIT registration (gdb, lldb) has almost no cost and writes no files. It is automatic when the level is not `None`, as native debug data is.

### 3.3 Tier 2: granular control

- The per-function attributes stay: `#[cube(debug_symbols)]` forces `Full`. `#[cube(no_debug_symbols)]` forces `None`.
- `CUBECL_DEBUG=1` at build time stays. It forces `Full`, for compatibility.
- New: `KernelSettings::debug_info(DebugInfo)`. The existing `debug_symbols()` becomes `debug_info(Full)`.
- New and optional: `compilation.debug_info` in `cubecl.toml`, or `CUBECL_DEBUG_INFO`. It can only **lower** the compiled level, for example to measure JIT time without debug data. It cannot raise the level, because the locations do not exist in a build without them.

### 3.4 Protections against common traps (built in)

| Trap | Protection |
|---|---|
| NVRTC `-G` or HIP `-g` turns off optimization, so the profile does not show release code. | Use `-lineinfo`, `-gline-tables-only`, and `CU_JIT_GENERATE_LINE_INFO` only. Do not change the optimization level. |
| The host code has debug data, but the kernels do not. | The same cargo `debug` key controls both (§5 step 0). |
| A per-package profile override gives the kernel crate a different `debug` value. | cubecl uses the `debug` value of `cubecl-runtime`. This is the value of the profile for all crates without an override. The book must tell users to put an override on `cubecl-runtime` too. |
| A cached kernel without debug data is used for a build with debug data. | Add the level to the `KernelId` hash (§6 step 8). |
| All code is inlined, so each sample goes to the kernel entry. | Use a `CallSite` chain for each op (§5). |
| An LLVM verifier error: "inlinable function call in a function with debug info must have a !dbg location". | Give each instruction a location. Use the kernel entry location as the fallback (§6 step 4). |
| JIT code is not visible to `perf`. | Use the perf JIT listener (§7). |
| `cargo flamegraph` does not run `perf inject --jit`, so it cannot read a jitdump. | Also write a perf map, which `perf script` reads with no extra step (§7 step 3). |
| `perf --call-graph fp` gives broken stacks in JIT code. | Follow `-C force-frame-pointers` (§7 step 3). |
| `file!()` gives a path relative to the workspace. Tools cannot find the file. | Use the rustc working directory as the DWARF compile directory, as rustc does. Users remove build paths with `--remap-path-prefix` or cargo `trim-paths`, as for other Rust code. |

## 4. Phase 0: repair the dump switch — done

**Status:** done in [`86d4822`](https://github.com/skewballfox/cubecl/commit/86d482246b293ea183653e4414290a94e47d0f26). The implementation differs from the first draft in one point: the shared helper is a type in `cubecl-core` (`KernelDump`), not a method on `CompilationConfig`. `cubecl-core` enables pliron's `std` feature, so it can build the pass manager settings. `cubecl-runtime` cannot.

- [x] **Step 1. Configuration.** `CompilationConfig` has `dump_dir` (std only) and `time_passes`: [`compilation.rs#L20-L27`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl-runtime/src/config/compilation.rs#L20-L27). `CUBECL_DEBUG_PLIRON` and `CUBECL_TIME_PASSES` set them: [`base.rs#L173-L179`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl-runtime/src/config/base.rs#L173-L179). The book documents both: [`config.md#L108-L123`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/cubecl-book/src/advanced-usage/config.md#L108-L123), [`config.md#L213-L215`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/cubecl-book/src/advanced-usage/config.md#L213-L215).
- [x] **Step 2. One helper.** `KernelDump` ([`dump.rs#L1-L81`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl-core/src/codegen/dump.rs#L1-L81)) sanitizes the kernel name, creates the directory, writes files only when enabled, and logs a failed write in place of a panic. It replaces the five helpers from §2.1. Call sites:
  - C++: [`base.rs#L221-L223`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl-cpp/src/shared/base.rs#L221-L223), [`base.rs#L335-L338`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl-cpp/src/shared/base.rs#L335-L338)
  - SPIR-V: [`compiler.rs#L119-L132`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl-spirv/src/compiler.rs#L119-L132), [`compiler.rs#L150-L156`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl-spirv/src/compiler.rs#L150-L156), [`compiler.rs#L183-L194`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl-spirv/src/compiler.rs#L183-L194)
  - WGSL: [`compiler.rs#L103-L121`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl-wgpu/src/compiler/wgsl/compiler.rs#L103-L121)
  - LLVM: [`base.rs#L405`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl-llvm/src/shared/base.rs#L405), [`engine.rs#L69-L74`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl-llvm/src/cpu/jit/engine.rs#L69-L74), [`codegen.rs#L103-L105`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl-llvm/src/nvptx/codegen.rs#L103-L105), [`codegen.rs#L92-L98`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl-llvm/src/amdgpu/codegen.rs#L92-L98)
- [x] **Step 3. No build-time switch.** The `pliron-dump` cfg is removed from all four `build.rs` files ([commit diff](https://github.com/skewballfox/cubecl/commit/86d482246b293ea183653e4414290a94e47d0f26)). The `cubecl-cpp` and `cubecl-spirv` build scripts are deleted, because they had no other content. `sanitize-filename` moves from three backends to `cubecl-core`. The `pliron-dump` feature stays, with no effect: [`Cargo.toml#L75-L82`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl/Cargo.toml#L75-L82). READMEs: [`README.md#L18`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl-spirv/README.md#L18), [`README.md#L81-L86`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl-llvm/README.md#L81-L86).
- [x] **Step 4. No IR in the `info` log when dumps are off.** `print_after_all` follows the dump: [`dump.rs#L46-L54`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl-core/src/codegen/dump.rs#L46-L54). The LLVM pipeline uses it: [`base.rs#L405`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl-llvm/src/shared/base.rs#L405). Checked: one CPU test with `RUST_LOG=info` printed the IR 21 times before, and 0 times after.
- [x] **Step 5. AMDGPU assembly only for a dumped kernel:** [`codegen.rs#L119-L127`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl-llvm/src/amdgpu/codegen.rs#L119-L127).
- [x] **Step 6. Pass timing.** `time_passes` maps to `PMConfig::time_all_passes` for every compiler: [`dump.rs#L51`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl-core/src/codegen/dump.rs#L51).

**Tests:** [`dump.rs#L83-L116`](https://github.com/skewballfox/cubecl/blob/86d482246b293ea183653e4414290a94e47d0f26/crates/cubecl-core/src/codegen/dump.rs#L83-L116) (a disabled dump writes nothing; an enabled dump writes to the sanitized kernel directory). Manual check: `CUBECL_DEBUG_PLIRON=<dir> CUBECL_TIME_PASSES=1 cargo test -p cubecl-cpu --lib tests::f16_ty::test_add_assign_array` writes 23 files (pliron IR after each pass, `llvm.ll`, `llvm.opt.ll`) and logs each pass duration, with no rebuild flag.

## 5. Phase 1: location capture in the frontend

**Model.** Each op gets its full location when it is inserted. The `CallSite` chain is built directly from a frame stack (the eager method).

The rejected alternative sets `SrcPos` first and wraps each op in `CallSite` when its function returns. Both methods give the same final locations, so they also use the same memory at the end. The alternative also needs a list of the ops that each call inserted, rewrites an op once for each level of nesting, and gives the wrong frame to ops that the frontend inserts outside the current call (loop flags, `yield` ops). The chain construction is in one dedicated commit (step 2a), so a maintainer can revert it and put the alternative in its place.

For a leaf op in `inner`, called from `mid`, called from kernel `k`:

```text
CallSite {
  callee: Named { "inner", SrcPos(inner.rs:L_op:C) },
  caller: CallSite {
    callee: Named { "mid", SrcPos(mid.rs:L_call_inner:C) },
    caller: Named { "k", SrcPos(k.rs:L_call_mid:C) } } }
```

Steps:

0. **Detect the setting with the cargo build-script interface.**
   - In [`cubecl-runtime/build.rs`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/build.rs#L3-L16), read `DEBUG`. If it is `true`, emit `cargo:rustc-cfg=cubecl_debug_info`. Declare the cfg with `cargo::rustc-check-cfg`.
   - In `cubecl-runtime`, add `pub const DEBUG_INFO: DebugInfo`. It is `LineTables` with the cfg and `None` without it.
   - In `KernelBuilder::new` ([`builder.rs#L128-L141`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-core/src/compute/builder.rs#L128-L141)), set the level to the higher of `settings.debug_info` and `DEBUG_INFO`, then apply the lowering from `compilation.debug_info`. This is the place where the run-time `debug_symbols` value is already set.
   - The macro cannot know `DEBUG`. Thus it must always emit the capture calls. Change the gate at [`parse/kernel.rs#L362`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/parse/kernel.rs#L362), [`parse/cube_impl.rs#L137`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/parse/cube_impl.rs#L137), [`#L200`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/parse/cube_impl.rs#L200), and [`generate/kernel.rs#L47`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/kernel.rs#L47) from `cfg!(debug_symbols) || debug_symbols` to `!no_debug_symbols`. The calls do nothing when `DebugState.enabled` is false. The cost is a larger expansion and more compile time for crates with kernels, in all builds.
   - Keep the `cubecl-macros` cfg (`debug-symbols` feature, `CUBECL_DEBUG=1`) and `#[cube(debug_symbols)]`. They select `Full`: at [`generate/kernel.rs#L416-L420`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/kernel.rs#L416-L420) and [`#L508-L510`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/kernel.rs#L508-L510), emit `.debug_info(DebugInfo::Full)` in place of `.debug_symbols()`.
1. Create `crates/cubecl-ir/src/debug.rs` with `DebugState` (context aux data, like `GlobalState` at [`scope.rs#L147-L159`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-ir/src/scope.rs#L147-L159)):
   ```rust
   pub struct DebugState { enabled: bool, frames: Vec<Frame>, cache: Option<Location> }
   struct Frame { name: Option<String>, file: Source, pos: SourcePosition }
   impl DebugState {
       pub fn enter_call(&mut self, line: u32, col: u32);   // top.pos = call site; push unnamed frame
       pub fn enter_fn(&mut self, ctx, name, file, line, col); // name/file for the top frame
       pub fn exit_call(&mut self);                         // pop; restore caller pos
       pub fn set_pos(&mut self, line: u32, col: u32);
       pub fn current(&mut self) -> Location;               // chain above; cached until next change
   }
   ```
   A frame that `enter_fn` does not name is transparent. This applies to a call into a non-`#[cube]` expand function. `current()` does not include such a frame. Set `enabled` from `KernelSettings::debug_info != None` in `Scope::root` ([`scope.rs#L445`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-ir/src/scope.rs#L445)).
2. Stamp locations at insertion. Change `OpInserter` at [`scope.rs#L64`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-ir/src/scope.rs#L64) from `IRInserter<DummyListener>` to `IRInserter<LocationListener>`. In `notify_operation_inserted`, if `DebugState.enabled` and the op has `Unknown`, set `current()`. Do the same for the rewriter in [`cubecl-core/src/frontend/branch.rs#L802`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-core/src/frontend/branch.rs#L802). The alias is used everywhere, so other call sites do not change.
3. Fill the receivers in [`cubecl-core/src/frontend/debug.rs#L8-L44`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-core/src/frontend/debug.rs#L8-L44). `debug_call_expand` calls `enter_call`, the closure, and then `exit_call`. `debug_source_expand` calls `enter_fn`. It pushes a root frame if the stack is empty. Add `debug_span_expand(scope, line, col)`, which calls `set_pos`. Add `debug_source_dir_expand(scope, dir)`. The macro emits it after `debug_source_expand` with the rustc working directory (`std::env::current_dir()` at expansion) as a string literal. A new function keeps the existing signature unchanged.
4. In [`with_span`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/expression.rs#L968-L977), emit `debug_span_expand(scope, line!(), column!());` before `#tokens`. Keep `quote_spanned!` so that `line!()` gives the user's line. Use expression granularity for all levels other than `None`. It changes only the size of the line tables, as in rustc.
5. Ops that are built with regions and inserted with `insert_at_back` (for example the `yield` ops at [`scope.rs#L601-L604`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-ir/src/scope.rs#L601-L604)) do not go through the listener. The fallback pass in §6 step 2 gives them a location.

**Verify:** the `DummyListener` alias is still at `scope.rs:64`, and `debug.rs` still has the commented-out bodies.

## 6. Phase 2: keep locations through passes, and emit DWARF (LLVM paths)

1. **Propagate in rewrites.** In [`cubecl-ir/src/rewrite.rs#L40-L77`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-ir/src/rewrite.rs#L40-L77), wrap the inner `DialectConversion` / `MatchRewrite` in a `KeepLocation<T>` adapter. In `rewrite()`, record `n = rewriter.get_listener().events.len()` and the op's `loc`. Call the inner `rewrite`. Then, for each `RecorderEvent::InsertedOperation(op)` after index `n` that has `Unknown`, set the recorded location. This covers all cubecl passes without changes to each pattern.
2. **Fallback pass.** Add `InheritLocationPass` to `crates/cubecl-ir/src/rewrite.rs`. For each op with `Unknown`, it copies the location of the previous sibling op that has a known location. If no such op exists, it uses the parent op's location. Run it last before target export (LLVM, C++, SPIR-V). It covers pliron's own passes (SROA, Mem2Reg, `builtin_to_llvm_pass`). They do not go through the adapter.
3. **Encode for export.** Create `crates/cubecl-llvm/src/shared/debug_info.rs`. Run `encode_locations(ctx, module) -> LocationTable` after `lower` ([`shared/base.rs#L399-L470`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/shared/base.rs#L399-L470)) and before `to_llvm_ir::convert_module`. It interns each distinct `Location` and attaches the metadata `"cubecl.loc" = !{!"<index>"}` with `attach_metadata`, as [`loop_hints.rs`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/shared/loop_hints.rs#L24-L42) does. Do not attach to ops that fold to constants (`llvm.constant`, `llvm.zero`, `llvm.undef`, `llvm.poison`, `llvm.addressof`). Otherwise `pliron-llvm` logs a warning for each op ([`metadata_conversions.rs#L617-L627`](https://github.com/pliron-org/pliron/blob/3517dc6c08370e486297972be8da2b98ed0cd683/pliron-llvm/src/metadata_conversions.rs#L617-L627)). See [D4](DECISIONS.md#d4-llvm-debug-data-cubecl-side-metadata-bridge-or-pliron-llvm-change) for the alternative.
4. **Decode to DWARF.** Add `LlvmModule::attach_debug_info(&self, table: &LocationTable, level)` in [`shared/llvm_module.rs`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/shared/llvm_module.rs#L54-L135). It uses `llvm_sys::debuginfo`:
   - It creates one `DIBuilder`, one `DIFile` for each `Source` (directory: the rustc working directory that the macro records; `DW_AT_comp_dir` only when `file!()` is relative), and one `DICompileUnit` (`DW_LANG_Rust`, `LineTablesOnly`, `isOptimized = true`).
   - It creates one `DISubprogram` for the entry function (`LLVMSetSubprogram`). It creates one `DISubprogram` for each distinct `Named` callee, keyed by (name, file).
   - It walks `instructions()` ([`#L316`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/shared/llvm_module.rs#L316)) and reads `cubecl.loc`. It converts `CallSite{callee, caller}` to `DILocation(callee line/col, scope = callee SP, inlinedAt = DILocation(caller))`, recursively. It calls `LLVMInstructionSetDebugLoc` and then removes `cubecl.loc`.
   - It gives each instruction without `cubecl.loc` the entry location.
   - It adds the module flags `"Debug Info Version" = 3` and `"Dwarf Version" = 4` with `add_module_flag` ([`#L106`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/shared/llvm_module.rs#L106)). Then it calls `LLVMDIBuilderFinalize`.
   - Call it right after `LlvmModule::new` at [`nvptx/codegen.rs#L95`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/nvptx/codegen.rs#L95), [`amdgpu/codegen.rs#L86`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/amdgpu/codegen.rs#L86) and [`cpu/jit/engine.rs#L158`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/cpu/jit/engine.rs#L158).
5. With `Full`: set the `DIFile` source text from the `source_text` that `debug_source_expand` already receives.
6. Add `debug_info: DebugInfo` to `PlironOptions` ([`shared/base.rs#L66-L85`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/shared/base.rs#L66-L85)). It has `Default`, so this addition is compatible. Fill it from `KernelSettings::debug_info`, lowered by `compilation.debug_info` if that is set, in [`cubecl-cpu/src/compute/server.rs#L88-L91`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cpu/src/compute/server.rs#L88-L91), [`cubecl-cuda/src/compiler.rs#L132`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cuda/src/compiler.rs#L132) and [`cubecl-hip/src/compiler.rs#L129`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-hip/src/compiler.rs#L129).
7. CUDA with LLVM: when the PTX has line info, load it with `cuModuleLoadDataEx` and `CU_JIT_GENERATE_LINE_INFO = 1`, not `load_data` ([`cubecl-cuda/src/compute/context.rs#L480`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cuda/src/compute/context.rs#L480)).
8. Add the effective `DebugInfo` level (after the lowering in step 6) to `KernelId` ([`cubecl-runtime/src/id.rs#L89-L110`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/id.rs#L89-L110)). Use a new builder method `.debug_info(level)`. Call it in the macro `id()` ([`kernel.rs#L457-L470`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/kernel.rs#L457-L470)). Add it to `Hash`.

**Verify:** `pliron-llvm` still drops `Location` in `convert_block`. If a later `pliron-llvm` exports `Location` to `!dbg`, delete steps 3 and 4 and use that export.

## 7. Phase 3: CPU JIT symbols for gdb, `perf`, `samply` and `cargo flamegraph`

1. In [`cubecl-llvm/src/cpu/jit/engine.rs#L55-L98`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/cpu/jit/engine.rs#L55-L98), replace `LLVMLLJIT::new_with_default_builder()` with a local `JitBuilder` when the level is not `None`. `JitBuilder` calls `LLVMOrcCreateLLJITBuilder` and `LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator`. The creator calls `LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager` and then `LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener` with `LLVMCreateGDBRegistrationListener()` (gdb, lldb), and also with `LLVMCreatePerfJITEventListener()` (perf jitdump) if `CUBECL_JIT_SYMBOLS=perf` is set. Keep the current path when the level is `None`.
2. If `LLVMCreatePerfJITEventListener()` returns null (LLVM built without `LLVM_USE_PERF`), log one warning and continue. The perf map from step 3 still works.
3. **Perf map.** With `CUBECL_JIT_SYMBOLS=perf` (other triggers: [D10](DECISIONS.md#d10-trigger-for-perf-symbol-files)), also append `<addr hex> <size hex> <name>` for each kernel to `/tmp/perf-<pid>.map`. This is the open perf map format. `perf script` (thus `cargo flamegraph`) and `samply` read it with no extra step.
   - Address: the result of `lookup_symbol` ([`engine.rs#L85`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/cpu/jit/engine.rs#L85)).
   - Size: install an object transform with `LLVMOrcObjectTransformLayerSetTransform(LLVMOrcLLJITGetObjTransformLayer(jit), …)`. In the transform, read the symbol sizes with `LLVMCreateBinary`, `LLVMObjectFileCopySymbolIterator` and `LLVMGetSymbolSize`, and return the buffer unchanged.
   - Name: the kernel entry name. The perf map gives one frame for each kernel. The jitdump gives lines and inlined frames.
   - Open the file one time for each process, in append mode, behind a mutex.
4. **Frame pointers.** In `cubecl-llvm/build.rs`, read `CARGO_ENCODED_RUSTFLAGS`. If it contains `force-frame-pointers=yes`, set the cfg `cubecl_frame_pointers`. With this cfg, add `"frame-pointer"="all"` to each defined function. Use `add_attributes` ([`llvm_module.rs#L190`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/shared/llvm_module.rs#L190)) in `run_pipeline` ([`engine.rs#L157-L161`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/cpu/jit/engine.rs#L157-L161)), before `run_passes`.
5. Put the user procedure in the book (`cubecl-book/src/advanced-usage/profiling.md`, new):
   ```sh
   # Cargo.toml: [profile.profiling] inherits = "release", debug = "line-tables-only"
   cargo build --profile profiling
   # Nsight Systems (CUPTI) with host spans: build with the `tracing` feature and a tracing layer
   nsys profile ./target/profiling/app
   # CPU runtime:
   CUBECL_JIT_SYMBOLS=perf cargo flamegraph --profile profiling
   CUBECL_JIT_SYMBOLS=perf perf record -k 1 --call-graph dwarf ./target/profiling/app
   perf inject --jit -i perf.data -o perf.jit.data
   perf script -i perf.jit.data | inferno-collapse-perf | inferno-flamegraph > cpu.svg
   # or: CUBECL_JIT_SYMBOLS=perf samply record ./target/profiling/app
   # CUDA / HIP runtimes: no cubecl setting
   ncu --set full --import-source yes ./target/profiling/app
   ```
   Tell the user that CPU kernels run on the runtime worker threads. Thus the kernel stacks do not start at the host call site. The `tracing` span of the launch (§2.4) connects them.

**Verify:** make sure that the bundled LLVM (`tracel-llvm-bundler 23.1.0-3`) was built with `LLVM_USE_PERF=ON`. Run the JIT dump test in §10.

## 8. Phase 4: C++ and SPIR-V line directives

1. **C++ (CUDA, HIP, Metal).** In `block_to_cpp` ([`cubecl-cpp/src/shared/branch.rs#L15-L26`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cpp/src/shared/branch.rs#L15-L26)), before each op, emit `#line <line> "<file>"` when the leaf `SrcPos` changes and the level is not `None`. C++ cannot show inlined frames, so emit the leaf location only. Add `InheritLocationPass` as the last pass, after `CollectIncludesPass` at [`cubecl-cpp/src/shared/base.rs#L293-L296`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cpp/src/shared/base.rs#L293-L296). Do not run the C++ formatter on source that has `#line` directives, or make sure that it keeps them.
2. HIP: add `-gline-tables-only` to the options at [`cubecl-hip/src/compute/context.rs#L571-L575`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-hip/src/compute/context.rs#L571-L575) when the level is not `None`.
3. Metal: the `MTLCompileOptions` at [`cubecl-metal/src/compute/context.rs#L67-L73`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-metal/src/compute/context.rs#L67-L73) have no line-table setting. **Verify** that Xcode GPU capture uses `#line` from `newLibraryWithSource`. If it does not, record Metal as "not supported".
4. **SPIR-V.** Add a location hook to `PlironBuilder` in `pliron-spirv` ([`lib.rs#L65-L74`](https://github.com/tracel-ai/tracel-rspirv/blob/daa0d7b84e2cfe0786e00e42160303ce7587e726/pliron-spirv/src/lib.rs#L65-L74)): `pub fn set_location_hook(&mut self, f: Box<dyn FnMut(&Context, &Location, &mut Builder)>)`. Call it in `op_to_spirv` ([`#L271-L278`](https://github.com/tracel-ai/tracel-rspirv/blob/daa0d7b84e2cfe0786e00e42160303ce7587e726/pliron-spirv/src/lib.rs#L271-L278)) before `to_spirv`. This is a small change in a tracel-owned crate.
   - Add `SpirvDebugFormat { Auto, OpLine, NonSemantic }` in `cubecl-core`. `Auto` is the default.
   - `OpLine` (core SPIR-V): one `OpString` for each file, then `OpLine` when the leaf `SrcPos` changes. No inlined frames.
   - `NonSemantic` (`NonSemantic.Shader.DebugInfo.100`): `OpExtInstImport`, `DebugSource`, `DebugCompilationUnit`, one `DebugFunction` for each `Named` frame, `DebugInlinedAt` for each `CallSite`, and `DebugScope` / `DebugLine` for each op. Add `OpExtension "SPV_KHR_non_semantic_info"` below SPIR-V 1.6.
   - `Auto` selects `NonSemantic` if the device supports it, else `OpLine`.
   - Device support: in [`vulkan/features.rs#L113-L122`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-wgpu/src/backend/vulkan/features.rs#L113-L122), add `KHR_SHADER_NON_SEMANTIC_INFO_NAME; API_VERSION_1_3 => non_semantic_info` to `fill_core!`. Request the extension only if the level is not `None`. Add `supports_non_semantic_info: bool` to `VulkanCompilationOptions` ([`cubecl-core/src/codegen/compiler.rs#L12-L26`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-core/src/codegen/compiler.rs#L12-L26)) and set it next to `supports_dp4a` ([`vulkan.rs#L391-L395`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-wgpu/src/backend/vulkan.rs#L391-L395)).
   - In `cubecl-spirv` ([`compiler.rs#L177-L205`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-spirv/src/compiler.rs#L177-L205)), read the format and the device support from `WgpuCompilationOptions`, and install the hook. If the user selects `NonSemantic` and the device does not support it, log one warning and use `OpLine`.
   - The user selects the format through the surface in [D11](DECISIONS.md#d11-configuration-surface-for-the-spir-v-debug-format).
5. WGSL: naga does not write source lines in WGSL. Record WGSL as "not supported".

**Verify:** `op_to_spirv` is still the single dispatch point for ops in `pliron-spirv`, and `block_to_cpp` is still the single loop over ops in `cubecl-cpp`.

## 9. Phase 5: documentation of the conventional tools

No code. The host call site of a launch is already in the `tracing` span of `launch_inner` (§2.4). GPU time for each kernel comes from CUPTI, rocprofiler, or Tracy GPU zones (`profile-tracy`). cubecl adds no flamegraph writer of its own.

1. In `cubecl-book/src/advanced-usage/profiling.md`, put one section for each tool: `perf` / `cargo flamegraph`, `samply`, Nsight Systems and Nsight Compute (CUPTI), rocprofiler, RenderDoc (SPIR-V), Tracy (`profile-tracy`), and `tracing` layers (Perfetto, OpenTelemetry).
2. For each tool, give the cargo profile, the run-time switch if there is one (§3.2), and one command.
3. State the limits: WGSL has no lines (§8 step 5). Metal status follows §8 step 3. Tools that read neither jitdump, perf maps nor the GDB JIT interface cannot name CPU JIT frames. **Verify** the Pyroscope agents: the in-process Rust agent cannot name JIT code, and eBPF agent support for perf maps is not known.

**Verify:** the `tracing::instrument` attribute is still on `launch_inner`.

## 10. Tests

| Test | Location | Assertion |
|---|---|---|
| Level detection | `crates/cubecl-core` test | In the `dev` profile, `DEBUG_INFO == LineTables`. The CI release test job checks `None`. |
| Location chain | `crates/cubecl-core/src/runtime_tests/debug_info.rs` (new) | A kernel `k → mid → inner` with `#[cube(debug_symbols)]` gives the `CallSite` chain from §5 on a leaf op. |
| Rewrite keeps location | `crates/cubecl-ir` unit test | After `CubeToLLVMPass`, no op has `Unknown`. |
| DWARF | `crates/cubecl-cpu/tests/` | The printed IR has `DISubprogram(name: "inner"` and `inlinedAt:`. The module passes the verifier. |
| C++ `#line` | `crates/cubecl-cpp` | The source has `#line <n> "<file>.rs"`. |
| SPIR-V `OpLine` | `crates/cubecl-spirv` | With `OpLine`, the disassembly has `OpLine`. |
| SPIR-V NonSemantic | `crates/cubecl-spirv` | With `NonSemantic`, the disassembly has `DebugInlinedAt`, and `spirv-val` accepts the module. |
| SPIR-V fallback | `crates/cubecl-spirv` | `Auto` without device support gives `OpLine`. |
| Perf map | `crates/cubecl-cpu/tests/` (Linux) | With `CUBECL_JIT_SYMBOLS=perf`, `/tmp/perf-<pid>.map` has one line for each kernel, with a size greater than 0. |
| Cache key | `crates/cubecl-runtime/src/id.rs` | Two `KernelId`s that differ only in `debug_info` have different hashes. |
| JIT dump | `crates/cubecl-cpu/tests/` (Linux, ignored if the listener is null) | `$JITDUMPDIR/jit-<pid>.dump` exists. |
| No change when off | all (release profile with `debug = 0`) | The generated kernel output is byte-identical to the base commit. |

## 11. Order and dependencies

```text
P0 (dump) ─┐
P1 (capture) ─► P2 (propagate + LLVM DWARF) ─► P3 (CPU JIT)
                         └──────────────► P4 (C++ / SPIR-V)
P5 (docs): after P3 and P4
```

Each phase is one PR. P0 is done. P4 SPIR-V needs one `pliron-spirv` release.
