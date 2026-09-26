# Plan: Kernel Profiling and Flamegraph Support

| Item | Value |
|---|---|
| Base commit | [`a1bb768ce919260eea56dbd0b59c70e55236e22d`](https://github.com/skewballfox/cubecl/commit/a1bb768ce919260eea56dbd0b59c70e55236e22d) (`main`, "fix(llvm): the GPU targets read PLANE_POS (#1714)") |
| pliron | `0.18.0`, commit [`3517dc6`](https://github.com/pliron-org/pliron/tree/3517dc6c08370e486297972be8da2b98ed0cd683) |
| pliron-spirv | `0.15.0+sdk-1.4.357.0`, commit [`daa0d7b`](https://github.com/tracel-ai/tracel-rspirv/tree/daa0d7b84e2cfe0786e00e42160303ce7587e726/pliron-spirv) |
| Open decisions | [DECISIONS.md](DECISIONS.md) |
| Language | ASD-STE100 / Attempto Controlled English. Domain terms are permitted. |

All links point to the commits above. Before you implement a step on a later commit, do the check in its **Verify** line again.

---

## 1. Goal

1. A user sets one switch. Then the CPU profilers (`perf`, `samply`) and the GPU profilers (Nsight Compute, `rocprof`, RenderDoc) show kernel samples against the Rust source lines of `#[cube]` functions.
2. The debug data shows each inlined `#[cube]` function as a separate frame. A flamegraph shows the call chain inside a kernel.
3. A user can get a flamegraph of host call stacks, weighted by the device time of each kernel.
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
- `LaunchObserver` exists ([`cubecl-runtime/src/logging/observer.rs#L126-L204`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/logging/observer.rs#L126-L204)). It runs on the thread that issues the launch, and it can receive device durations. This is sufficient for a GPU-time flamegraph. No change to the launch path is necessary.
- **Defect:** `KernelId` does not include `debug_symbols` in its hash: [`cubecl-runtime/src/id.rs#L89-L110`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/id.rs#L89-L110), [`cubecl-macros/src/generate/kernel.rs#L457-L470`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/kernel.rs#L457-L470). With the `persistence` cache, a binary without debug info can be used for a run that requests debug info.

## 3. User-facing API

### 3.1 Tier 1: one switch

| Surface | Value | Effect |
|---|---|---|
| env | `CUBECL_DEBUG_OPTION=profile-native` | New value. The parser is at [`config/base.rs#L115-L139`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/config/base.rs#L115-L139). |
| `cubecl.toml` | `[profiling] native = true` | Same effect. |

`profile-native` sets these values:

- `compilation.debug_info = "line-tables"`. The optimization level does not change.
- `profiling.native.jit_symbols = "perf"` (CPU runtime).
- `profiling.native.frame_pointers = true` (CPU runtime).
- It enables location capture for each kernel, if the build permits it. See [D1](DECISIONS.md#d1-location-capture-in-the-macro-compile-time-or-run-time).

### 3.2 Tier 2: granular control

| Key | Type | Default | Env |
|---|---|---|---|
| `compilation.debug_info` | `DebugInfo { None, LineTables, Full }` | `None` | `CUBECL_DEBUG_INFO` |
| `compilation.dump_dir` | `Option<PathBuf>` (std) | `None` | `CUBECL_DEBUG_PLIRON` (name kept) |
| `compilation.time_passes` | `bool` | `false` | `CUBECL_TIME_PASSES` |
| `profiling.native.jit_symbols` | `JitSymbols { None, Perf, Gdb }` | `None` | `CUBECL_JIT_SYMBOLS` |
| `profiling.native.frame_pointers` | `bool` | `false` | — |

- `LineTables` gives lines and inlined frames. `Full` adds embedded source text (§6 step 5, §8). There are no variable locations in either level.
- The existing per-function controls stay: `#[cube(debug_symbols)]`, `#[cube(no_debug_symbols)]`, `KernelSettings::debug_symbols()`.
- There is a new function: `cubecl::profiling::FoldedStacks` (§9). It returns a guard. It writes a file when the guard drops.

### 3.3 Protections against common traps (built in)

| Trap | Protection |
|---|---|
| NVRTC `-G` or HIP `-g` turns off optimization, so the profile does not show release code. | Use `-lineinfo`, `-gline-tables-only`, and `CU_JIT_GENERATE_LINE_INFO` only. Do not change the optimization level. |
| A cached kernel without debug info is used for a debug run. | Add the effective debug level to the `KernelId` hash (§4 step 5). |
| The dump variable is set, but the binary does not contain the dump code. | Log one warning (§4 step 3). |
| The kernel has no locations because the macro cfg was off. | If `debug_info != None` and the root op has `Unknown`, log one warning that gives the rebuild command. |
| All code is inlined, so each sample goes to the kernel entry. | Use a `CallSite` chain for each op (§5). |
| An LLVM verifier error: "inlinable function call in a function with debug info must have a !dbg location". | Give each instruction a location. Use the kernel entry location as the fallback (§6 step 4). |
| JIT code is not visible to `perf`. | Use the perf JIT listener (§7). |
| `perf --call-graph fp` gives broken stacks in JIT code. | Add `frame-pointer=all` when `frame_pointers` is on (§7). |
| `TimingRequest::Resolved` runs kernels one at a time. The flamegraph then does not show pipeline time. | `FoldedStacks` uses `Deferred`. The file header shows the timing method (§9). |
| `file!()` gives a path relative to the workspace. Tools cannot find the file. | Set the `DIFile` directory to `CUBECL_SOURCE_ROOT` or to the current directory. See [D5](DECISIONS.md#d5-source-path-resolution). |

## 4. Phase 0: repair the dump switch

1. Add `dump_dir: Option<PathBuf>` (std only) to `CompilationConfig` in [`cubecl-runtime/src/config/compilation.rs#L4-L20`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/config/compilation.rs#L4-L20). Read `CUBECL_DEBUG_PLIRON` into it in [`config/base.rs#L80`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/config/base.rs#L80), near the `CUBECL_DEBUG_LOG` read. Add `CompilationConfig::kernel_dump_dir(&self, kernel_name) -> Option<PathBuf>`. This function sanitizes the name and creates the directory.
2. Replace the five helpers listed in §2.1 with calls to `kernel_dump_dir`.
3. For builds without `pliron-dump`: if `dump_dir.is_some()`, log one warning. The warning must say that the feature is off. The build-time switch itself is [D2](DECISIONS.md#d2-dump-code-build-time-feature-or-always-compiled).
4. In [`cubecl-llvm/src/shared/base.rs#L412`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/shared/base.rs#L412), set `print_after_all: ir_printing_dir.is_some()`.
5. In [`amdgpu/codegen.rs#L120-L125`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/amdgpu/codegen.rs#L120-L125), return `Keep` only if a dump directory exists for this kernel.
6. Map `compilation.time_passes` to `PMConfig::time_all_passes` ([`pliron/src/pass.rs#L576`](https://github.com/pliron-org/pliron/blob/3517dc6c08370e486297972be8da2b98ed0cd683/src/pass.rs#L576)) in each place that builds a `PMConfig`. This gives compile-time profile data for each pass.

**Verify:** the four `build.rs` files still set the cfg from `CUBECL_DEBUG_PLIRON`, and `shared/base.rs:412` still has `print_after_all: true`. **Done when:** `grep -rn CUBECL_DEBUG_PLIRON crates` finds only `config/base.rs` (and `build.rs` files, if D2 keeps them).

## 5. Phase 1: location capture in the frontend

**Model.** Each op gets its full location when it is inserted. The `CallSite` chain is built directly from a frame stack. The maintainer's method sets `SrcPos` first and wraps it in `CallSite` when the function returns. The output of the two methods is the same. The eager method does not need a second walk. See [D3](DECISIONS.md#d3-location-chain-eager-or-rewritten-at-scope-exit).

For a leaf op in `inner`, called from `mid`, called from kernel `k`:

```text
CallSite {
  callee: Named { "inner", SrcPos(inner.rs:L_op:C) },
  caller: CallSite {
    callee: Named { "mid", SrcPos(mid.rs:L_call_inner:C) },
    caller: Named { "k", SrcPos(k.rs:L_call_mid:C) } } }
```

Steps:

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
   A frame that `enter_fn` does not name is transparent. This applies to a call into a non-`#[cube]` expand function. `current()` does not include such a frame. Set `enabled` from `KernelSettings::debug_symbols` in `Scope::root` ([`scope.rs#L445`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-ir/src/scope.rs#L445)).
2. Stamp locations at insertion. Change `OpInserter` at [`scope.rs#L64`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-ir/src/scope.rs#L64) from `IRInserter<DummyListener>` to `IRInserter<LocationListener>`. In `notify_operation_inserted`, if `DebugState.enabled` and the op has `Unknown`, set `current()`. Do the same for the rewriter in [`cubecl-core/src/frontend/branch.rs#L802`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-core/src/frontend/branch.rs#L802). The alias is used everywhere, so other call sites do not change.
3. Fill the receivers in [`cubecl-core/src/frontend/debug.rs#L8-L44`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-core/src/frontend/debug.rs#L8-L44). `debug_call_expand` calls `enter_call`, the closure, and then `exit_call`. `debug_source_expand` calls `enter_fn`. It pushes a root frame if the stack is empty. Add `debug_span_expand(scope, line, col)`, which calls `set_pos`.
4. In [`with_span`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/expression.rs#L968-L977), emit `debug_span_expand(scope, line!(), column!());` before `#tokens`. Keep `quote_spanned!` so that `line!()` gives the user's line. The granularity is [D9](DECISIONS.md#d9-location-granularity).
5. Ops that are built with regions and inserted with `insert_at_back` (for example the `yield` ops at [`scope.rs#L601-L604`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-ir/src/scope.rs#L601-L604)) do not go through the listener. The fallback pass in §6 step 2 gives them a location.

**Verify:** the `DummyListener` alias is still at `scope.rs:64`, and `debug.rs` still has the commented-out bodies.

## 6. Phase 2: keep locations through passes, and emit DWARF (LLVM paths)

1. **Propagate in rewrites.** In [`cubecl-ir/src/rewrite.rs#L40-L77`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-ir/src/rewrite.rs#L40-L77), wrap the inner `DialectConversion` / `MatchRewrite` in a `KeepLocation<T>` adapter. In `rewrite()`, record `n = rewriter.get_listener().events.len()` and the op's `loc`. Call the inner `rewrite`. Then, for each `RecorderEvent::InsertedOperation(op)` after index `n` that has `Unknown`, set the recorded location. This covers all cubecl passes without changes to each pattern.
2. **Fallback pass.** Add `InheritLocationPass` to `crates/cubecl-ir/src/rewrite.rs`. For each op with `Unknown`, it copies the location of the previous sibling op that has a known location. If no such op exists, it uses the parent op's location. Run it last before target export (LLVM, C++, SPIR-V). It covers pliron's own passes (SROA, Mem2Reg, `builtin_to_llvm_pass`). They do not go through the adapter.
3. **Encode for export.** Create `crates/cubecl-llvm/src/shared/debug_info.rs`. Run `encode_locations(ctx, module) -> LocationTable` after `lower` ([`shared/base.rs#L399-L470`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/shared/base.rs#L399-L470)) and before `to_llvm_ir::convert_module`. It interns each distinct `Location` and attaches the metadata `"cubecl.loc" = !{!"<index>"}` with `attach_metadata`, as [`loop_hints.rs`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/shared/loop_hints.rs#L24-L42) does. Do not attach to ops that fold to constants (`llvm.constant`, `llvm.zero`, `llvm.undef`, `llvm.poison`, `llvm.addressof`). Otherwise `pliron-llvm` logs a warning for each op ([`metadata_conversions.rs#L617-L627`](https://github.com/pliron-org/pliron/blob/3517dc6c08370e486297972be8da2b98ed0cd683/pliron-llvm/src/metadata_conversions.rs#L617-L627)). See [D4](DECISIONS.md#d4-llvm-debug-info-cubecl-side-metadata-bridge-or-pliron-llvm-change) for the alternative.
4. **Decode to DWARF.** Add `LlvmModule::attach_debug_info(&self, table: &LocationTable, level)` in [`shared/llvm_module.rs`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/shared/llvm_module.rs#L54-L135). It uses `llvm_sys::debuginfo`:
   - It creates one `DIBuilder`, one `DIFile` for each `Source`, and one `DICompileUnit` (`DW_LANG_Rust`, `LineTablesOnly`, `isOptimized = true`).
   - It creates one `DISubprogram` for the entry function (`LLVMSetSubprogram`). It creates one `DISubprogram` for each distinct `Named` callee, keyed by (name, file).
   - It walks `instructions()` ([`#L316`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/shared/llvm_module.rs#L316)) and reads `cubecl.loc`. It converts `CallSite{callee, caller}` to `DILocation(callee line/col, scope = callee SP, inlinedAt = DILocation(caller))`, recursively. It calls `LLVMInstructionSetDebugLoc` and then removes `cubecl.loc`.
   - It gives each instruction without `cubecl.loc` the entry location.
   - It adds the module flags `"Debug Info Version" = 3` and `"Dwarf Version" = 4` with `add_module_flag` ([`#L106`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/shared/llvm_module.rs#L106)). Then it calls `LLVMDIBuilderFinalize`.
   - Call it right after `LlvmModule::new` at [`nvptx/codegen.rs#L95`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/nvptx/codegen.rs#L95), [`amdgpu/codegen.rs#L86`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/amdgpu/codegen.rs#L86) and [`cpu/jit/engine.rs#L158`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/cpu/jit/engine.rs#L158).
5. With `Full`: set the `DIFile` source text from the `source_text` that `debug_source_expand` already receives.
6. Add `debug_info: DebugInfo` to `PlironOptions` ([`shared/base.rs#L66-L85`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/shared/base.rs#L66-L85)). It has `Default`, so this addition is compatible. Fill it from the config in [`cubecl-cpu/src/compute/server.rs#L88-L91`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cpu/src/compute/server.rs#L88-L91), [`cubecl-cuda/src/compiler.rs#L132`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cuda/src/compiler.rs#L132) and [`cubecl-hip/src/compiler.rs#L129`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-hip/src/compiler.rs#L129).
7. CUDA with LLVM: when the PTX has line info, load it with `cuModuleLoadDataEx` and `CU_JIT_GENERATE_LINE_INFO = 1`, not `load_data` ([`cubecl-cuda/src/compute/context.rs#L480`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cuda/src/compute/context.rs#L480)).
8. Add the effective `DebugInfo` level to `KernelId` ([`cubecl-runtime/src/id.rs#L89-L110`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/id.rs#L89-L110)). Use a new builder method `.debug_info(level)`. Call it in the macro `id()` ([`kernel.rs#L457-L470`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/kernel.rs#L457-L470)). Add it to `Hash`.

**Verify:** `pliron-llvm` still drops `Location` in `convert_block`. If a later `pliron-llvm` exports `Location` to `!dbg`, delete steps 3 and 4 and use that export.

## 7. Phase 3: CPU JIT symbols for `perf` / `samply`

1. In [`cubecl-llvm/src/cpu/jit/engine.rs#L55-L98`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/cpu/jit/engine.rs#L55-L98), replace `LLVMLLJIT::new_with_default_builder()` with a local `JitBuilder` when `jit_symbols != None`. `JitBuilder` calls `LLVMOrcCreateLLJITBuilder` and `LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator`. The creator calls `LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager` and then `LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener` with `LLVMCreatePerfJITEventListener()` (`Perf`) or `LLVMCreateGDBRegistrationListener()` (`Gdb`). Keep the current path when `jit_symbols == None`.
2. If `LLVMCreatePerfJITEventListener()` returns null (LLVM built without `LLVM_USE_PERF`), log one warning and continue. See [D6](DECISIONS.md#d6-fallback-when-the-perf-jit-listener-is-unavailable).
3. If `frame_pointers` is on, add `"frame-pointer"="all"` to each defined function. Use `add_attributes` ([`llvm_module.rs#L190`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/shared/llvm_module.rs#L190)) in `run_pipeline` ([`engine.rs#L157-L161`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-llvm/src/cpu/jit/engine.rs#L157-L161)), before `run_passes`.
4. Put the user procedure in the book (`cubecl-book/src/advanced-usage/profiling.md`, new):
   ```sh
   CUBECL_DEBUG=1 CUBECL_DEBUG_OPTION=profile-native cargo build --release
   perf record -k 1 --call-graph dwarf ./target/release/app
   perf inject --jit -i perf.data -o perf.jit.data
   perf script -i perf.jit.data | inferno-collapse-perf | inferno-flamegraph > cpu.svg
   # or: samply record ./target/release/app
   ```
   Tell the user that CPU kernels run on the runtime worker threads. Thus the kernel stacks do not start at the host call site. Use §9 for host attribution.

**Verify:** make sure that the bundled LLVM (`tracel-llvm-bundler 23.1.0-3`) was built with `LLVM_USE_PERF=ON`. Run the JIT dump test in §10.

## 8. Phase 4: C++ and SPIR-V line directives

1. **C++ (CUDA, HIP, Metal).** In `block_to_cpp` ([`cubecl-cpp/src/shared/branch.rs#L15-L26`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cpp/src/shared/branch.rs#L15-L26)), before each op, emit `#line <line> "<file>"` when the leaf `SrcPos` changes and the level is not `None`. C++ cannot show inlined frames, so emit the leaf location only. Add `InheritLocationPass` as the last pass, after `CollectIncludesPass` at [`cubecl-cpp/src/shared/base.rs#L293-L296`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-cpp/src/shared/base.rs#L293-L296). Do not run the C++ formatter on source that has `#line` directives, or make sure that it keeps them.
2. HIP: add `-gline-tables-only` to the options at [`cubecl-hip/src/compute/context.rs#L571-L575`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-hip/src/compute/context.rs#L571-L575) when the level is not `None`.
3. Metal: the `MTLCompileOptions` at [`cubecl-metal/src/compute/context.rs#L67-L73`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-metal/src/compute/context.rs#L67-L73) have no line-table setting. **Verify** that Xcode GPU capture uses `#line` from `newLibraryWithSource`. If it does not, record Metal as "not supported".
4. **SPIR-V.** Add a location hook to `PlironBuilder` in `pliron-spirv` ([`lib.rs#L65-L74`](https://github.com/tracel-ai/tracel-rspirv/blob/daa0d7b84e2cfe0786e00e42160303ce7587e726/pliron-spirv/src/lib.rs#L65-L74)): `pub fn set_location_hook(&mut self, f: Box<dyn FnMut(&Context, &Location, &mut Builder)>)`. Call it in `op_to_spirv` ([`#L271-L278`](https://github.com/tracel-ai/tracel-rspirv/blob/daa0d7b84e2cfe0786e00e42160303ce7587e726/pliron-spirv/src/lib.rs#L271-L278)) before `to_spirv`. This is a small change in a tracel-owned crate. In `cubecl-spirv` ([`compiler.rs#L177-L205`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-spirv/src/compiler.rs#L177-L205)), install a hook that emits line info. The format is [D7](DECISIONS.md#d7-spir-v-line-format).
5. WGSL: naga does not write source lines in WGSL. Record WGSL as "not supported".

**Verify:** `op_to_spirv` is still the single dispatch point for ops in `pliron-spirv`, and `block_to_cpp` is still the single loop over ops in `cubecl-cpp`.

## 9. Phase 5: GPU-time flamegraph of host stacks

1. Create `crates/cubecl-runtime/src/logging/folded.rs` with `FoldedStacks: LaunchObserver` ([`observer.rs#L126-L204`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/logging/observer.rs#L126-L204)):
   - `timing()` returns `Deferred`.
   - `launched(kernel)` reads the current host stack (the source is [D8](DECISIONS.md#d8-host-stack-source-for-foldedstacks)) and stores `(stack, kernel)`.
   - `profiled` / `timed` add the duration in nanoseconds to that key.
   - When the guard drops, it writes `frame;frame;…;kernel <ns>` lines. `inferno-flamegraph`, `flamegraph.pl` and speedscope accept this format. The first line is a comment with the `TimingMethod`.
2. API: `FoldedStacks::install(path) -> FoldedStacksGuard`. It is re-exported as `cubecl::profiling::FoldedStacks`.
3. Put the procedure in the book page from §7 step 4.

**Verify:** `LaunchObserver` still has `launched`, `timing`, `profiled` and `timed`, and `launched` still runs on the thread that issues the launch.

## 10. Tests

| Test | Location | Assertion |
|---|---|---|
| Location chain | `crates/cubecl-core/src/runtime_tests/debug_info.rs` (new) | A kernel `k → mid → inner` expanded with `debug_symbols` gives the `CallSite` chain from §5 on a leaf op. |
| Rewrite keeps location | `crates/cubecl-ir` unit test | After `CubeToLLVMPass`, no op has `Unknown`. |
| DWARF | `crates/cubecl-cpu/tests/` | The printed IR has `DISubprogram(name: "inner"` and `inlinedAt:`. The module passes the verifier. |
| C++ `#line` | `crates/cubecl-cpp` | The source has `#line <n> "<file>.rs"`. |
| SPIR-V | `crates/cubecl-spirv` | The disassembly has the line format from D7. |
| Cache key | `crates/cubecl-runtime/src/id.rs` | Two `KernelId`s that differ only in `debug_info` have different hashes. |
| JIT dump | `crates/cubecl-cpu/tests/` (Linux, ignored if the listener is null) | `$JITDUMPDIR/jit-<pid>.dump` exists. |
| Folded output | `crates/cubecl-runtime` | The output lines match `^[^ ]+ \d+$`. |
| No change when off | all | With `debug_info = None`, the generated output is byte-identical to the base commit. |

## 11. Order and dependencies

```text
P0 (dump) ─┐
P1 (capture) ─► P2 (propagate + LLVM DWARF) ─► P3 (CPU JIT)
                         └──────────────► P4 (C++ / SPIR-V)
P5 (folded stacks): no dependency
```

Each phase is one PR. P0 and P5 can start now. P4 SPIR-V needs one `pliron-spirv` release.
