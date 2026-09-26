# Open Decisions: Kernel Profiling and Flamegraph Support

Base commit: [`a1bb768ce919260eea56dbd0b59c70e55236e22d`](https://github.com/skewballfox/cubecl/commit/a1bb768ce919260eea56dbd0b59c70e55236e22d). Plan: [PLAN.md](PLAN.md).

This file lists the choices that have no clear best option. For each choice, the plan uses a **provisional** option, so work can start. A maintainer must confirm or change each provisional option.

| ID | Subject | Provisional | Blocks |
|---|---|---|---|
| D1 | Location capture: compile time or run time | A | P1 |
| D2 | Dump code: feature or always compiled | A | P0 step 3 |
| D3 | Location chain: eager or at scope exit | A | P1 |
| D4 | LLVM debug info: cubecl bridge or pliron-llvm change | A | P2 steps 3–4 |
| D5 | Source path resolution | A | P2 step 4 |
| D6 | Fallback without the perf JIT listener | A | P3 step 2 |
| D7 | SPIR-V line format | A | P4 step 4 |
| D8 | Host stack source for `FoldedStacks` | C | P5 |
| D9 | Location granularity | A | P1 step 4 |

---

## D1. Location capture in the macro: compile time or run time

Now, the macro emits the location calls only if `cubecl-macros` was compiled with `CUBECL_DEBUG=1` or `debug-symbols` ([`build.rs#L5-L16`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/build.rs#L5-L16)). A run-time switch alone cannot give locations.

| Option | For | Against |
|---|---|---|
| **A. Keep the compile-time cfg.** Log one warning when `debug_info` is set but no location exists. | No cost for users who do not profile. No change to the generated code. | The user must rebuild. The two variables `CUBECL_DEBUG` and `CUBECL_DEBUG_OPTION` are easy to confuse. |
| B. Always emit the calls. `DebugState.enabled` controls them at run time. | One switch at run time. No rebuild. | The expanded code is larger for each call and expression. Kernel expansion has one branch more for each op. The compile time of user crates increases. |
| C. B for `debug_call_expand` and `debug_source_expand` only. A for `debug_span_expand`. | Function-level frames at run time, with a small cost. | Line data inside a function still needs a rebuild. The behavior has two parts, which is harder to explain. |

**Decide with:** a measurement of the expansion time and the `cargo build` time of `cubecl-std` or `burn` with option B.

## D2. Dump code: build-time feature or always compiled

| Option | For | Against |
|---|---|---|
| **A. Keep `pliron-dump` and the `build.rs` switch.** Add the warning (PLAN §4 step 3). | No `std::fs` code and no `sanitize_filename` dependency in normal builds. | A rebuild is necessary to get dumps. |
| B. Remove the `build.rs` switch. Always compile the dump code with `std`. Only `compilation.dump_dir` controls it. | It works at run time. It uses the same model as `CUBECL_DEBUG_LOG`. | A small increase in binary size and in dependencies. It changes the meaning of an existing feature (the feature does nothing). |

**Decide with:** the maintainers' policy on binary size for `std` builds.

## D3. Location chain: eager or rewritten at scope exit

| Option | For | Against |
|---|---|---|
| **A. Eager.** Build the full `CallSite` chain at insertion from the frame stack (PLAN §5). | One pass. It is easy to test. Each op is final when it is inserted. | Each op keeps a copy of the full chain. The memory cost is O(ops × depth). A cache for each frame makes this smaller. |
| B. Lazy (the method the maintainer described). Set `SrcPos` on the leaf. At function exit, wrap each op of that call in `CallSite`. | Each op gets a short location while its function is open. It is the same as the MLIR inliner. | It must track the ops that each call inserted. Ops that a later pass moves or duplicates can get the wrong frame. It walks the ops once more at each exit. |

**Decide with:** the peak memory of expansion with A for the largest `cubecl-matmul` kernel.

## D4. LLVM debug info: cubecl-side metadata bridge or pliron-llvm change

| Option | For | Against |
|---|---|---|
| **A. cubecl bridge.** `cubecl.loc` metadata, then `llvm-sys` `DIBuilder` after `LlvmModule::new` (PLAN §6 steps 3–4). | Changes cubecl only. It works with `pliron-llvm 0.18`. It uses the text round trip that already exists. | It uses a private metadata kind as a side channel. It has about 300 lines of unsafe FFI in cubecl. Other `pliron-llvm` users do not get the feature. |
| B. `pliron-llvm` change. In `convert_block` ([`to_llvm_ir.rs#L2052-L2083`](https://github.com/pliron-org/pliron/blob/3517dc6c08370e486297972be8da2b98ed0cd683/pliron-llvm/src/to_llvm_ir.rs#L2052-L2083)), call `LLVMSetCurrentDebugLocation2` from `op.loc()`. Add a `DISubprogram` for each function. | It is the correct layer. The work helps all pliron users. cubecl has no side channel. | It needs upstream review and a release. It needs a design for `Named` and `Fused` in the upstream crate, and the `DI*` nodes are not modelled yet ([`metadata.rs#L36-L38`](https://github.com/pliron-org/pliron/blob/3517dc6c08370e486297972be8da2b98ed0cd683/pliron-llvm/src/metadata.rs#L36-L38)). |
| C. A now, B later. Remove A when B is released. | You get the feature now, with a path to the clean design. | The work is done twice. |

**Decide with:** whether the pliron maintainers accept B, and how long a release takes.

## D5. Source path resolution

`file!()` gives a path relative to the workspace root of the crate that contains the kernel. For registry crates, it gives an absolute path.

| Option | For | Against |
|---|---|---|
| **A. `DIFile(name = file!(), dir = $CUBECL_SOURCE_ROOT or the current directory at run time)`.** | No absolute paths go into the kernels. It is easy to redirect. | It is wrong if the binary does not run from the workspace root. The user must know the variable. |
| B. The macro gives an absolute path, from `Span::local_file()` (already used at [`kernel.rs#L53-L56`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/src/generate/kernel.rs#L53-L56)) or from `env!("CARGO_MANIFEST_DIR")`. | Tools find the file with no setup. | The build machine's paths go into the kernels and into the persistent cache, which is a privacy problem. It breaks reproducible builds unless the user adds `--remap-path-prefix`. |
| C. Option A, plus the embedded source (level `Full`) as a fallback. | Nsight and RenderDoc show the source when the path is wrong. | It works only with the `Full` level and with tools that read embedded source. |

## D6. Fallback when the perf JIT listener is unavailable

`LLVMCreatePerfJITEventListener()` returns null if LLVM was built without `LLVM_USE_PERF`.

| Option | For | Against |
|---|---|---|
| **A. Warn only.** | No new code. | `perf` shows no CPU kernel symbols. |
| B. Write `/tmp/perf-<pid>.map` in cubecl (`<addr> <size> <name>`). | `perf` and `samply` read it with no setup. It uses about 50 lines. | Function level only. All kernel code is inlined, so the result is one frame for each kernel. The function size must come from the object file. |
| C. Write a jitdump in cubecl, with `JIT_CODE_DEBUG_INFO` records. | Line-level data, with no dependency on the LLVM build. | About 400 lines. The line table must be read from the object file (DWARF parser). The format has little documentation. |
| D. Ask `tracel-llvm-bundler` to build with `LLVM_USE_PERF=ON`. | The correct fix, with no cubecl code. | It depends on a different project. It does not help users with their own LLVM. |

## D7. SPIR-V line format

| Option | For | Against |
|---|---|---|
| **A. `OpLine` (core SPIR-V).** | Simple. All drivers accept it. RenderDoc shows it. | No inlined frames. |
| B. `NonSemantic.Shader.DebugInfo.100`: `DebugLine`, `DebugScope`, `DebugInlinedAt`. | Full inlined call chains, the same as DWARF. | It needs `SPV_KHR_non_semantic_info` (Vulkan 1.3 or `VK_KHR_shader_non_semantic_info`). The modules are larger. Some tools and drivers do not support it fully. |
| C. A for `LineTables`, B for `Full`. | Each level has the best format for its purpose. | Two code paths and two sets of tests. |

## D8. Host stack source for `FoldedStacks`

| Option | For | Against |
|---|---|---|
| A. `tracing` span stack (`tracing::Span::current()` and its parents, from the registry). | Low cost. It uses the spans that already exist in `client.rs` and in `burn`. | It needs the `tracing` feature and a subscriber with a registry. Code without spans gives a flat graph. |
| B. `std::backtrace::Backtrace::force_capture()` on each launch. | No setup. Full call stacks. | A large cost for each launch (microseconds to milliseconds). The frames depend on the symbols in the host binary. |
| **C. A if a `tracing` subscriber is active, else B.** | It works in all cases. It uses the low-cost path where it can. | Two behaviors. Output from two runs can have different shapes. |
| D. Only an explicit guard API (`folded::region("name")`). | Deterministic and low cost. | Manual work for the user. |

## D9. Location granularity

| Option | For | Against |
|---|---|---|
| **A. Expression level** (each `with_span` call, PLAN §5 step 4). | The most precise attribution in the source. | More line table entries. The debug data is larger. The PTX has more `.loc` lines. The profile does not change. |
| B. Statement level only (`let` and expression statements). | Smaller debug data. | Samples in a long expression go to one line. |

**Decide with:** a comparison of PTX size and compile time for one large matmul kernel with A and with B.
