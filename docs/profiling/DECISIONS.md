# Open Decisions: Kernel Profiling and Flamegraph Support

Base commit: [`a1bb768ce919260eea56dbd0b59c70e55236e22d`](https://github.com/skewballfox/cubecl/commit/a1bb768ce919260eea56dbd0b59c70e55236e22d). Plan: [PLAN.md](PLAN.md).

## Rule for all decisions

The behavior must be the same as for ordinary Rust code. The user sets `debug` in a cargo profile. Then the profilers work. A choice that needs a compile-time setting **and** a run-time switch is correct only if the feature has a run-time cost or a side effect that is too high to have in every build with debug data (PLAN §3.0).

This file lists the choices that remain after this rule. For each choice, the plan uses a **provisional** option, so work can start. A maintainer must confirm or change each provisional option.

| ID | Subject | Status | Provisional | Blocks |
|---|---|---|---|---|
| D1 | How the macro detects the cargo `debug` level | Open | A | P1 step 0 |
| D2 | Dump code: feature or always compiled | Closed by the rule | B | — |
| D3 | Location chain: eager or at scope exit | Open | A | P1 |
| D4 | LLVM debug data: cubecl bridge or pliron-llvm change | Open | C | P2 steps 3–4 |
| D5 | Source path resolution | Closed by the rule | B | — |
| D6 | JIT symbol registration: default and fallback | Open | A | P3 |
| D7 | SPIR-V line format | Open | C | P4 step 4 |
| D8 | Host stack source for `FoldedStacks` | Open | C | P5 |
| D9 | Location granularity | Closed by the rule | — | — |

---

## D1. How the macro detects the cargo `debug` level

The rule needs the macro to know the `debug` level of the crate that it expands. The current cfg ([`cubecl-macros/build.rs#L5-L16`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-macros/build.rs#L5-L16)) cannot do this. It belongs to the build of `cubecl-macros`, which uses the `build-override` profile. These facts were checked with cargo 1.94.1 (PLAN §2.5).

| Option | For | Against |
|---|---|---|
| **A. The macro reads `-C debuginfo=<level>` from `std::env::args()`.** | It gives the exact level. It applies to each crate separately, as rustc does. With `debug = 0`, the macro emits no code. The generated code does not change. | It is not an official interface. A future rustc can run proc macros in a separate process, and the detection then stops silently. rust-analyzer expansions show no debug code, but this is only in the IDE. |
| B. A `build.rs` in `cubecl-core` reads `DEBUG`. It sets `pub const DEBUG_INFO: bool`. The macro always emits the location calls, and each call tests the constant. | It uses only official cargo interfaces. | It gives true or false only, with no level. It uses the profile of `cubecl-core`, not of the crate that contains the kernel. All builds get larger expanded code and a longer compile time, also with `debug = 0`. |
| C. A, and B as a fallback if the arguments have no rustc flags. | It works if rustc changes. | Two mechanisms to maintain and test. |

**Mitigation for A:** a test in CI compiles a fixture crate with `debug = "line-tables-only"` and checks that its kernels have locations. If rustc changes, this test fails.

**Decide with:** whether the maintainers accept one unofficial interface that a CI test guards.

## D2. Dump code: build-time feature or always compiled — closed

The rule selects **B**. The dumps cost something only when `CUBECL_DEBUG_PLIRON` is set, so a run-time switch is sufficient. Remove the `build.rs` switch. Compile the dump code in all `std` builds. Keep `pliron-dump` as an empty feature for compatibility (PLAN §4 step 3). The remaining cost is a small increase in binary size and the `sanitize_filename` dependency.

## D3. Location chain: eager or rewritten at scope exit

The rule does not apply. Both options give the same output.

| Option | For | Against |
|---|---|---|
| **A. Eager.** Build the full `CallSite` chain at insertion, from the frame stack (PLAN §5). | One pass. It is easy to test. Each op is final when it is inserted. | Each op keeps a copy of the full chain. The memory cost is O(ops × depth). A cache for each frame makes this smaller. |
| B. Lazy (the method the maintainer described). Set `SrcPos` on the leaf. At function exit, wrap each op of that call in `CallSite`. | Each op keeps a short location while its function is open. It is the same as the MLIR inliner. | It must track the ops that each call inserted. Ops that a later pass moves or duplicates can get the wrong frame. It walks the ops again at each exit. |

**Decide with:** the peak memory of expansion with A for the largest `cubecl-matmul` kernel.

## D4. LLVM debug data: cubecl-side metadata bridge or pliron-llvm change

The rule does not apply. The user-visible behavior is the same.

| Option | For | Against |
|---|---|---|
| A. cubecl bridge. `cubecl.loc` metadata, then the `llvm-sys` `DIBuilder` after `LlvmModule::new` (PLAN §6 steps 3–4). | It changes cubecl only. It works with `pliron-llvm 0.18`. It uses the text round trip that already exists. | It uses a private metadata kind as a side channel. It adds about 300 lines of unsafe FFI to cubecl. Other `pliron-llvm` users do not get the feature. |
| B. `pliron-llvm` change. In `convert_block` ([`to_llvm_ir.rs#L2052-L2083`](https://github.com/pliron-org/pliron/blob/3517dc6c08370e486297972be8da2b98ed0cd683/pliron-llvm/src/to_llvm_ir.rs#L2052-L2083)), call `LLVMSetCurrentDebugLocation2` from `op.loc()`. Add a `DISubprogram` for each function. | It is the correct layer. All pliron users get the feature. cubecl has no side channel. | It needs upstream review and a release. It needs an upstream design for `Named` and `Fused`. The `DI*` nodes are not modelled yet ([`metadata.rs#L36-L38`](https://github.com/pliron-org/pliron/blob/3517dc6c08370e486297972be8da2b98ed0cd683/pliron-llvm/src/metadata.rs#L36-L38)). |
| **C. A now, B later.** Remove A when B is released. | You get the feature now, with a path to the clean design. | The work is done twice. |

**Decide with:** whether the pliron maintainers accept B, and how long a release takes.

## D5. Source path resolution — closed

The rule selects **B**, the rustc behavior. rustc writes the file name as `file!()` gives it, and writes its working directory as `DW_AT_comp_dir`. The macro records `std::env::current_dir()` at expansion (PLAN §2.5) and uses it as the `DIFile` directory. Users who do not want build paths in the output use `--remap-path-prefix` or the cargo `trim-paths` option, as for other Rust code.

**Verify during P2:** `file!()` is remapped by `--remap-path-prefix`. Apply the same remap to the recorded directory. The macro can read the remap flags from the same argument list as D1.

## D6. JIT symbol registration: default and fallback

The CPU runtime makes its machine code at run time. Native Rust code needs no registration, because the symbols are in the ELF file. JIT code has no ELF file. Thus the rule does not give a result directly. The cost of each listener decides it.

**Part 1: default**

| Option | For | Against |
|---|---|---|
| **A. GDB registration automatic when the level is not `None`. perf jitdump only with `CUBECL_JIT_SYMBOLS=perf`.** | `gdb` and `lldb` work with no setting, as for native code. No files are written without a request. This is the same as the JVM (`-XX:+DumpPerfMapAtExit`) and Node (`--perf-prof`). | `perf` needs one variable more than for native code. |
| B. Both listeners automatic when the level is not `None`. | `perf` and `samply` work with no setting. It follows the rule literally. | Each process with debug data writes a jitdump file, and the files stay on disk. This is a side effect of a build setting. |

**Part 2: fallback if `LLVMCreatePerfJITEventListener()` returns null** (LLVM built without `LLVM_USE_PERF`)

| Option | For | Against |
|---|---|---|
| **A. Warn only.** | No new code. | `perf` shows no CPU kernel symbols. |
| B. Write `/tmp/perf-<pid>.map` in cubecl. | `perf` and `samply` read it. About 50 lines. | Function level only. All code is inlined, so the result is one frame for each kernel. |
| C. Write jitdump in cubecl, with `JIT_CODE_DEBUG_INFO` records. | Line-level data, independent of the LLVM build. | About 400 lines. It must read DWARF from the object file. |
| D. Ask `tracel-llvm-bundler` to build with `LLVM_USE_PERF=ON`. | The correct fix. No cubecl code. | It depends on another project. It does not help users with their own LLVM. |

**Decide with:** a check of the bundled LLVM. If it has perf support, part 2 is not necessary.

## D7. SPIR-V line format

The rule maps the format to the cargo level. A device capability limits it.

| Option | For | Against |
|---|---|---|
| A. `OpLine` (core SPIR-V) for all levels. | Simple. All drivers accept it. RenderDoc shows it. | No inlined frames. |
| B. `NonSemantic.Shader.DebugInfo.100` (`DebugLine`, `DebugScope`, `DebugInlinedAt`) for all levels. | Full inlined call chains, as in DWARF. | It needs `VK_KHR_shader_non_semantic_info` or Vulkan 1.3. The modules are larger. Tool support is not complete. |
| **C. B if the device has the extension, else A.** | Inlined frames when possible, and a working result on each device. | Two code paths and two sets of tests. |

## D8. Host stack source for `FoldedStacks`

`FoldedStacks` is a profiler that the user starts. It is outside the rule.

| Option | For | Against |
|---|---|---|
| A. The `tracing` span stack (`Span::current()` and its parents). | Low cost. It uses the spans that already exist in `client.rs` and in `burn`. | It needs the `tracing` feature and a subscriber with a registry. Code without spans gives a flat graph. |
| B. `std::backtrace::Backtrace::force_capture()` at each launch. | No setup. Full call stacks. | A large cost for each launch (microseconds to milliseconds). The frames depend on the host debug data. |
| **C. A if a `tracing` subscriber is active, else B.** | It works in all cases. It uses the low-cost path where it can. | Two behaviors. Two runs can give output with different shapes. |
| D. Only an explicit guard API (`folded::region("name")`). | Deterministic and low cost. | Manual work for the user. |

## D9. Location granularity — closed

The rule selects expression granularity for each level other than `None`, as rustc does. The granularity changes only the size of the line tables. It does not change the machine code or the run time (PLAN §5 step 4).
