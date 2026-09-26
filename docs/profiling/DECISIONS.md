# Open Decisions: Kernel Profiling and Flamegraph Support

Base commit: [`a1bb768ce919260eea56dbd0b59c70e55236e22d`](https://github.com/skewballfox/cubecl/commit/a1bb768ce919260eea56dbd0b59c70e55236e22d). Plan: [PLAN.md](PLAN.md).

## Rules for all decisions

1. The behavior must be the same as for ordinary Rust code. The user sets `debug` in a cargo profile. Then the profilers work. A choice that needs a compile-time setting **and** a run-time switch is correct only if the feature has a run-time cost or a side effect that is too high to have in every build with debug data (PLAN §3.0).
2. cubecl uses only open formats and interfaces that existing tools read (perf, samply, `cargo flamegraph`, CUPTI, rocprofiler, gdb, `tracing` layers). cubecl adds no profiler and no output format of its own.

This file lists only the open choices. The plan contains the result of each closed choice. For each open choice, the plan uses the **provisional** option, so work can start. A maintainer must confirm or change it.

| ID | Subject | Provisional | Blocks |
|---|---|---|---|
| D3 | Location chain: eager or at scope exit | A | P1 |
| D4 | LLVM debug data: cubecl bridge or pliron-llvm change | C | P2 steps 3–4 |
| D10 | Trigger for perf symbol files | A + B | P3 step 3 |
| D11 | Configuration surface for the SPIR-V debug format | A + B | P4 step 4 |

---

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

## D10. Trigger for perf symbol files

The perf jitdump and the perf map (PLAN §7) write files that stay after the process stops. Thus rule 1 requires a run-time trigger. The question is which signal starts them.

**Facts (checked in the source code):**

- `cargo flamegraph` 0.6.14 runs `perf record` (Linux) or `dtrace` / `xctrace` (macOS). It sets no environment variable on the profiled program.
- `samply` 0.13.1 sets `DOTNET_PerfMapEnabled` on the program that it starts, if the variable is not already set: `2` (jitdump) on Linux, `3` (perf map) on macOS. The name comes from .NET: `1` = both, `2` = jitdump only, `3` = perf map only.
- No JIT-neutral standard exists for "a profiler asks for JIT symbols". Each JIT runtime uses its own opt-in: .NET `DOTNET_PerfMapEnabled`, Python `PYTHONPERFSUPPORT` / `-X perf`, Node `--perf-basic-prof`, JVM `-XX:+DumpPerfMapAtExit`.
- Detection of the parent process (for example `/proc/<ppid>/comm == "perf"`) is not reliable. `perf record -p`, `perf record -a`, eBPF agents and wrapper scripts have a different parent.

| Option | For | Against |
|---|---|---|
| **A. `CUBECL_JIT_SYMBOLS=perf` (or `jitdump`, `perfmap`).** | It is the same convention as all other JIT runtimes. It is clear and not hacky. | One variable more than for native Rust code. |
| **B. Also accept `DOTNET_PerfMapEnabled` (`1`, `2`, `3`, same meanings).** | `samply record` works with no cubecl variable. | It uses a name from a different runtime. A user who profiles .NET code in the same shell also enables cubecl output. `cargo flamegraph` still needs A. |
| C. Always write the files when the level is not `None`. | `cargo flamegraph` and `samply` work with no variable, as for native code. | Each dev-profile process, each `cargo test` process included, leaves files in `/tmp` and `~/.debug/jit`. |

No clean automatic detection for `cargo flamegraph` exists. B is automatic for `samply` only.

## D11. Configuration surface for the SPIR-V debug format

The format is decided: `Auto` selects `NonSemantic.Shader.DebugInfo.100` when the device supports it, else `OpLine` (PLAN §8 step 4). This choice is about **how the user overrides `Auto`**. Each option is a code location where the setting can enter.

| Option | Code location | For | Against |
|---|---|---|---|
| **A. Automatic (default).** Device support sets `supports_non_semantic_info`. | [`vulkan/features.rs#L113-L122`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-wgpu/src/backend/vulkan/features.rs#L113-L122), [`vulkan.rs#L391-L395`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-wgpu/src/backend/vulkan.rs#L391-L395) | No setting. It is always necessary as the base. | No override alone. |
| **B. Global config: `[compilation] spirv_debug_format = "auto" \| "op-line" \| "non-semantic"` in `cubecl.toml`, and `CUBECL_SPIRV_DEBUG_FORMAT`.** | `CompilationConfig` ([`config/compilation.rs#L4-L20`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/config/compilation.rs#L4-L20)), env read in [`config/base.rs`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-runtime/src/config/base.rs#L80) | Same pattern as `check_mode` and `f16_evaluation`. No API change. It works for tools that fail on one format. | Global, not per device. |
| C. Per device: a field in `RuntimeOptions`. | [`cubecl-wgpu/src/runtime.rs#L283-L288`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-wgpu/src/runtime.rs#L283-L288) | Programmatic. Different devices can use different formats. | `RuntimeOptions` has public fields and no `#[non_exhaustive]`, so a new field breaks struct literals in user code. |
| D. Per device: a new `WgpuSetup` builder method or `init_device_with_options`. | [`runtime.rs#L315-L337`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-wgpu/src/runtime.rs#L315-L337) | Programmatic, and additive. | A new function for one setting. |
| E. Cargo feature on `cubecl-wgpu` (`spirv-non-semantic`). | `crates/cubecl-wgpu/Cargo.toml` | Simple. | It violates rule 1: the format has no run-time cost, so a build switch is not justified. Features are additive, so "force `OpLine`" cannot be a feature. |
| F. Per kernel: `KernelSettings` or `#[cube(...)]`. | [`cubecl-ir/src/settings.rs#L79-L92`](https://github.com/skewballfox/cubecl/blob/a1bb768ce919260eea56dbd0b59c70e55236e22d/crates/cubecl-ir/src/settings.rs#L79-L92) | Maximum control. | The format is a property of the device and the tool, not of the kernel. It adds noise to the kernel API. |

Provisional: A as the default, B as the override. Add D only if a user needs different formats on two devices in one process.
