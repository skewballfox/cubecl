# Profiling Kernels

CubeCL gives kernel debug data to the usual profilers and debuggers: `perf`, `samply`,
`cargo flamegraph` and gdb. It uses only open formats: DWARF, the perf jitdump, the perf map and
the GDB JIT interface. CubeCL adds no profiler of its own.

At this time, only the `cpu` runtime gives kernel source lines to these tools.

## Enable Debug Data

Kernels get debug data from the same setting as your host code: the `debug` key of the cargo
profile. The `dev` profile sets it. For a release build, add a profile:

```toml
[profile.profiling]
inherits = "release"
debug = "line-tables-only"
```

Then each kernel has line tables. Each inlined `#[cube]` function is a frame of its own, so a
flamegraph shows the call chain inside a kernel. Line tables do not change the machine code.
They add only compile time.

If you override the profile for one package, put the same override on `cubecl-runtime`. CubeCL
reads the `debug` value of `cubecl-runtime`.

To remove the debug data from a build that has it, set `CUBECL_DEBUG_INFO=none` (see
[Configuration](./config.md)).

## Profiler Symbol Files

A profiler finds JIT code only through symbol files. These files stay after the process stops, so
you must ask for them at run time with `CUBECL_JIT_SYMBOLS`:

| Value | Files |
|---|---|
| `perf` | The perf map and the perf jitdump. |
| `perfmap` | The perf map only. |
| `jitdump` | The perf jitdump only. |

- The **perf map** (`/tmp/perf-<pid>.map`) gives the name and the address range of each kernel.
  `perf report`, `perf script` and `samply` read it with no extra step.
- The **perf jitdump** (`jit-<pid>.dump` in a directory under `$JITDUMPDIR/.debug/jit`, or under
  `~/.debug/jit`) also gives the source lines and the unwind data of each kernel.
  `perf inject --jit` reads it. It needs the `jitdump` feature and Linux:

  ```toml
  cubecl = { version = "...", features = ["cpu", "jitdump"] }
  ```

  Without the feature, CubeCL logs one warning and writes only the perf map.

If `CUBECL_JIT_SYMBOLS` is not set, CubeCL reads `DOTNET_PerfMapEnabled`, which `samply` sets:
`1` gives both files, `2` the jitdump, and `3` the perf map.

A kernel without debug data writes no symbols.

## Frame Pointers

`perf --call-graph fp` walks the stack with frame pointers. Build with
`RUSTFLAGS="-C force-frame-pointers=yes"`. Then the JIT kernels keep frame pointers too.

## Commands

`cargo flamegraph` does not run `perf inject --jit`, so it uses the perf map. It must record with
frame pointers, because `perf` cannot unwind JIT code with DWARF before `perf inject`:

```sh
RUSTFLAGS="-C force-frame-pointers=yes" CUBECL_JIT_SYMBOLS=perf \
    cargo flamegraph --profile profiling -c "record -F 997 --call-graph fp -g"
```

`perf` with the jitdump gives source lines in JIT code:

```sh
cargo build --profile profiling --features jitdump
CUBECL_JIT_SYMBOLS=perf perf record -k 1 --call-graph dwarf ./target/profiling/app
perf inject --jit -i perf.data -o perf.jit.data
perf report -i perf.jit.data
```

`samply` sets `DOTNET_PerfMapEnabled`, so it needs no other setting:

```sh
samply record ./target/profiling/app
```

gdb and lldb see each kernel through the GDB JIT interface. This is automatic when the kernel has
debug data, and it writes no files. A breakpoint in a `#[cube]` function or an interrupt shows the
kernel frames with their source lines.

## Worker Threads

The `cpu` runtime runs kernels on its worker threads. Thus a kernel stack does not start at the
host function that launched the kernel. To connect them, build with the `tracing` feature: each
launch is a `tracing` span with the host call site.
