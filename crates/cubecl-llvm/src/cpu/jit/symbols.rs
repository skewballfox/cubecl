//! Symbol files for the profilers, in the open formats that `perf` and `samply` read.
//!
//! The files stay after the process stops, so a run-time switch must ask for them:
//! `CUBECL_JIT_SYMBOLS`, or `DOTNET_PerfMapEnabled`, which `samply` sets.

use std::{
    fs::File,
    io::Write,
    sync::{Mutex, OnceLock},
};

/// The symbol files that the environment asks for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct JitSymbols {
    /// `/tmp/perf-<pid>.map`: one line for each kernel.
    pub perf_map: bool,
    /// `jit-<pid>.dump`: lines and inlined frames, for `perf inject --jit`.
    pub jitdump: bool,
}

impl JitSymbols {
    /// The files that `CUBECL_JIT_SYMBOLS` asks for. Without it, the files that
    /// `DOTNET_PerfMapEnabled` asks for.
    pub(crate) fn from_env() -> Self {
        static FROM_ENV: OnceLock<JitSymbols> = OnceLock::new();
        *FROM_ENV.get_or_init(|| {
            Self::parse(
                std::env::var("CUBECL_JIT_SYMBOLS").ok().as_deref(),
                std::env::var("DOTNET_PerfMapEnabled").ok().as_deref(),
            )
        })
    }

    /// `cubecl` is `perf` (both files), `perfmap` or `jitdump`. Any other value asks for no
    /// file. `dotnet` has the .NET meanings: `1` both files, `2` the jitdump, `3` the perf map.
    fn parse(cubecl: Option<&str>, dotnet: Option<&str>) -> Self {
        let (perf_map, jitdump) = match (cubecl, dotnet) {
            (Some("perf"), _) => (true, true),
            (Some("perfmap"), _) => (true, false),
            (Some("jitdump"), _) => (false, true),
            (Some(_), _) => (false, false),
            (None, Some("1")) => (true, true),
            (None, Some("2")) => (false, true),
            (None, Some("3")) => (true, false),
            (None, _) => (false, false),
        };
        Self { perf_map, jitdump }
    }
}

/// The symbol sizes in the object code of one JIT.
#[derive(Default)]
pub(crate) struct SymbolSizes(Mutex<Vec<(String, u64)>>);

impl SymbolSizes {
    pub(crate) fn insert(&self, name: String, size: u64) {
        if let Ok(mut sizes) = self.0.lock() {
            sizes.push((name, size));
        }
    }

    pub(crate) fn get(&self, name: &str) -> Option<u64> {
        let sizes = self.0.lock().ok()?;
        sizes.iter().find(|(n, _)| n == name).map(|(_, size)| *size)
    }
}

/// Appends `<addr> <size> <name>` to `/tmp/perf-<pid>.map`, the perf map format.
pub(crate) fn write_perf_map(addr: u64, size: u64, name: &str) {
    static PERF_MAP: OnceLock<Option<Mutex<File>>> = OnceLock::new();
    let file = PERF_MAP.get_or_init(|| {
        let path = format!("/tmp/perf-{}.map", std::process::id());
        match File::options().create(true).append(true).open(&path) {
            Ok(file) => Some(Mutex::new(file)),
            Err(err) => {
                log::warn!("Can't open the perf map {path}: {err}");
                None
            }
        }
    });
    let Some(Ok(mut file)) = file.as_ref().map(Mutex::lock) else {
        return;
    };
    // One write for each line, so a reader never sees half a line.
    let line = format!("{addr:x} {size:x} {name}\n");
    if let Err(err) = file.write_all(line.as_bytes()) {
        log::warn!("Can't write the perf map: {err}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubecl_variable_selects_the_files() {
        let both = JitSymbols {
            perf_map: true,
            jitdump: true,
        };
        assert_eq!(JitSymbols::parse(Some("perf"), None), both);
        assert!(JitSymbols::parse(Some("perfmap"), None).perf_map);
        assert!(!JitSymbols::parse(Some("perfmap"), None).jitdump);
        assert!(JitSymbols::parse(Some("jitdump"), None).jitdump);
        assert_eq!(JitSymbols::parse(None, None), JitSymbols::default());
    }

    #[test]
    fn cubecl_variable_overrides_dotnet() {
        assert_eq!(
            JitSymbols::parse(Some("none"), Some("1")),
            JitSymbols::default()
        );
        // `samply record` sets `2` on Linux.
        let samply = JitSymbols::parse(None, Some("2"));
        assert!(samply.jitdump && !samply.perf_map);
        assert!(JitSymbols::parse(None, Some("3")).perf_map);
    }
}
