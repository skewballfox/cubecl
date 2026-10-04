//! The compile directory of the kernel debug data, found when a kernel compiles.
//!
//! `file!()` gives a path relative to the rustc working directory, usually the workspace root.
//! A debugger or a profiler finds such a file only when it runs in that directory. The binary
//! keeps the relative path, so `--remap-path-prefix` and `trim-paths` apply to it. When a kernel
//! compiles, cubecl looks for the file, and gives the directory that contains it to the debug
//! data. Thus only the debug data of the kernel has the absolute path, on the computer that has
//! the source.
//!
//! The search occurs one time for each file in each process, not for each kernel.

use alloc::{
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};
use core::{fmt::Write, hash::BuildHasher};
use cubecl_ir::{debug::DebugState, settings::DebugInfo};
use md5::{Digest, Md5};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock, PoisonError},
};

/// The variable that names the root, for a binary that does not run in its source tree.
pub const SOURCE_ROOT_VAR: &str = "CUBECL_SOURCE_ROOT";

/// The MD5 of the source text of each file of a kernel at `level`, by the path of the file. Empty
/// below [`DebugInfo::Full`], as the macro records the texts only at `Full`.
#[must_use]
pub fn source_md5s(debug: &DebugState, level: DebugInfo) -> HashMap<&str, Arc<str>> {
    match level {
        DebugInfo::Full => debug
            .sources()
            .iter()
            .map(|(path, text)| (path.as_str(), source_md5(text)))
            .collect(),
        _ => HashMap::new(),
    }
}

/// The directory for the relative source paths of a kernel with the debug data `debug`. `md5s`
/// are the MD5s of the texts that the kernel has ([`source_md5s`]). A file with an MD5 matches
/// only a file with the same text.
///
/// The search roots are `CUBECL_SOURCE_ROOT`, then the working directory and its parents.
/// Returns `None` when no directory has the files, or the directory is not UTF-8.
#[must_use]
pub fn kernel_source_root<S: BuildHasher>(
    debug: &DebugState,
    md5s: &HashMap<&str, Arc<str>, S>,
) -> Option<String> {
    let files = debug
        .files()
        .iter()
        .map(|path| (path.as_str(), md5s.get(path.as_str()).map(AsRef::as_ref)));
    source_root(files).and_then(|root| root.to_str().map(str::to_string))
}

/// The root of the first file in `files` that exists under a search root. Each item is a path
/// and, if the kernel has the text of the file, the MD5 of that text.
fn source_root<'a>(files: impl IntoIterator<Item = (&'a str, Option<&'a str>)>) -> Option<PathBuf> {
    /// The root of each path and MD5, or `None` if no root has the file.
    type Found = HashMap<(String, Option<String>), Option<PathBuf>>;
    static FOUND: OnceLock<Mutex<Found>> = OnceLock::new();
    let found = FOUND.get_or_init(Mutex::default);

    files
        .into_iter()
        .filter(|(path, _)| Path::new(path).is_relative())
        .find_map(|(path, md5)| {
            let key = (path.to_string(), md5.map(str::to_string));
            found
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .entry(key)
                .or_insert_with(|| find_root(path, md5, search_roots()))
                .clone()
        })
}

/// `CUBECL_SOURCE_ROOT`, then the working directory and its parents. They are read one time.
fn search_roots() -> &'static [PathBuf] {
    static ROOTS: OnceLock<Vec<PathBuf>> = OnceLock::new();
    ROOTS.get_or_init(|| {
        let var = std::env::var_os(SOURCE_ROOT_VAR).map(PathBuf::from);
        let cwd = std::env::current_dir().ok();
        let parents = cwd
            .iter()
            .flat_map(|cwd| cwd.ancestors().map(Path::to_path_buf));
        var.into_iter().chain(parents).collect()
    })
}

/// The first of `roots` that contains the file `path`. With `md5`, the file must have that MD5.
fn find_root<'a>(
    path: &str,
    md5: Option<&str>,
    roots: impl IntoIterator<Item = &'a PathBuf>,
) -> Option<PathBuf> {
    roots
        .into_iter()
        .find(|root| {
            let file = root.join(path);
            match md5 {
                None => file.is_file(),
                Some(md5) => std::fs::read(&file).is_ok_and(|bytes| md5_hex(&bytes) == md5),
            }
        })
        .cloned()
}

/// The MD5 of the source text `text`, calculated one time for each text in each process.
///
/// The text is a `&'static str` from `include_str!`, so its address identifies it. One MD5 takes
/// approximately 25 µs for a text of 20 KB. If the first compile is too slow, `cubecl-macros` can
/// calculate the MD5 at build time and give it to `debug_source_expand` with the text.
fn source_md5(text: &'static str) -> Arc<str> {
    /// The MD5 of each text, by its address and length.
    type Md5s = HashMap<(usize, usize), Arc<str>>;
    static MD5S: OnceLock<Mutex<Md5s>> = OnceLock::new();
    MD5S.get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .entry((text.as_ptr().addr(), text.len()))
        .or_insert_with(|| md5_hex(text).into())
        .clone()
}

/// The MD5 of `text`, in lowercase hexadecimal.
pub fn md5_hex(text: impl AsRef<[u8]>) -> String {
    Md5::digest(text)
        .iter()
        .fold(String::with_capacity(32), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A new directory `name` under the temporary directory.
    fn temporary(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(alloc::format!(
            "cubecl-debug-source-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A new directory with the file `src/k.rs`, which has the text `text`.
    fn tree(name: &str, text: &str) -> PathBuf {
        let root = temporary(name);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/k.rs"), text).unwrap();
        root
    }

    #[test]
    fn md5_hex_is_the_md5_digest() {
        assert_eq!(md5_hex(""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5_hex("abc"), "900150983cd24fb0d6963f7d28e17f72");
    }

    #[test]
    fn the_first_root_that_has_the_file() {
        let root = tree("first", "fn k() {}");
        let roots = [root.join("src"), root.clone(), PathBuf::from("/")];
        assert_eq!(find_root("src/k.rs", None, &roots), Some(root.clone()));
        assert_eq!(find_root("src/none.rs", None, &roots), None);
        std::fs::remove_dir_all(root).unwrap();
    }

    /// With the MD5 of the compiled text, a file that was changed is not the source.
    #[test]
    fn a_file_with_a_different_text_is_not_the_source() {
        let old = tree("old", "fn k() {}");
        let new = tree("new", "fn k() { changed }");
        let md5 = md5_hex("fn k() {}");
        let roots = [new.clone(), old.clone()];
        assert_eq!(find_root("src/k.rs", Some(&md5), &roots), Some(old.clone()));
        std::fs::remove_dir_all(old).unwrap();
        std::fs::remove_dir_all(new).unwrap();
    }

    /// The tests run in the crate directory. The workspace root, a parent, has the kernel file.
    #[test]
    fn the_workspace_root_is_a_parent_of_the_working_directory() {
        let path = "crates/cubecl-runtime/src/debug_source.rs";
        let root = source_root([(path, None)]).expect("the workspace root");
        assert!(root.join(path).is_file(), "{}", root.display());
        assert_eq!(source_root([("/abs/k.rs", None)]), None);
    }
}
