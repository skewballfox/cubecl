//! The compile directory of the debug data, found when a kernel compiles.
//!
//! `file!()` gives a path relative to the rustc working directory, usually the workspace root.
//! A debugger or a profiler finds such a file only when it runs in that directory. The binary
//! keeps the relative path, so `--remap-path-prefix` and `trim-paths` apply to it. When a kernel
//! compiles, cubecl looks for the file, and gives the directory that contains it to the debug
//! data. Thus only the debug data of the kernel has the absolute path, on the computer that has
//! the source.
//!
//! The search occurs one time for each file in each process, not for each kernel.

use super::debug_info::md5_hex;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock, PoisonError},
};

/// The variable that names the root, for a binary that does not run in its source tree.
pub(crate) const SOURCE_ROOT_VAR: &str = "CUBECL_SOURCE_ROOT";

/// The directory for the relative paths of a kernel: the root of the first file in `files` that
/// exists under a search root. Each item is a path and, if the kernel has the text of the file,
/// the MD5 of that text. A file with an MD5 matches only a file with the same text.
///
/// The search roots are `CUBECL_SOURCE_ROOT`, then the working directory and its parents.
/// Returns `None` when no relative file exists, or the root is not UTF-8.
pub(crate) fn source_root<'a>(
    files: impl IntoIterator<Item = (&'a str, Option<&'a str>)>,
) -> Option<String> {
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
        .and_then(|root| root.to_str().map(str::to_string))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A new directory with the file `src/k.rs`, which has the text `text`.
    fn tree(name: &str, text: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("cubecl-source-root-{}-{name}", std::process::id()));
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/k.rs"), text).unwrap();
        root
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
        let path = "crates/cubecl-llvm/src/shared/source_root.rs";
        let root = source_root([(path, None)]).expect("the workspace root");
        assert!(Path::new(&root).join(path).is_file(), "{root}");
        assert_eq!(source_root([("/abs/k.rs", None)]), None);
    }
}
