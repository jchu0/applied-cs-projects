//! OverlayFS filesystem management
//!
//! Provides overlay filesystem support for container layers.

use std::fs;
use std::path::{Path, PathBuf};

use nix::mount::{mount, umount2, MntFlags, MsFlags};

use crate::{Error, Result};

/// OverlayFS mount for container filesystem
pub struct OverlayMount {
    /// Merged directory (container root)
    pub merged: PathBuf,
    /// Upper directory (writable layer)
    pub upper: PathBuf,
    /// Work directory (required by overlayfs)
    pub work: PathBuf,
    /// Lower directories (read-only layers)
    pub lower: Vec<PathBuf>,
    /// Whether the mount is active
    mounted: bool,
}

impl OverlayMount {
    /// Create a new overlay mount specification
    pub fn new(
        merged: impl Into<PathBuf>,
        upper: impl Into<PathBuf>,
        work: impl Into<PathBuf>,
        lower: Vec<PathBuf>,
    ) -> Self {
        Self {
            merged: merged.into(),
            upper: upper.into(),
            work: work.into(),
            lower,
            mounted: false,
        }
    }

    /// Mount the overlay filesystem
    pub fn mount(&mut self) -> Result<()> {
        if self.mounted {
            return Err(Error::Mount("Overlay already mounted".to_string()));
        }

        // Create directories
        fs::create_dir_all(&self.merged)?;
        fs::create_dir_all(&self.upper)?;
        fs::create_dir_all(&self.work)?;

        // Build lowerdir option
        let lowerdir = self
            .lower
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join(":");

        // Build mount options
        let options = format!(
            "lowerdir={},upperdir={},workdir={}",
            lowerdir,
            self.upper.to_string_lossy(),
            self.work.to_string_lossy()
        );

        // Mount overlay
        mount(
            Some("overlay"),
            &self.merged,
            Some("overlay"),
            MsFlags::empty(),
            Some(options.as_str()),
        ).map_err(|e| {
            Error::Mount(format!("Failed to mount overlay: {}", e))
        })?;

        self.mounted = true;
        log::info!("Mounted overlay at {:?}", self.merged);

        Ok(())
    }

    /// Unmount the overlay filesystem
    pub fn unmount(&mut self) -> Result<()> {
        if !self.mounted {
            return Ok(());
        }

        umount2(&self.merged, MntFlags::MNT_DETACH).map_err(|e| {
            Error::Mount(format!("Failed to unmount overlay: {}", e))
        })?;

        self.mounted = false;
        log::info!("Unmounted overlay at {:?}", self.merged);

        Ok(())
    }

    /// Check if mounted
    pub fn is_mounted(&self) -> bool {
        self.mounted
    }
}

impl Drop for OverlayMount {
    fn drop(&mut self) {
        if self.mounted {
            let _ = self.unmount();
        }
    }
}

/// Layer manager for container filesystems
pub struct LayerManager {
    /// Base directory for layers
    root: PathBuf,
}

impl LayerManager {
    /// Create a new layer manager
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Create directories for a new container
    pub fn create_container_dirs(&self, container_id: &str) -> Result<ContainerDirs> {
        let base = self.root.join(container_id);

        let dirs = ContainerDirs {
            root: base.clone(),
            merged: base.join("merged"),
            upper: base.join("upper"),
            work: base.join("work"),
        };

        fs::create_dir_all(&dirs.merged)?;
        fs::create_dir_all(&dirs.upper)?;
        fs::create_dir_all(&dirs.work)?;

        Ok(dirs)
    }

    /// Remove container directories
    pub fn remove_container_dirs(&self, container_id: &str) -> Result<()> {
        let base = self.root.join(container_id);
        if base.exists() {
            fs::remove_dir_all(&base)?;
        }
        Ok(())
    }

    /// Extract a layer from a tar archive
    pub fn extract_layer(&self, layer_id: &str, tar_data: &[u8]) -> Result<PathBuf> {
        let layer_dir = self.root.join("layers").join(layer_id);
        fs::create_dir_all(&layer_dir)?;

        // Decompress if gzipped
        let reader: Box<dyn std::io::Read> = if tar_data.starts_with(&[0x1f, 0x8b]) {
            Box::new(flate2::read::GzDecoder::new(tar_data))
        } else {
            Box::new(tar_data)
        };

        // Extract tar
        let mut archive = tar::Archive::new(reader);
        archive.unpack(&layer_dir)?;

        log::info!("Extracted layer {} to {:?}", layer_id, layer_dir);
        Ok(layer_dir)
    }

    /// Get layer paths in order for overlay mount
    pub fn get_layer_paths(&self, layer_ids: &[String]) -> Vec<PathBuf> {
        layer_ids
            .iter()
            .map(|id| self.root.join("layers").join(id))
            .filter(|p| p.exists())
            .collect()
    }

    /// Create an overlay mount for a container
    pub fn create_overlay(
        &self,
        container_id: &str,
        layer_ids: &[String],
    ) -> Result<OverlayMount> {
        let dirs = self.create_container_dirs(container_id)?;
        let lower = self.get_layer_paths(layer_ids);

        if lower.is_empty() {
            return Err(Error::Mount("No layers found for overlay".to_string()));
        }

        Ok(OverlayMount::new(dirs.merged, dirs.upper, dirs.work, lower))
    }

    /// Calculate total layer size
    pub fn layer_size(&self, layer_id: &str) -> Result<u64> {
        let layer_dir = self.root.join("layers").join(layer_id);
        dir_size(&layer_dir)
    }
}

/// Container directory structure
pub struct ContainerDirs {
    /// Root directory for container
    pub root: PathBuf,
    /// Merged (mount point) directory
    pub merged: PathBuf,
    /// Upper (writable) directory
    pub upper: PathBuf,
    /// Work directory
    pub work: PathBuf,
}

/// Calculate directory size recursively
fn dir_size(path: &Path) -> Result<u64> {
    let mut size = 0;

    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;

            if metadata.is_dir() {
                size += dir_size(&entry.path())?;
            } else {
                size += metadata.len();
            }
        }
    }

    Ok(size)
}

/// Copy-on-write snapshot of a layer
pub fn create_snapshot(
    source: &Path,
    dest: &Path,
) -> Result<()> {
    // Create destination directory
    fs::create_dir_all(dest)?;

    // Copy directory structure using cp -a equivalent
    copy_dir_recursive(source, dest)?;

    Ok(())
}

/// Recursively copy directory
fn copy_dir_recursive(source: &Path, dest: &Path) -> Result<()> {
    if !dest.exists() {
        fs::create_dir_all(dest)?;
    }

    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let src_path = entry.path();
        let dst_path = dest.join(entry.file_name());

        if file_type.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else if file_type.is_symlink() {
            let target = fs::read_link(&src_path)?;
            std::os::unix::fs::symlink(target, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)?;
        }
    }

    Ok(())
}

/// Whiteout handling for deleted files in layers
pub mod whiteout {
    use std::path::Path;

    /// Prefix for whiteout files
    pub const WHITEOUT_PREFIX: &str = ".wh.";

    /// Check if a path is a whiteout file
    pub fn is_whiteout(path: &Path) -> bool {
        path.file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with(WHITEOUT_PREFIX))
            .unwrap_or(false)
    }

    /// Get the original filename from a whiteout
    pub fn get_original_name(whiteout: &Path) -> Option<String> {
        whiteout
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_prefix(WHITEOUT_PREFIX))
            .map(|s| s.to_string())
    }

    /// Check if path is opaque directory marker
    pub fn is_opaque(path: &Path) -> bool {
        path.file_name()
            .and_then(|n| n.to_str())
            .map(|n| n == ".wh..wh..opq")
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_layer_manager_new() {
        let temp = TempDir::new().unwrap();
        let _manager = LayerManager::new(temp.path());
    }

    #[test]
    fn test_container_dirs() {
        let temp = TempDir::new().unwrap();
        let manager = LayerManager::new(temp.path());
        let dirs = manager.create_container_dirs("test-container").unwrap();

        assert!(dirs.merged.exists());
        assert!(dirs.upper.exists());
        assert!(dirs.work.exists());
    }

    #[test]
    fn test_whiteout_detection() {
        use std::path::PathBuf;

        let whiteout = PathBuf::from("/tmp/.wh.deleted_file");
        assert!(whiteout::is_whiteout(&whiteout));
        assert_eq!(
            whiteout::get_original_name(&whiteout),
            Some("deleted_file".to_string())
        );

        let normal = PathBuf::from("/tmp/normal_file");
        assert!(!whiteout::is_whiteout(&normal));
    }
}
