//! Image management
//!
//! Provides OCI image handling and layer management.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Digest, Error, Result};

/// Image reference (e.g., "docker.io/library/alpine:latest")
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageRef {
    /// Registry host
    pub registry: String,
    /// Repository (e.g., "library/alpine")
    pub repository: String,
    /// Tag or digest
    pub reference: String,
}

impl ImageRef {
    /// Parse an image reference string
    pub fn parse(s: &str) -> Result<Self> {
        // Simple parsing - real implementation would be more robust
        let mut parts = s.splitn(2, '/');
        let first = parts.next().unwrap_or("");

        // Check if first part is a registry (contains '.' for domain, or ':' followed by digits for port)
        let is_registry = first.contains('.') ||
            (first.contains(':') && first.split(':').nth(1).map(|p| p.chars().all(|c| c.is_ascii_digit())).unwrap_or(false));

        let (registry, remainder) = if is_registry && parts.clone().next().is_some() {
            (first.to_string(), parts.next().unwrap_or(""))
        } else {
            ("docker.io".to_string(), s)
        };

        let (repo, tag) = if let Some(colon_idx) = remainder.rfind(':') {
            (&remainder[..colon_idx], &remainder[colon_idx + 1..])
        } else {
            (remainder, "latest")
        };

        // Add "library/" prefix for official images
        let repository = if registry == "docker.io" && !repo.contains('/') {
            format!("library/{}", repo)
        } else {
            repo.to_string()
        };

        Ok(Self {
            registry,
            repository,
            reference: tag.to_string(),
        })
    }
}

impl std::fmt::Display for ImageRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}:{}", self.registry, self.repository, self.reference)
    }
}

/// OCI Image Manifest
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    /// Schema version
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    /// Media type
    #[serde(rename = "mediaType")]
    pub media_type: String,
    /// Config descriptor
    pub config: Descriptor,
    /// Layer descriptors
    pub layers: Vec<Descriptor>,
    /// Annotations
    #[serde(default)]
    pub annotations: HashMap<String, String>,
}

/// Content descriptor
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Descriptor {
    /// Media type
    #[serde(rename = "mediaType")]
    pub media_type: String,
    /// Content digest
    pub digest: String,
    /// Content size
    pub size: u64,
    /// URLs for downloading
    #[serde(default)]
    pub urls: Vec<String>,
    /// Annotations
    #[serde(default)]
    pub annotations: HashMap<String, String>,
}

/// Image configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageConfig {
    /// Architecture
    pub architecture: String,
    /// OS
    pub os: String,
    /// Config
    pub config: ContainerConfig,
    /// Rootfs
    pub rootfs: RootFs,
    /// History
    #[serde(default)]
    pub history: Vec<HistoryEntry>,
}

/// Container configuration from image
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContainerConfig {
    /// User
    #[serde(rename = "User", default)]
    pub user: String,
    /// Environment
    #[serde(rename = "Env", default)]
    pub env: Vec<String>,
    /// Entrypoint
    #[serde(rename = "Entrypoint", default)]
    pub entrypoint: Vec<String>,
    /// Cmd
    #[serde(rename = "Cmd", default)]
    pub cmd: Vec<String>,
    /// Working directory
    #[serde(rename = "WorkingDir", default)]
    pub working_dir: String,
    /// Labels
    #[serde(rename = "Labels", default)]
    pub labels: HashMap<String, String>,
}

/// Rootfs configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RootFs {
    /// Type (usually "layers")
    #[serde(rename = "type")]
    pub rootfs_type: String,
    /// Layer diff IDs
    pub diff_ids: Vec<String>,
}

/// History entry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// Created timestamp
    pub created: String,
    /// Created by command
    #[serde(rename = "created_by", default)]
    pub created_by: String,
    /// Empty layer
    #[serde(default)]
    pub empty_layer: bool,
}

/// Image store for managing downloaded images
#[derive(Debug)]
pub struct ImageStore {
    /// Root directory
    root: PathBuf,
}

impl ImageStore {
    /// Create a new image store
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Get the blobs directory
    pub fn blobs_dir(&self) -> PathBuf {
        self.root.join("blobs").join("sha256")
    }

    /// Get path to a blob by digest
    pub fn blob_path(&self, digest: &Digest) -> PathBuf {
        self.blobs_dir().join(&digest.hash)
    }

    /// Check if a blob exists
    pub fn has_blob(&self, digest: &Digest) -> bool {
        self.blob_path(digest).exists()
    }

    /// Store a blob
    pub fn store_blob(&self, digest: &Digest, data: &[u8]) -> Result<()> {
        let path = self.blob_path(digest);
        std::fs::create_dir_all(path.parent().unwrap())?;
        std::fs::write(&path, data)?;
        Ok(())
    }

    /// Load a blob
    pub fn load_blob(&self, digest: &Digest) -> Result<Vec<u8>> {
        let path = self.blob_path(digest);
        if !path.exists() {
            return Err(Error::LayerNotFound(digest.to_string()));
        }
        Ok(std::fs::read(&path)?)
    }

    /// Extract layer to directory
    pub fn extract_layer(&self, digest: &Digest, dest: &Path) -> Result<()> {
        let data = self.load_blob(digest)?;

        // Decompress gzip
        let decoder = flate2::read::GzDecoder::new(&data[..]);

        // Extract tar
        let mut archive = tar::Archive::new(decoder);
        archive.unpack(dest)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_image_ref() {
        let image = ImageRef::parse("alpine:3.18").unwrap();
        assert_eq!(image.registry, "docker.io");
        assert_eq!(image.repository, "library/alpine");
        assert_eq!(image.reference, "3.18");

        let image = ImageRef::parse("nginx").unwrap();
        assert_eq!(image.reference, "latest");

        let image = ImageRef::parse("gcr.io/project/image:tag").unwrap();
        assert_eq!(image.registry, "gcr.io");
        assert_eq!(image.repository, "project/image");
        assert_eq!(image.reference, "tag");
    }
}
