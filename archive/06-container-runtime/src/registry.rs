//! Docker Registry Client
//!
//! Implements the Docker Registry HTTP API V2 for pulling images.
//! Supports Docker Hub and other OCI-compliant registries.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use base64::Engine;
use serde::{Deserialize, Serialize};

use crate::image::{Descriptor, ImageConfig, ImageRef, ImageStore, Manifest};
use crate::{Digest, Error, Result};

/// Registry authentication token
#[derive(Clone, Debug)]
pub struct AuthToken {
    /// Bearer token
    pub token: String,
    /// Expiration time (if provided)
    pub expires_in: Option<u64>,
}

/// Registry client configuration
#[derive(Clone, Debug)]
pub struct RegistryConfig {
    /// Connection timeout in seconds
    pub timeout_secs: u64,
    /// Maximum retries for failed requests
    pub max_retries: u32,
    /// User agent string
    pub user_agent: String,
    /// Skip TLS verification (for testing)
    pub insecure: bool,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 30,
            max_retries: 3,
            user_agent: "docklet/0.1.0".to_string(),
            insecure: false,
        }
    }
}

/// Docker Registry client
pub struct RegistryClient {
    config: RegistryConfig,
    client: reqwest::blocking::Client,
    /// Cached auth tokens per registry
    tokens: HashMap<String, AuthToken>,
}

impl RegistryClient {
    /// Create a new registry client
    pub fn new(config: RegistryConfig) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .danger_accept_invalid_certs(config.insecure)
            .user_agent(&config.user_agent)
            .build()
            .map_err(|e| Error::Registry(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            config,
            client,
            tokens: HashMap::new(),
        })
    }

    /// Get the base URL for a registry
    fn registry_url(&self, registry: &str) -> String {
        let host = if registry == "docker.io" {
            "registry-1.docker.io"
        } else {
            registry
        };

        if host.starts_with("localhost") || host.contains("127.0.0.1") {
            format!("http://{}", host)
        } else {
            format!("https://{}", host)
        }
    }

    /// Authenticate with Docker Hub
    fn authenticate_docker_hub(&mut self, repository: &str, actions: &[&str]) -> Result<AuthToken> {
        let scope = format!("repository:{}:{}", repository, actions.join(","));
        let url = format!(
            "https://auth.docker.io/token?service=registry.docker.io&scope={}",
            scope
        );

        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|e| Error::Registry(format!("Auth request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(Error::Registry(format!(
                "Auth failed with status: {}",
                response.status()
            )));
        }

        #[derive(Deserialize)]
        struct TokenResponse {
            token: String,
            expires_in: Option<u64>,
        }

        let token_resp: TokenResponse = response
            .json()
            .map_err(|e| Error::Registry(format!("Failed to parse auth response: {}", e)))?;

        Ok(AuthToken {
            token: token_resp.token,
            expires_in: token_resp.expires_in,
        })
    }

    /// Get or refresh auth token for a registry
    fn get_token(&mut self, image: &ImageRef, actions: &[&str]) -> Result<String> {
        let cache_key = format!("{}:{}", image.registry, image.repository);

        // Check cache
        if let Some(token) = self.tokens.get(&cache_key) {
            return Ok(token.token.clone());
        }

        // Authenticate
        let token = if image.registry == "docker.io" {
            self.authenticate_docker_hub(&image.repository, actions)?
        } else {
            // For other registries, try anonymous access first
            // Real implementation would handle WWW-Authenticate header
            AuthToken {
                token: String::new(),
                expires_in: None,
            }
        };

        let token_str = token.token.clone();
        self.tokens.insert(cache_key, token);
        Ok(token_str)
    }

    /// Make an authenticated request
    fn authenticated_request(
        &mut self,
        image: &ImageRef,
        path: &str,
        accept: &str,
    ) -> Result<reqwest::blocking::Response> {
        let base_url = self.registry_url(&image.registry);
        let url = format!("{}/v2/{}/{}", base_url, image.repository, path);

        let token = self.get_token(image, &["pull"])?;

        let mut request = self.client.get(&url).header("Accept", accept);

        if !token.is_empty() {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request
            .send()
            .map_err(|e| Error::Registry(format!("Request failed: {}", e)))?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            // Token might have expired, clear cache and retry
            let cache_key = format!("{}:{}", image.registry, image.repository);
            self.tokens.remove(&cache_key);

            let token = self.get_token(image, &["pull"])?;
            let mut request = self.client.get(&url).header("Accept", accept);

            if !token.is_empty() {
                request = request.header("Authorization", format!("Bearer {}", token));
            }

            return request
                .send()
                .map_err(|e| Error::Registry(format!("Retry request failed: {}", e)));
        }

        Ok(response)
    }

    /// Fetch image manifest
    pub fn get_manifest(&mut self, image: &ImageRef) -> Result<Manifest> {
        let accept = "application/vnd.docker.distribution.manifest.v2+json, \
                      application/vnd.oci.image.manifest.v1+json";

        let path = format!("manifests/{}", image.reference);
        let response = self.authenticated_request(image, &path, accept)?;

        if !response.status().is_success() {
            return Err(Error::Registry(format!(
                "Failed to get manifest: {} - {}",
                response.status(),
                response.text().unwrap_or_default()
            )));
        }

        let manifest: Manifest = response
            .json()
            .map_err(|e| Error::Registry(format!("Failed to parse manifest: {}", e)))?;

        Ok(manifest)
    }

    /// Fetch image configuration
    pub fn get_config(&mut self, image: &ImageRef, config_digest: &str) -> Result<ImageConfig> {
        let accept = "application/vnd.docker.container.image.v1+json, \
                      application/vnd.oci.image.config.v1+json";

        let path = format!("blobs/{}", config_digest);
        let response = self.authenticated_request(image, &path, accept)?;

        if !response.status().is_success() {
            return Err(Error::Registry(format!(
                "Failed to get config: {}",
                response.status()
            )));
        }

        let config: ImageConfig = response
            .json()
            .map_err(|e| Error::Registry(format!("Failed to parse config: {}", e)))?;

        Ok(config)
    }

    /// Download a blob (layer) to a file
    pub fn download_blob(
        &mut self,
        image: &ImageRef,
        digest: &str,
        dest: &Path,
    ) -> Result<u64> {
        let accept = "application/octet-stream";
        let path = format!("blobs/{}", digest);
        let mut response = self.authenticated_request(image, &path, accept)?;

        if !response.status().is_success() {
            return Err(Error::Registry(format!(
                "Failed to download blob: {}",
                response.status()
            )));
        }

        // Create parent directories
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = std::fs::File::create(dest)?;
        let mut total_bytes = 0u64;
        let mut buffer = vec![0u8; 8192];

        loop {
            let bytes_read = response
                .read(&mut buffer)
                .map_err(|e| Error::Registry(format!("Failed to read blob: {}", e)))?;

            if bytes_read == 0 {
                break;
            }

            file.write_all(&buffer[..bytes_read])?;
            total_bytes += bytes_read as u64;
        }

        Ok(total_bytes)
    }

    /// Pull an image and store it locally
    pub fn pull_image(&mut self, image: &ImageRef, store: &ImageStore) -> Result<PullResult> {
        log::info!("Pulling image: {}", image);

        // Get manifest
        log::debug!("Fetching manifest...");
        let manifest = self.get_manifest(image)?;

        // Get config
        log::debug!("Fetching config...");
        let config = self.get_config(image, &manifest.config.digest)?;

        // Store config blob
        let config_digest = Digest::parse(&manifest.config.digest)?;
        let config_bytes = serde_json::to_vec(&config)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        store.store_blob(&config_digest, &config_bytes)?;

        // Download layers
        let mut layers_downloaded = 0;
        let mut bytes_downloaded = 0u64;

        for (i, layer) in manifest.layers.iter().enumerate() {
            let layer_digest = Digest::parse(&layer.digest)?;

            // Skip if already exists
            if store.has_blob(&layer_digest) {
                log::debug!("Layer {} already exists, skipping", i + 1);
                continue;
            }

            log::info!(
                "Downloading layer {}/{}: {} ({} bytes)",
                i + 1,
                manifest.layers.len(),
                &layer.digest[..19],
                layer.size
            );

            let dest = store.blob_path(&layer_digest);
            let downloaded = self.download_blob(image, &layer.digest, &dest)?;

            layers_downloaded += 1;
            bytes_downloaded += downloaded;
        }

        // Store manifest
        let manifest_bytes = serde_json::to_vec(&manifest)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        let manifest_digest = Digest::sha256(&manifest_bytes);
        store.store_blob(&manifest_digest, &manifest_bytes)?;

        Ok(PullResult {
            manifest_digest,
            config_digest,
            layers: manifest.layers.iter().map(|l| l.digest.clone()).collect(),
            layers_downloaded,
            bytes_downloaded,
        })
    }
}

/// Result of pulling an image
#[derive(Debug)]
pub struct PullResult {
    /// Manifest digest
    pub manifest_digest: Digest,
    /// Config digest
    pub config_digest: Digest,
    /// Layer digests in order
    pub layers: Vec<String>,
    /// Number of new layers downloaded
    pub layers_downloaded: usize,
    /// Total bytes downloaded
    pub bytes_downloaded: u64,
}

/// Check if an image exists locally
pub fn image_exists(store: &ImageStore, image: &ImageRef) -> bool {
    // This is a simplified check - real implementation would maintain an index
    let blobs_dir = store.blobs_dir();
    blobs_dir.exists() && std::fs::read_dir(&blobs_dir).map(|d| d.count() > 0).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_url() {
        let client = RegistryClient::new(RegistryConfig::default()).unwrap();

        assert_eq!(
            client.registry_url("docker.io"),
            "https://registry-1.docker.io"
        );
        assert_eq!(
            client.registry_url("gcr.io"),
            "https://gcr.io"
        );
        assert_eq!(
            client.registry_url("localhost:5000"),
            "http://localhost:5000"
        );
    }

    #[test]
    fn test_registry_config_default() {
        let config = RegistryConfig::default();
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.max_retries, 3);
        assert!(!config.insecure);
    }
}
