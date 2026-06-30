//! xDS client for proxies to connect to control plane.

use super::types::*;
use crate::{Error, Result};

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// xDS client for sidecar proxy.
pub struct XdsClient {
    /// Control plane address.
    control_plane: SocketAddr,
    /// Node identifier.
    node: Node,
    /// Current cluster configurations.
    clusters: Arc<RwLock<HashMap<String, Cluster>>>,
    /// Current endpoint configurations.
    endpoints: Arc<RwLock<HashMap<String, ClusterLoadAssignment>>>,
    /// Current listener configurations.
    listeners: Arc<RwLock<HashMap<String, Listener>>>,
    /// Current route configurations.
    routes: Arc<RwLock<HashMap<String, RouteConfiguration>>>,
    /// Version info by type.
    versions: Arc<RwLock<HashMap<String, String>>>,
    /// Connection state.
    connected: Arc<RwLock<bool>>,
    /// Retry configuration.
    retry_config: RetryConfig,
}

/// Retry configuration for xDS client.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Initial backoff duration.
    pub initial_backoff: Duration,
    /// Maximum backoff duration.
    pub max_backoff: Duration,
    /// Backoff multiplier.
    pub backoff_multiplier: f64,
    /// Maximum retries before giving up (0 = infinite).
    pub max_retries: u32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            max_retries: 0, // Infinite retries
        }
    }
}

/// Update callback type.
pub type UpdateCallback = Box<dyn Fn(&str, &[u8]) + Send + Sync>;

impl XdsClient {
    /// Create a new xDS client.
    pub fn new(control_plane: SocketAddr, node: Node) -> Self {
        Self {
            control_plane,
            node,
            clusters: Arc::new(RwLock::new(HashMap::new())),
            endpoints: Arc::new(RwLock::new(HashMap::new())),
            listeners: Arc::new(RwLock::new(HashMap::new())),
            routes: Arc::new(RwLock::new(HashMap::new())),
            versions: Arc::new(RwLock::new(HashMap::new())),
            connected: Arc::new(RwLock::new(false)),
            retry_config: RetryConfig::default(),
        }
    }

    /// Set retry configuration.
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }

    /// Check if client is connected.
    pub async fn is_connected(&self) -> bool {
        *self.connected.read().await
    }

    /// Get current clusters.
    pub async fn get_clusters(&self) -> Vec<Cluster> {
        self.clusters.read().await.values().cloned().collect()
    }

    /// Get a specific cluster.
    pub async fn get_cluster(&self, name: &str) -> Option<Cluster> {
        self.clusters.read().await.get(name).cloned()
    }

    /// Get current endpoints.
    pub async fn get_endpoints(&self, cluster: &str) -> Option<ClusterLoadAssignment> {
        self.endpoints.read().await.get(cluster).cloned()
    }

    /// Get all endpoints.
    pub async fn get_all_endpoints(&self) -> Vec<ClusterLoadAssignment> {
        self.endpoints.read().await.values().cloned().collect()
    }

    /// Get current listeners.
    pub async fn get_listeners(&self) -> Vec<Listener> {
        self.listeners.read().await.values().cloned().collect()
    }

    /// Get a specific listener.
    pub async fn get_listener(&self, name: &str) -> Option<Listener> {
        self.listeners.read().await.get(name).cloned()
    }

    /// Get route configurations.
    pub async fn get_route_configs(&self) -> Vec<RouteConfiguration> {
        self.routes.read().await.values().cloned().collect()
    }

    /// Get a specific route configuration.
    pub async fn get_route_config(&self, name: &str) -> Option<RouteConfiguration> {
        self.routes.read().await.get(name).cloned()
    }

    /// Create a discovery request.
    fn create_request(&self, type_url: &str, resource_names: Vec<String>) -> DiscoveryRequest {
        DiscoveryRequest {
            version_info: String::new(),
            node: self.node.clone(),
            resource_names,
            type_url: type_url.to_string(),
            response_nonce: String::new(),
            error_detail: None,
        }
    }

    /// Process a discovery response.
    pub async fn process_response(&self, response: DiscoveryResponse) -> Result<()> {
        let type_url = response.type_url.as_str();

        // Update version
        {
            let mut versions = self.versions.write().await;
            versions.insert(type_url.to_string(), response.version_info.clone());
        }

        // Process resources based on type
        match type_url {
            TYPE_URL_CDS => {
                let mut clusters = self.clusters.write().await;
                for resource in response.resources {
                    if let Ok(cluster) = serde_json::from_slice::<Cluster>(&resource.value) {
                        info!("Received cluster: {}", cluster.name);
                        clusters.insert(cluster.name.clone(), cluster);
                    }
                }
            }
            TYPE_URL_EDS => {
                let mut endpoints = self.endpoints.write().await;
                for resource in response.resources {
                    if let Ok(assignment) = serde_json::from_slice::<ClusterLoadAssignment>(&resource.value) {
                        info!("Received endpoints for cluster: {}", assignment.cluster_name);
                        endpoints.insert(assignment.cluster_name.clone(), assignment);
                    }
                }
            }
            TYPE_URL_LDS => {
                let mut listeners = self.listeners.write().await;
                for resource in response.resources {
                    if let Ok(listener) = serde_json::from_slice::<Listener>(&resource.value) {
                        info!("Received listener: {}", listener.name);
                        listeners.insert(listener.name.clone(), listener);
                    }
                }
            }
            TYPE_URL_RDS => {
                let mut routes = self.routes.write().await;
                for resource in response.resources {
                    if let Ok(route_config) = serde_json::from_slice::<RouteConfiguration>(&resource.value) {
                        info!("Received route config: {}", route_config.name);
                        routes.insert(route_config.name.clone(), route_config);
                    }
                }
            }
            _ => {
                warn!("Unknown resource type: {}", type_url);
            }
        }

        Ok(())
    }

    /// Fetch clusters from control plane (sync request-response pattern).
    pub async fn fetch_clusters(&self, names: Vec<String>) -> Result<Vec<Cluster>> {
        let request = self.create_request(TYPE_URL_CDS, names);

        // In a real implementation, this would make an HTTP/gRPC request
        // For now, return cached clusters
        let clusters = self.clusters.read().await;
        if request.resource_names.is_empty() {
            Ok(clusters.values().cloned().collect())
        } else {
            Ok(request.resource_names
                .iter()
                .filter_map(|name| clusters.get(name).cloned())
                .collect())
        }
    }

    /// Fetch endpoints from control plane.
    pub async fn fetch_endpoints(&self, cluster_names: Vec<String>) -> Result<Vec<ClusterLoadAssignment>> {
        let request = self.create_request(TYPE_URL_EDS, cluster_names);

        let endpoints = self.endpoints.read().await;
        if request.resource_names.is_empty() {
            Ok(endpoints.values().cloned().collect())
        } else {
            Ok(request.resource_names
                .iter()
                .filter_map(|name| endpoints.get(name).cloned())
                .collect())
        }
    }

    /// Request specific resources from control plane.
    pub fn create_discovery_request(
        &self,
        type_url: &str,
        resource_names: Vec<String>,
        version_info: &str,
        response_nonce: &str,
    ) -> DiscoveryRequest {
        DiscoveryRequest {
            version_info: version_info.to_string(),
            node: self.node.clone(),
            resource_names,
            type_url: type_url.to_string(),
            response_nonce: response_nonce.to_string(),
            error_detail: None,
        }
    }

    /// Create a NACK response for rejected configuration.
    pub fn create_nack(
        &self,
        type_url: &str,
        version_info: &str,
        response_nonce: &str,
        error_message: &str,
    ) -> DiscoveryRequest {
        DiscoveryRequest {
            version_info: version_info.to_string(),
            node: self.node.clone(),
            resource_names: vec![],
            type_url: type_url.to_string(),
            response_nonce: response_nonce.to_string(),
            error_detail: Some(Status {
                code: 3, // INVALID_ARGUMENT
                message: error_message.to_string(),
            }),
        }
    }

    /// Get the current version for a resource type.
    pub async fn get_version(&self, type_url: &str) -> Option<String> {
        self.versions.read().await.get(type_url).cloned()
    }

    /// Clear all cached resources.
    pub async fn clear_cache(&self) {
        self.clusters.write().await.clear();
        self.endpoints.write().await.clear();
        self.listeners.write().await.clear();
        self.routes.write().await.clear();
        self.versions.write().await.clear();
    }

    /// Get client statistics.
    pub async fn stats(&self) -> XdsClientStats {
        XdsClientStats {
            cluster_count: self.clusters.read().await.len(),
            endpoint_count: self.endpoints.read().await.len(),
            listener_count: self.listeners.read().await.len(),
            route_count: self.routes.read().await.len(),
            connected: *self.connected.read().await,
        }
    }
}

/// xDS client statistics.
#[derive(Debug, Clone)]
pub struct XdsClientStats {
    pub cluster_count: usize,
    pub endpoint_count: usize,
    pub listener_count: usize,
    pub route_count: usize,
    pub connected: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_node() -> Node {
        Node {
            id: "test-proxy".to_string(),
            cluster: "test-cluster".to_string(),
            metadata: HashMap::new(),
            locality: Some(Locality {
                region: "us-east-1".to_string(),
                zone: "us-east-1a".to_string(),
                sub_zone: String::new(),
            }),
        }
    }

    #[tokio::test]
    async fn test_client_creation() {
        let addr = "127.0.0.1:18000".parse().unwrap();
        let client = XdsClient::new(addr, test_node());

        assert!(!client.is_connected().await);
        assert!(client.get_clusters().await.is_empty());
    }

    #[tokio::test]
    async fn test_process_cds_response() {
        let addr = "127.0.0.1:18001".parse().unwrap();
        let client = XdsClient::new(addr, test_node());

        let cluster = Cluster {
            name: "my-service".to_string(),
            connect_timeout: Duration::from_secs(5),
            cluster_type: ClusterType::Eds,
            lb_policy: LbPolicy::RoundRobin,
            health_checks: vec![],
            circuit_breakers: None,
            tls_context: None,
        };

        let response = DiscoveryResponse {
            version_info: "v1".to_string(),
            resources: vec![Resource {
                type_url: TYPE_URL_CDS.to_string(),
                value: serde_json::to_vec(&cluster).unwrap(),
            }],
            type_url: TYPE_URL_CDS.to_string(),
            nonce: "nonce1".to_string(),
        };

        client.process_response(response).await.unwrap();

        let clusters = client.get_clusters().await;
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].name, "my-service");
    }

    #[tokio::test]
    async fn test_process_eds_response() {
        let addr = "127.0.0.1:18002".parse().unwrap();
        let client = XdsClient::new(addr, test_node());

        let assignment = ClusterLoadAssignment {
            cluster_name: "my-service".to_string(),
            endpoints: vec![LocalityLbEndpoints {
                locality: None,
                lb_endpoints: vec![LbEndpoint {
                    endpoint: Endpoint {
                        address: SocketAddress {
                            address: "10.0.0.1".to_string(),
                            port: 8080,
                            protocol: Protocol::Tcp,
                        },
                        health_check_config: None,
                    },
                    health_status: HealthStatus::Healthy,
                    load_balancing_weight: Some(100),
                    metadata: HashMap::new(),
                }],
                load_balancing_weight: None,
                priority: 0,
            }],
            policy: None,
        };

        let response = DiscoveryResponse {
            version_info: "v1".to_string(),
            resources: vec![Resource {
                type_url: TYPE_URL_EDS.to_string(),
                value: serde_json::to_vec(&assignment).unwrap(),
            }],
            type_url: TYPE_URL_EDS.to_string(),
            nonce: "nonce1".to_string(),
        };

        client.process_response(response).await.unwrap();

        let endpoints = client.get_endpoints("my-service").await;
        assert!(endpoints.is_some());
        let eps = endpoints.unwrap();
        assert_eq!(eps.endpoints.len(), 1);
    }

    #[tokio::test]
    async fn test_clear_cache() {
        let addr = "127.0.0.1:18003".parse().unwrap();
        let client = XdsClient::new(addr, test_node());

        // Add some data
        let cluster = Cluster {
            name: "test".to_string(),
            ..Default::default()
        };

        let response = DiscoveryResponse {
            version_info: "v1".to_string(),
            resources: vec![Resource {
                type_url: TYPE_URL_CDS.to_string(),
                value: serde_json::to_vec(&cluster).unwrap(),
            }],
            type_url: TYPE_URL_CDS.to_string(),
            nonce: "n1".to_string(),
        };

        client.process_response(response).await.unwrap();
        assert!(!client.get_clusters().await.is_empty());

        // Clear
        client.clear_cache().await;
        assert!(client.get_clusters().await.is_empty());
    }

    #[tokio::test]
    async fn test_stats() {
        let addr = "127.0.0.1:18004".parse().unwrap();
        let client = XdsClient::new(addr, test_node());

        let stats = client.stats().await;
        assert_eq!(stats.cluster_count, 0);
        assert!(!stats.connected);
    }

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.initial_backoff, Duration::from_millis(100));
        assert_eq!(config.max_retries, 0);
    }
}
