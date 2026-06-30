//! xDS control plane server.

use super::types::*;
use crate::{Error, Result};

use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

/// xDS control plane server.
pub struct XdsServer {
    /// Cluster configurations (CDS).
    clusters: DashMap<String, Cluster>,
    /// Endpoint configurations (EDS).
    endpoints: DashMap<String, ClusterLoadAssignment>,
    /// Listener configurations (LDS).
    listeners: DashMap<String, Listener>,
    /// Route configurations (RDS).
    routes: DashMap<String, RouteConfiguration>,
    /// Version counter for generating version strings.
    version_counter: AtomicU64,
    /// Subscriptions by node ID.
    subscriptions: DashMap<String, Subscription>,
    /// Update notifier.
    update_tx: broadcast::Sender<ResourceUpdate>,
    /// Server address.
    address: SocketAddr,
}

/// Resource subscription.
struct Subscription {
    /// Node information.
    node: Node,
    /// Subscribed resource types.
    resource_types: Vec<String>,
    /// Last sent versions by type.
    versions: HashMap<String, String>,
}

/// Resource update notification.
#[derive(Clone, Debug)]
pub struct ResourceUpdate {
    /// Resource type URL.
    pub type_url: String,
    /// Resource names that changed.
    pub resource_names: Vec<String>,
    /// New version.
    pub version: String,
}

impl XdsServer {
    /// Create a new xDS server.
    pub fn new(address: SocketAddr) -> Self {
        let (update_tx, _) = broadcast::channel(1024);

        Self {
            clusters: DashMap::new(),
            endpoints: DashMap::new(),
            listeners: DashMap::new(),
            routes: DashMap::new(),
            version_counter: AtomicU64::new(1),
            subscriptions: DashMap::new(),
            update_tx,
            address,
        }
    }

    /// Get the server address.
    pub fn address(&self) -> SocketAddr {
        self.address
    }

    /// Generate a new version string.
    fn next_version(&self) -> String {
        let v = self.version_counter.fetch_add(1, Ordering::SeqCst);
        format!("v{}", v)
    }

    /// Subscribe to updates.
    pub fn subscribe(&self) -> broadcast::Receiver<ResourceUpdate> {
        self.update_tx.subscribe()
    }

    // =============== CDS (Cluster Discovery Service) ===============

    /// Add or update a cluster.
    pub fn set_cluster(&self, cluster: Cluster) -> Result<()> {
        let name = cluster.name.clone();
        let version = self.next_version();

        self.clusters.insert(name.clone(), cluster);

        let _ = self.update_tx.send(ResourceUpdate {
            type_url: TYPE_URL_CDS.to_string(),
            resource_names: vec![name.clone()],
            version: version.clone(),
        });

        info!("Updated cluster '{}' to version {}", name, version);
        Ok(())
    }

    /// Remove a cluster.
    pub fn remove_cluster(&self, name: &str) -> Result<()> {
        if self.clusters.remove(name).is_some() {
            let version = self.next_version();
            let _ = self.update_tx.send(ResourceUpdate {
                type_url: TYPE_URL_CDS.to_string(),
                resource_names: vec![name.to_string()],
                version,
            });
            info!("Removed cluster '{}'", name);
        }
        Ok(())
    }

    /// Get a cluster by name.
    pub fn get_cluster(&self, name: &str) -> Option<Cluster> {
        self.clusters.get(name).map(|c| c.clone())
    }

    /// Get all clusters.
    pub fn get_clusters(&self) -> Vec<Cluster> {
        self.clusters.iter().map(|e| e.value().clone()).collect()
    }

    /// Handle CDS discovery request.
    pub fn handle_cds_request(&self, request: &DiscoveryRequest) -> DiscoveryResponse {
        let clusters: Vec<Cluster> = if request.resource_names.is_empty() {
            // Return all clusters
            self.get_clusters()
        } else {
            // Return requested clusters
            request.resource_names
                .iter()
                .filter_map(|name| self.get_cluster(name))
                .collect()
        };

        let version = self.next_version();

        DiscoveryResponse {
            version_info: version.clone(),
            resources: clusters.into_iter()
                .map(|c| Resource {
                    type_url: TYPE_URL_CDS.to_string(),
                    value: serde_json::to_vec(&c).unwrap_or_default(),
                })
                .collect(),
            type_url: TYPE_URL_CDS.to_string(),
            nonce: version,
        }
    }

    // =============== EDS (Endpoint Discovery Service) ===============

    /// Set endpoints for a cluster.
    pub fn set_endpoints(&self, assignment: ClusterLoadAssignment) -> Result<()> {
        let name = assignment.cluster_name.clone();
        let version = self.next_version();

        self.endpoints.insert(name.clone(), assignment);

        let _ = self.update_tx.send(ResourceUpdate {
            type_url: TYPE_URL_EDS.to_string(),
            resource_names: vec![name.clone()],
            version: version.clone(),
        });

        info!("Updated endpoints for cluster '{}' to version {}", name, version);
        Ok(())
    }

    /// Remove endpoints for a cluster.
    pub fn remove_endpoints(&self, cluster_name: &str) -> Result<()> {
        if self.endpoints.remove(cluster_name).is_some() {
            let version = self.next_version();
            let _ = self.update_tx.send(ResourceUpdate {
                type_url: TYPE_URL_EDS.to_string(),
                resource_names: vec![cluster_name.to_string()],
                version,
            });
            info!("Removed endpoints for cluster '{}'", cluster_name);
        }
        Ok(())
    }

    /// Get endpoints for a cluster.
    pub fn get_endpoints(&self, cluster_name: &str) -> Option<ClusterLoadAssignment> {
        self.endpoints.get(cluster_name).map(|e| e.clone())
    }

    /// Get all endpoint assignments.
    pub fn get_all_endpoints(&self) -> Vec<ClusterLoadAssignment> {
        self.endpoints.iter().map(|e| e.value().clone()).collect()
    }

    /// Handle EDS discovery request.
    pub fn handle_eds_request(&self, request: &DiscoveryRequest) -> DiscoveryResponse {
        let assignments: Vec<ClusterLoadAssignment> = if request.resource_names.is_empty() {
            self.get_all_endpoints()
        } else {
            request.resource_names
                .iter()
                .filter_map(|name| self.get_endpoints(name))
                .collect()
        };

        let version = self.next_version();

        DiscoveryResponse {
            version_info: version.clone(),
            resources: assignments.into_iter()
                .map(|a| Resource {
                    type_url: TYPE_URL_EDS.to_string(),
                    value: serde_json::to_vec(&a).unwrap_or_default(),
                })
                .collect(),
            type_url: TYPE_URL_EDS.to_string(),
            nonce: version,
        }
    }

    // =============== LDS (Listener Discovery Service) ===============

    /// Add or update a listener.
    pub fn set_listener(&self, listener: Listener) -> Result<()> {
        let name = listener.name.clone();
        let version = self.next_version();

        self.listeners.insert(name.clone(), listener);

        let _ = self.update_tx.send(ResourceUpdate {
            type_url: TYPE_URL_LDS.to_string(),
            resource_names: vec![name.clone()],
            version: version.clone(),
        });

        info!("Updated listener '{}' to version {}", name, version);
        Ok(())
    }

    /// Remove a listener.
    pub fn remove_listener(&self, name: &str) -> Result<()> {
        if self.listeners.remove(name).is_some() {
            let version = self.next_version();
            let _ = self.update_tx.send(ResourceUpdate {
                type_url: TYPE_URL_LDS.to_string(),
                resource_names: vec![name.to_string()],
                version,
            });
            info!("Removed listener '{}'", name);
        }
        Ok(())
    }

    /// Get a listener by name.
    pub fn get_listener(&self, name: &str) -> Option<Listener> {
        self.listeners.get(name).map(|l| l.clone())
    }

    /// Get all listeners.
    pub fn get_listeners(&self) -> Vec<Listener> {
        self.listeners.iter().map(|e| e.value().clone()).collect()
    }

    /// Handle LDS discovery request.
    pub fn handle_lds_request(&self, request: &DiscoveryRequest) -> DiscoveryResponse {
        let listeners: Vec<Listener> = if request.resource_names.is_empty() {
            self.get_listeners()
        } else {
            request.resource_names
                .iter()
                .filter_map(|name| self.get_listener(name))
                .collect()
        };

        let version = self.next_version();

        DiscoveryResponse {
            version_info: version.clone(),
            resources: listeners.into_iter()
                .map(|l| Resource {
                    type_url: TYPE_URL_LDS.to_string(),
                    value: serde_json::to_vec(&l).unwrap_or_default(),
                })
                .collect(),
            type_url: TYPE_URL_LDS.to_string(),
            nonce: version,
        }
    }

    // =============== RDS (Route Discovery Service) ===============

    /// Add or update a route configuration.
    pub fn set_route_config(&self, route_config: RouteConfiguration) -> Result<()> {
        let name = route_config.name.clone();
        let version = self.next_version();

        self.routes.insert(name.clone(), route_config);

        let _ = self.update_tx.send(ResourceUpdate {
            type_url: TYPE_URL_RDS.to_string(),
            resource_names: vec![name.clone()],
            version: version.clone(),
        });

        info!("Updated route config '{}' to version {}", name, version);
        Ok(())
    }

    /// Remove a route configuration.
    pub fn remove_route_config(&self, name: &str) -> Result<()> {
        if self.routes.remove(name).is_some() {
            let version = self.next_version();
            let _ = self.update_tx.send(ResourceUpdate {
                type_url: TYPE_URL_RDS.to_string(),
                resource_names: vec![name.to_string()],
                version,
            });
            info!("Removed route config '{}'", name);
        }
        Ok(())
    }

    /// Get a route configuration by name.
    pub fn get_route_config(&self, name: &str) -> Option<RouteConfiguration> {
        self.routes.get(name).map(|r| r.clone())
    }

    /// Get all route configurations.
    pub fn get_route_configs(&self) -> Vec<RouteConfiguration> {
        self.routes.iter().map(|e| e.value().clone()).collect()
    }

    /// Handle RDS discovery request.
    pub fn handle_rds_request(&self, request: &DiscoveryRequest) -> DiscoveryResponse {
        let routes: Vec<RouteConfiguration> = if request.resource_names.is_empty() {
            self.get_route_configs()
        } else {
            request.resource_names
                .iter()
                .filter_map(|name| self.get_route_config(name))
                .collect()
        };

        let version = self.next_version();

        DiscoveryResponse {
            version_info: version.clone(),
            resources: routes.into_iter()
                .map(|r| Resource {
                    type_url: TYPE_URL_RDS.to_string(),
                    value: serde_json::to_vec(&r).unwrap_or_default(),
                })
                .collect(),
            type_url: TYPE_URL_RDS.to_string(),
            nonce: version,
        }
    }

    // =============== Unified Discovery Handler ===============

    /// Handle a discovery request for any resource type.
    pub fn handle_discovery_request(&self, request: &DiscoveryRequest) -> DiscoveryResponse {
        match request.type_url.as_str() {
            TYPE_URL_CDS => self.handle_cds_request(request),
            TYPE_URL_EDS => self.handle_eds_request(request),
            TYPE_URL_LDS => self.handle_lds_request(request),
            TYPE_URL_RDS => self.handle_rds_request(request),
            _ => {
                warn!("Unknown resource type: {}", request.type_url);
                DiscoveryResponse {
                    version_info: String::new(),
                    resources: vec![],
                    type_url: request.type_url.clone(),
                    nonce: String::new(),
                }
            }
        }
    }

    /// Register a node subscription.
    pub fn register_subscription(
        &self,
        node: Node,
        resource_types: Vec<String>,
    ) {
        let node_id = node.id.clone();
        self.subscriptions.insert(node_id.clone(), Subscription {
            node,
            resource_types,
            versions: HashMap::new(),
        });
        debug!("Registered subscription for node '{}'", node_id);
    }

    /// Unregister a node subscription.
    pub fn unregister_subscription(&self, node_id: &str) {
        if self.subscriptions.remove(node_id).is_some() {
            debug!("Unregistered subscription for node '{}'", node_id);
        }
    }

    /// Get statistics about the server.
    pub fn stats(&self) -> XdsServerStats {
        XdsServerStats {
            cluster_count: self.clusters.len(),
            endpoint_count: self.endpoints.len(),
            listener_count: self.listeners.len(),
            route_count: self.routes.len(),
            subscription_count: self.subscriptions.len(),
            version: self.version_counter.load(Ordering::SeqCst),
        }
    }
}

/// xDS server statistics.
#[derive(Debug, Clone)]
pub struct XdsServerStats {
    pub cluster_count: usize,
    pub endpoint_count: usize,
    pub listener_count: usize,
    pub route_count: usize,
    pub subscription_count: usize,
    pub version: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_xds_server_creation() {
        let addr = "127.0.0.1:18000".parse().unwrap();
        let server = XdsServer::new(addr);
        assert_eq!(server.address(), addr);
    }

    #[test]
    fn test_cluster_operations() {
        let addr = "127.0.0.1:18001".parse().unwrap();
        let server = XdsServer::new(addr);

        let cluster = Cluster {
            name: "test-cluster".to_string(),
            connect_timeout: Duration::from_secs(5),
            cluster_type: ClusterType::Eds,
            lb_policy: LbPolicy::RoundRobin,
            health_checks: vec![],
            circuit_breakers: None,
            tls_context: None,
        };

        server.set_cluster(cluster.clone()).unwrap();

        let retrieved = server.get_cluster("test-cluster").unwrap();
        assert_eq!(retrieved.name, "test-cluster");

        server.remove_cluster("test-cluster").unwrap();
        assert!(server.get_cluster("test-cluster").is_none());
    }

    #[test]
    fn test_endpoint_operations() {
        let addr = "127.0.0.1:18002".parse().unwrap();
        let server = XdsServer::new(addr);

        let assignment = ClusterLoadAssignment {
            cluster_name: "test-cluster".to_string(),
            endpoints: vec![
                LocalityLbEndpoints {
                    locality: None,
                    lb_endpoints: vec![
                        LbEndpoint {
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
                        },
                    ],
                    load_balancing_weight: None,
                    priority: 0,
                },
            ],
            policy: None,
        };

        server.set_endpoints(assignment).unwrap();

        let retrieved = server.get_endpoints("test-cluster").unwrap();
        assert_eq!(retrieved.cluster_name, "test-cluster");
        assert_eq!(retrieved.endpoints.len(), 1);
    }

    #[test]
    fn test_discovery_request_handling() {
        let addr = "127.0.0.1:18003".parse().unwrap();
        let server = XdsServer::new(addr);

        // Add a cluster
        let cluster = Cluster {
            name: "service-a".to_string(),
            ..Default::default()
        };
        server.set_cluster(cluster).unwrap();

        // Request all clusters
        let request = DiscoveryRequest {
            version_info: String::new(),
            node: Node::default(),
            resource_names: vec![],
            type_url: TYPE_URL_CDS.to_string(),
            response_nonce: String::new(),
            error_detail: None,
        };

        let response = server.handle_discovery_request(&request);
        assert_eq!(response.type_url, TYPE_URL_CDS);
        assert_eq!(response.resources.len(), 1);
    }

    #[test]
    fn test_version_increment() {
        let addr = "127.0.0.1:18004".parse().unwrap();
        let server = XdsServer::new(addr);

        let v1 = server.next_version();
        let v2 = server.next_version();

        assert_ne!(v1, v2);
        assert!(v2 > v1);
    }

    #[test]
    fn test_stats() {
        let addr = "127.0.0.1:18005".parse().unwrap();
        let server = XdsServer::new(addr);

        server.set_cluster(Cluster {
            name: "c1".to_string(),
            ..Default::default()
        }).unwrap();

        server.set_cluster(Cluster {
            name: "c2".to_string(),
            ..Default::default()
        }).unwrap();

        let stats = server.stats();
        assert_eq!(stats.cluster_count, 2);
    }
}
