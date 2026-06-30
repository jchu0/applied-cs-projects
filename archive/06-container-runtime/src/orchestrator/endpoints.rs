//! Endpoints and EndpointSlices
//!
//! Tracks the IP addresses of pods backing a service.

use super::{ObjectMeta, ResourceId, Pod, Service, LabelSelector};
use super::service::Protocol;
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::net::IpAddr;
use std::time::Instant;

/// Endpoints for a service
#[derive(Clone, Debug)]
pub struct Endpoints {
    pub metadata: ObjectMeta,
    pub subsets: Vec<EndpointSubset>,
}

impl Endpoints {
    pub fn new(name: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            metadata: ObjectMeta::new(name, namespace),
            subsets: Vec::new(),
        }
    }

    /// Create endpoints from a service and matching pods
    pub fn from_service_and_pods(service: &Service, pods: &[&Pod]) -> Self {
        let mut endpoints = Self::new(
            service.metadata.name.clone(),
            service.metadata.namespace.clone(),
        );

        let ready_pods: Vec<_> = pods.iter()
            .filter(|p| p.is_ready())
            .copied()
            .collect();

        let not_ready_pods: Vec<_> = pods.iter()
            .filter(|p| !p.is_ready())
            .copied()
            .collect();

        let ports: Vec<EndpointPort> = service.spec.ports.iter()
            .map(|sp| EndpointPort {
                name: sp.name.clone(),
                port: match &sp.target_port {
                    super::service::TargetPort::Number(n) => *n,
                    super::service::TargetPort::Name(_) => sp.port, // Would need to resolve by name
                },
                protocol: sp.protocol,
                app_protocol: sp.app_protocol.clone(),
            })
            .collect();

        if !ready_pods.is_empty() || !not_ready_pods.is_empty() {
            let subset = EndpointSubset {
                addresses: ready_pods.iter()
                    .filter_map(|p| {
                        p.status.pod_ip.as_ref().map(|ip| EndpointAddress {
                            ip: ip.parse().ok()?,
                            hostname: Some(p.metadata.name.clone()),
                            node_name: p.status.node_name.clone(),
                            target_ref: Some(ObjectReference {
                                kind: "Pod".to_string(),
                                namespace: p.metadata.namespace.clone(),
                                name: p.metadata.name.clone(),
                                uid: p.metadata.uid.clone(),
                            }),
                        })
                    })
                    .collect(),
                not_ready_addresses: not_ready_pods.iter()
                    .filter_map(|p| {
                        p.status.pod_ip.as_ref().map(|ip| EndpointAddress {
                            ip: ip.parse().ok()?,
                            hostname: Some(p.metadata.name.clone()),
                            node_name: p.status.node_name.clone(),
                            target_ref: Some(ObjectReference {
                                kind: "Pod".to_string(),
                                namespace: p.metadata.namespace.clone(),
                                name: p.metadata.name.clone(),
                                uid: p.metadata.uid.clone(),
                            }),
                        })
                    })
                    .collect(),
                ports: ports.clone(),
            };

            endpoints.subsets.push(subset);
        }

        endpoints
    }

    /// Get all ready addresses
    pub fn get_ready_addresses(&self) -> Vec<&EndpointAddress> {
        self.subsets.iter()
            .flat_map(|s| s.addresses.iter())
            .collect()
    }

    /// Get all not ready addresses
    pub fn get_not_ready_addresses(&self) -> Vec<&EndpointAddress> {
        self.subsets.iter()
            .flat_map(|s| s.not_ready_addresses.iter())
            .collect()
    }

    /// Get addresses count
    pub fn ready_count(&self) -> usize {
        self.subsets.iter()
            .map(|s| s.addresses.len())
            .sum()
    }

    /// Check if endpoints have any ready addresses
    pub fn has_ready_addresses(&self) -> bool {
        self.subsets.iter().any(|s| !s.addresses.is_empty())
    }
}

/// Endpoint subset
#[derive(Clone, Debug)]
pub struct EndpointSubset {
    pub addresses: Vec<EndpointAddress>,
    pub not_ready_addresses: Vec<EndpointAddress>,
    pub ports: Vec<EndpointPort>,
}

/// Endpoint address
#[derive(Clone, Debug)]
pub struct EndpointAddress {
    pub ip: IpAddr,
    pub hostname: Option<String>,
    pub node_name: Option<String>,
    pub target_ref: Option<ObjectReference>,
}

/// Endpoint port
#[derive(Clone, Debug)]
pub struct EndpointPort {
    pub name: String,
    pub port: u16,
    pub protocol: Protocol,
    pub app_protocol: Option<String>,
}

/// Object reference
#[derive(Clone, Debug)]
pub struct ObjectReference {
    pub kind: String,
    pub namespace: String,
    pub name: String,
    pub uid: ResourceId,
}

/// EndpointSlice for scalable endpoint tracking
#[derive(Clone, Debug)]
pub struct EndpointSlice {
    pub metadata: ObjectMeta,
    pub address_type: AddressType,
    pub endpoints: Vec<Endpoint>,
    pub ports: Vec<EndpointSlicePort>,
}

impl EndpointSlice {
    pub fn new(name: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            metadata: ObjectMeta::new(name, namespace),
            address_type: AddressType::IPv4,
            endpoints: Vec::new(),
            ports: Vec::new(),
        }
    }

    /// Get ready endpoints
    pub fn get_ready(&self) -> Vec<&Endpoint> {
        self.endpoints.iter()
            .filter(|e| e.conditions.ready.unwrap_or(false))
            .collect()
    }

    /// Get serving endpoints
    pub fn get_serving(&self) -> Vec<&Endpoint> {
        self.endpoints.iter()
            .filter(|e| e.conditions.serving.unwrap_or(false))
            .collect()
    }

    /// Get terminating endpoints
    pub fn get_terminating(&self) -> Vec<&Endpoint> {
        self.endpoints.iter()
            .filter(|e| e.conditions.terminating.unwrap_or(false))
            .collect()
    }
}

/// Address type
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AddressType {
    #[default]
    IPv4,
    IPv6,
    FQDN,
}

/// Endpoint in an EndpointSlice
#[derive(Clone, Debug)]
pub struct Endpoint {
    pub addresses: Vec<String>,
    pub conditions: EndpointConditions,
    pub hostname: Option<String>,
    pub target_ref: Option<ObjectReference>,
    pub deprecate_topology: Option<HashMap<String, String>>,
    pub node_name: Option<String>,
    pub zone: Option<String>,
    pub hints: Option<EndpointHints>,
}

/// Endpoint conditions
#[derive(Clone, Debug, Default)]
pub struct EndpointConditions {
    pub ready: Option<bool>,
    pub serving: Option<bool>,
    pub terminating: Option<bool>,
}

/// Endpoint hints for topology-aware routing
#[derive(Clone, Debug)]
pub struct EndpointHints {
    pub for_zones: Vec<ForZone>,
}

/// Zone hint
#[derive(Clone, Debug)]
pub struct ForZone {
    pub name: String,
}

/// EndpointSlice port
#[derive(Clone, Debug)]
pub struct EndpointSlicePort {
    pub name: Option<String>,
    pub protocol: Protocol,
    pub port: Option<u16>,
    pub app_protocol: Option<String>,
}

/// Endpoints controller
#[derive(Debug)]
pub struct EndpointsController {
    endpoints: HashMap<String, Endpoints>,
    endpoint_slices: HashMap<String, Vec<EndpointSlice>>,
}

impl EndpointsController {
    pub fn new() -> Self {
        Self {
            endpoints: HashMap::new(),
            endpoint_slices: HashMap::new(),
        }
    }

    /// Sync endpoints for a service
    pub fn sync_endpoints(&mut self, service: &Service, pods: &[&Pod]) -> Result<()> {
        let key = self.service_key(service);

        let endpoints = Endpoints::from_service_and_pods(service, pods);
        self.endpoints.insert(key.clone(), endpoints);

        // Also create EndpointSlices
        self.sync_endpoint_slices(service, pods)?;

        Ok(())
    }

    /// Sync endpoint slices
    fn sync_endpoint_slices(&mut self, service: &Service, pods: &[&Pod]) -> Result<()> {
        let key = self.service_key(service);

        let ports: Vec<EndpointSlicePort> = service.spec.ports.iter()
            .map(|p| EndpointSlicePort {
                name: Some(p.name.clone()),
                protocol: p.protocol,
                port: Some(p.port),
                app_protocol: p.app_protocol.clone(),
            })
            .collect();

        // Group pods into slices (max 100 endpoints per slice)
        let max_per_slice = 100;
        let mut slices = Vec::new();

        for chunk in pods.chunks(max_per_slice) {
            let mut slice = EndpointSlice::new(
                format!("{}-{}", service.metadata.name, slices.len()),
                service.metadata.namespace.clone(),
            );

            slice.ports = ports.clone();

            for pod in chunk {
                if let Some(ip) = &pod.status.pod_ip {
                    slice.endpoints.push(Endpoint {
                        addresses: vec![ip.clone()],
                        conditions: EndpointConditions {
                            ready: Some(pod.is_ready()),
                            serving: Some(pod.is_running()),
                            terminating: Some(pod.is_terminated()),
                        },
                        hostname: Some(pod.metadata.name.clone()),
                        target_ref: Some(ObjectReference {
                            kind: "Pod".to_string(),
                            namespace: pod.metadata.namespace.clone(),
                            name: pod.metadata.name.clone(),
                            uid: pod.metadata.uid.clone(),
                        }),
                        deprecate_topology: None,
                        node_name: pod.status.node_name.clone(),
                        zone: None,
                        hints: None,
                    });
                }
            }

            slices.push(slice);
        }

        self.endpoint_slices.insert(key, slices);
        Ok(())
    }

    /// Get endpoints for a service
    pub fn get_endpoints(&self, namespace: &str, name: &str) -> Option<&Endpoints> {
        let key = format!("{}/{}", namespace, name);
        self.endpoints.get(&key)
    }

    /// Get endpoint slices for a service
    pub fn get_endpoint_slices(&self, namespace: &str, name: &str) -> Vec<&EndpointSlice> {
        let key = format!("{}/{}", namespace, name);
        self.endpoint_slices.get(&key)
            .map(|slices| slices.iter().collect())
            .unwrap_or_default()
    }

    /// Remove endpoints for a service
    pub fn remove_endpoints(&mut self, namespace: &str, name: &str) {
        let key = format!("{}/{}", namespace, name);
        self.endpoints.remove(&key);
        self.endpoint_slices.remove(&key);
    }

    fn service_key(&self, service: &Service) -> String {
        format!("{}/{}", service.metadata.namespace, service.metadata.name)
    }
}

impl Default for EndpointsController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestrator::service::ServicePort;

    fn create_test_pod(name: &str, ip: &str, ready: bool) -> Pod {
        let mut pod = Pod::new(name, "default");
        pod.status.pod_ip = Some(ip.to_string());
        if ready {
            pod.status.phase = super::super::pod::PodPhase::Running;
            pod.status.conditions.push(super::super::pod::PodCondition {
                condition_type: super::super::pod::PodConditionType::Ready,
                status: true,
                last_probe_time: None,
                last_transition_time: Instant::now(),
                reason: "".into(),
                message: "".into(),
            });
        }
        pod
    }

    fn create_test_service(name: &str) -> Service {
        Service::new(name, "default")
            .with_port(ServicePort::tcp("http", 80, 8080))
            .with_selector("app", "web")
    }

    #[test]
    fn test_endpoints_new() {
        let ep = Endpoints::new("web", "default");
        assert_eq!(ep.metadata.name, "web");
        assert!(ep.subsets.is_empty());
    }

    #[test]
    fn test_endpoints_from_service_and_pods() {
        let service = create_test_service("web");

        let pods: Vec<Pod> = vec![
            create_test_pod("pod-1", "10.0.0.1", true),
            create_test_pod("pod-2", "10.0.0.2", true),
            create_test_pod("pod-3", "10.0.0.3", false),
        ];

        let pod_refs: Vec<&Pod> = pods.iter().collect();
        let endpoints = Endpoints::from_service_and_pods(&service, &pod_refs);

        assert_eq!(endpoints.ready_count(), 2);
        assert_eq!(endpoints.get_not_ready_addresses().len(), 1);
    }

    #[test]
    fn test_endpoints_has_ready() {
        let service = create_test_service("web");
        let pods: Vec<Pod> = vec![
            create_test_pod("pod-1", "10.0.0.1", true),
        ];

        let pod_refs: Vec<&Pod> = pods.iter().collect();
        let endpoints = Endpoints::from_service_and_pods(&service, &pod_refs);

        assert!(endpoints.has_ready_addresses());
    }

    #[test]
    fn test_endpoint_slice_new() {
        let slice = EndpointSlice::new("web-1", "default");
        assert_eq!(slice.metadata.name, "web-1");
        assert!(slice.endpoints.is_empty());
    }

    #[test]
    fn test_endpoint_slice_ready() {
        let mut slice = EndpointSlice::new("web-1", "default");

        slice.endpoints.push(Endpoint {
            addresses: vec!["10.0.0.1".to_string()],
            conditions: EndpointConditions {
                ready: Some(true),
                serving: Some(true),
                terminating: Some(false),
            },
            hostname: None,
            target_ref: None,
            deprecate_topology: None,
            node_name: None,
            zone: None,
            hints: None,
        });

        slice.endpoints.push(Endpoint {
            addresses: vec!["10.0.0.2".to_string()],
            conditions: EndpointConditions {
                ready: Some(false),
                serving: Some(false),
                terminating: Some(true),
            },
            hostname: None,
            target_ref: None,
            deprecate_topology: None,
            node_name: None,
            zone: None,
            hints: None,
        });

        assert_eq!(slice.get_ready().len(), 1);
        assert_eq!(slice.get_terminating().len(), 1);
    }

    #[test]
    fn test_endpoints_controller_sync() {
        let mut controller = EndpointsController::new();

        let service = create_test_service("api");
        let pods: Vec<Pod> = vec![
            create_test_pod("api-1", "10.0.0.1", true),
            create_test_pod("api-2", "10.0.0.2", true),
        ];

        let pod_refs: Vec<&Pod> = pods.iter().collect();
        controller.sync_endpoints(&service, &pod_refs).unwrap();

        let endpoints = controller.get_endpoints("default", "api");
        assert!(endpoints.is_some());
        assert_eq!(endpoints.unwrap().ready_count(), 2);

        let slices = controller.get_endpoint_slices("default", "api");
        assert!(!slices.is_empty());
    }

    #[test]
    fn test_endpoints_controller_remove() {
        let mut controller = EndpointsController::new();

        let service = create_test_service("web");
        let pods: Vec<Pod> = vec![create_test_pod("web-1", "10.0.0.1", true)];

        let pod_refs: Vec<&Pod> = pods.iter().collect();
        controller.sync_endpoints(&service, &pod_refs).unwrap();

        assert!(controller.get_endpoints("default", "web").is_some());

        controller.remove_endpoints("default", "web");
        assert!(controller.get_endpoints("default", "web").is_none());
    }

    #[test]
    fn test_endpoint_address() {
        let addr = EndpointAddress {
            ip: "10.0.0.1".parse().unwrap(),
            hostname: Some("pod-1".to_string()),
            node_name: Some("node-1".to_string()),
            target_ref: None,
        };

        assert_eq!(addr.ip.to_string(), "10.0.0.1");
    }

    #[test]
    fn test_address_type() {
        assert_eq!(AddressType::default(), AddressType::IPv4);
    }

    #[test]
    fn test_endpoint_conditions() {
        let conditions = EndpointConditions {
            ready: Some(true),
            serving: Some(true),
            terminating: Some(false),
        };

        assert!(conditions.ready.unwrap());
        assert!(conditions.serving.unwrap());
        assert!(!conditions.terminating.unwrap());
    }

    #[test]
    fn test_large_endpoint_slicing() {
        let mut controller = EndpointsController::new();

        let service = create_test_service("large");

        // Create 150 pods to test slicing
        let pods: Vec<Pod> = (0..150)
            .map(|i| create_test_pod(&format!("pod-{}", i), &format!("10.0.0.{}", i % 256), true))
            .collect();

        let pod_refs: Vec<&Pod> = pods.iter().collect();
        controller.sync_endpoints(&service, &pod_refs).unwrap();

        let slices = controller.get_endpoint_slices("default", "large");
        assert!(slices.len() >= 2); // Should be split into multiple slices
    }
}
