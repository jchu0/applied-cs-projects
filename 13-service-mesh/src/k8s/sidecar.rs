//! Sidecar injection logic.

use super::types::*;
use crate::{Error, Result};

use serde_json::json;
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Configuration for sidecar injection.
#[derive(Debug, Clone)]
pub struct InjectionConfig {
    /// Sidecar image.
    pub sidecar_image: String,
    /// Sidecar image pull policy.
    pub image_pull_policy: String,
    /// Init container image.
    pub init_image: String,
    /// Control plane address.
    pub control_plane_address: String,
    /// Inbound port.
    pub inbound_port: i32,
    /// Outbound port.
    pub outbound_port: i32,
    /// Admin port.
    pub admin_port: i32,
    /// Enable mTLS.
    pub enable_mtls: bool,
    /// CPU request.
    pub cpu_request: String,
    /// CPU limit.
    pub cpu_limit: String,
    /// Memory request.
    pub memory_request: String,
    /// Memory limit.
    pub memory_limit: String,
    /// Excluded inbound ports.
    pub exclude_inbound_ports: Vec<i32>,
    /// Excluded outbound ports.
    pub exclude_outbound_ports: Vec<i32>,
    /// Service account volume name.
    pub service_account_volume: String,
}

impl Default for InjectionConfig {
    fn default() -> Self {
        Self {
            sidecar_image: "service-mesh/proxy:latest".to_string(),
            image_pull_policy: "IfNotPresent".to_string(),
            init_image: "service-mesh/iptables-init:latest".to_string(),
            control_plane_address: "istiod.istio-system.svc:15012".to_string(),
            inbound_port: 15006,
            outbound_port: 15001,
            admin_port: 15000,
            enable_mtls: true,
            cpu_request: "10m".to_string(),
            cpu_limit: "2000m".to_string(),
            memory_request: "40Mi".to_string(),
            memory_limit: "1024Mi".to_string(),
            exclude_inbound_ports: vec![15020, 15021, 15090],
            exclude_outbound_ports: vec![],
            service_account_volume: "service-account-token".to_string(),
        }
    }
}

/// Sidecar injector for Kubernetes pods.
pub struct SidecarInjector {
    /// Injection configuration.
    config: InjectionConfig,
    /// Injection enabled annotation key.
    injection_annotation: String,
    /// Injected annotation key.
    injected_annotation: String,
}

impl SidecarInjector {
    /// Create a new sidecar injector.
    pub fn new(config: InjectionConfig) -> Self {
        Self {
            config,
            injection_annotation: "sidecar.mesh.io/inject".to_string(),
            injected_annotation: "sidecar.mesh.io/injected".to_string(),
        }
    }

    /// Check if a pod should be injected.
    pub fn should_inject(&self, pod: &Pod) -> bool {
        // Check if already injected
        if let Some(value) = pod.metadata.annotations.get(&self.injected_annotation) {
            if value == "true" {
                debug!("Pod already injected, skipping");
                return false;
            }
        }

        // Check for explicit injection annotation
        if let Some(value) = pod.metadata.annotations.get(&self.injection_annotation) {
            return value == "true";
        }

        // Check namespace label (would need namespace info)
        // Default to inject if no annotation present
        true
    }

    /// Generate JSON patch for sidecar injection.
    pub fn generate_patch(&self, pod: &Pod) -> Result<Vec<JsonPatchOp>> {
        let mut patches = Vec::new();

        // Add init container for iptables setup
        let init_container = self.create_init_container(pod);
        patches.push(self.add_init_container_patch(pod, init_container)?);

        // Add sidecar container
        let sidecar = self.create_sidecar_container(pod);
        patches.push(self.add_container_patch(sidecar)?);

        // Add volumes
        for volume_patch in self.add_volumes_patch(pod)? {
            patches.push(volume_patch);
        }

        // Add injected annotation
        patches.push(self.add_annotation_patch(
            &self.injected_annotation,
            "true",
            &pod.metadata.annotations,
        )?);

        Ok(patches)
    }

    /// Create the init container for iptables rules.
    fn create_init_container(&self, _pod: &Pod) -> Container {
        let exclude_inbound: String = self.config.exclude_inbound_ports
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(",");

        let exclude_outbound: String = self.config.exclude_outbound_ports
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(",");

        Container {
            name: "mesh-init".to_string(),
            image: self.config.init_image.clone(),
            image_pull_policy: Some(self.config.image_pull_policy.clone()),
            args: vec![
                "-p".to_string(), self.config.inbound_port.to_string(),
                "-z".to_string(), self.config.outbound_port.to_string(),
                "-u".to_string(), "1337".to_string(), // Proxy UID
                "-m".to_string(), "REDIRECT".to_string(),
                "-i".to_string(), "*".to_string(), // Include all IPs
                "-x".to_string(), "".to_string(), // Exclude no IPs
                "-b".to_string(), "*".to_string(), // Intercept all inbound
                "-d".to_string(), exclude_inbound,
            ],
            security_context: Some(SecurityContext {
                run_as_user: Some(0),
                run_as_group: Some(0),
                run_as_non_root: Some(false),
                privileged: Some(false),
                capabilities: Some(Capabilities {
                    add: vec!["NET_ADMIN".to_string(), "NET_RAW".to_string()],
                    drop: vec!["ALL".to_string()],
                }),
            }),
            resources: Some(ResourceRequirements {
                limits: [
                    ("cpu".to_string(), "2000m".to_string()),
                    ("memory".to_string(), "128Mi".to_string()),
                ].into_iter().collect(),
                requests: [
                    ("cpu".to_string(), "10m".to_string()),
                    ("memory".to_string(), "10Mi".to_string()),
                ].into_iter().collect(),
            }),
            ..Default::default()
        }
    }

    /// Create the sidecar proxy container.
    fn create_sidecar_container(&self, pod: &Pod) -> Container {
        let service_name = pod.metadata.name.clone().unwrap_or_default();
        let namespace = pod.metadata.namespace.clone().unwrap_or_else(|| "default".to_string());

        Container {
            name: "mesh-proxy".to_string(),
            image: self.config.sidecar_image.clone(),
            image_pull_policy: Some(self.config.image_pull_policy.clone()),
            args: vec![
                "proxy".to_string(),
                "--inbound-port".to_string(), self.config.inbound_port.to_string(),
                "--outbound-port".to_string(), self.config.outbound_port.to_string(),
                "--admin-port".to_string(), self.config.admin_port.to_string(),
                "--control-plane".to_string(), self.config.control_plane_address.clone(),
            ],
            env: vec![
                EnvVar {
                    name: "POD_NAME".to_string(),
                    value: None,
                    value_from: Some(EnvVarSource {
                        field_ref: Some(ObjectFieldSelector {
                            field_path: "metadata.name".to_string(),
                        }),
                        ..Default::default()
                    }),
                },
                EnvVar {
                    name: "POD_NAMESPACE".to_string(),
                    value: None,
                    value_from: Some(EnvVarSource {
                        field_ref: Some(ObjectFieldSelector {
                            field_path: "metadata.namespace".to_string(),
                        }),
                        ..Default::default()
                    }),
                },
                EnvVar {
                    name: "POD_IP".to_string(),
                    value: None,
                    value_from: Some(EnvVarSource {
                        field_ref: Some(ObjectFieldSelector {
                            field_path: "status.podIP".to_string(),
                        }),
                        ..Default::default()
                    }),
                },
                EnvVar {
                    name: "SERVICE_NAME".to_string(),
                    value: Some(service_name),
                    value_from: None,
                },
                EnvVar {
                    name: "MESH_MTLS_ENABLED".to_string(),
                    value: Some(self.config.enable_mtls.to_string()),
                    value_from: None,
                },
            ],
            ports: vec![
                ContainerPort {
                    name: Some("http-envoy-prom".to_string()),
                    container_port: 15090,
                    protocol: Some("TCP".to_string()),
                },
            ],
            volume_mounts: vec![
                VolumeMount {
                    name: "mesh-certs".to_string(),
                    mount_path: "/etc/certs".to_string(),
                    read_only: true,
                    sub_path: None,
                },
                VolumeMount {
                    name: "mesh-config".to_string(),
                    mount_path: "/etc/mesh/config".to_string(),
                    read_only: true,
                    sub_path: None,
                },
            ],
            resources: Some(ResourceRequirements {
                limits: [
                    ("cpu".to_string(), self.config.cpu_limit.clone()),
                    ("memory".to_string(), self.config.memory_limit.clone()),
                ].into_iter().collect(),
                requests: [
                    ("cpu".to_string(), self.config.cpu_request.clone()),
                    ("memory".to_string(), self.config.memory_request.clone()),
                ].into_iter().collect(),
            }),
            security_context: Some(SecurityContext {
                run_as_user: Some(1337),
                run_as_group: Some(1337),
                run_as_non_root: Some(true),
                privileged: Some(false),
                capabilities: Some(Capabilities {
                    add: vec![],
                    drop: vec!["ALL".to_string()],
                }),
            }),
            readiness_probe: Some(Probe {
                http_get: Some(HTTPGetAction {
                    path: Some("/ready".to_string()),
                    port: self.config.admin_port,
                    scheme: Some("HTTP".to_string()),
                }),
                initial_delay_seconds: Some(1),
                period_seconds: Some(2),
                timeout_seconds: Some(3),
                failure_threshold: Some(30),
                success_threshold: None,
                tcp_socket: None,
                exec: None,
            }),
            liveness_probe: Some(Probe {
                http_get: Some(HTTPGetAction {
                    path: Some("/health".to_string()),
                    port: self.config.admin_port,
                    scheme: Some("HTTP".to_string()),
                }),
                initial_delay_seconds: Some(10),
                period_seconds: Some(10),
                timeout_seconds: Some(3),
                failure_threshold: Some(3),
                success_threshold: None,
                tcp_socket: None,
                exec: None,
            }),
            ..Default::default()
        }
    }

    /// Create JSON patch to add init container.
    fn add_init_container_patch(&self, pod: &Pod, container: Container) -> Result<JsonPatchOp> {
        if pod.spec.init_containers.is_empty() {
            Ok(JsonPatchOp {
                op: "add".to_string(),
                path: "/spec/initContainers".to_string(),
                value: Some(json!([container])),
            })
        } else {
            Ok(JsonPatchOp {
                op: "add".to_string(),
                path: "/spec/initContainers/0".to_string(),
                value: Some(serde_json::to_value(&container)?),
            })
        }
    }

    /// Create JSON patch to add container.
    fn add_container_patch(&self, container: Container) -> Result<JsonPatchOp> {
        Ok(JsonPatchOp {
            op: "add".to_string(),
            path: "/spec/containers/-".to_string(),
            value: Some(serde_json::to_value(&container)?),
        })
    }

    /// Create JSON patch to add volumes.
    fn add_volumes_patch(&self, pod: &Pod) -> Result<Vec<JsonPatchOp>> {
        let mut patches = Vec::new();

        let volumes = vec![
            Volume {
                name: "mesh-certs".to_string(),
                empty_dir: Some(EmptyDirVolumeSource {
                    medium: Some("Memory".to_string()),
                    size_limit: None,
                }),
                ..Default::default()
            },
            Volume {
                name: "mesh-config".to_string(),
                config_map: Some(ConfigMapVolumeSource {
                    name: "mesh-config".to_string(),
                    optional: true,
                }),
                ..Default::default()
            },
        ];

        if pod.spec.volumes.is_empty() {
            patches.push(JsonPatchOp {
                op: "add".to_string(),
                path: "/spec/volumes".to_string(),
                value: Some(serde_json::to_value(&volumes)?),
            });
        } else {
            for volume in volumes {
                patches.push(JsonPatchOp {
                    op: "add".to_string(),
                    path: "/spec/volumes/-".to_string(),
                    value: Some(serde_json::to_value(&volume)?),
                });
            }
        }

        Ok(patches)
    }

    /// Create JSON patch to add annotation.
    fn add_annotation_patch(
        &self,
        key: &str,
        value: &str,
        existing: &HashMap<String, String>,
    ) -> Result<JsonPatchOp> {
        if existing.is_empty() {
            Ok(JsonPatchOp {
                op: "add".to_string(),
                path: "/metadata/annotations".to_string(),
                value: Some(json!({ key: value })),
            })
        } else {
            // Escape slashes in annotation key for JSON Pointer
            let escaped_key = key.replace("/", "~1");
            Ok(JsonPatchOp {
                op: "add".to_string(),
                path: format!("/metadata/annotations/{}", escaped_key),
                value: Some(json!(value)),
            })
        }
    }

    /// Get the injection configuration.
    pub fn config(&self) -> &InjectionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pod() -> Pod {
        Pod {
            api_version: Some("v1".to_string()),
            kind: Some("Pod".to_string()),
            metadata: ObjectMeta {
                name: Some("test-pod".to_string()),
                namespace: Some("default".to_string()),
                ..Default::default()
            },
            spec: PodSpec {
                containers: vec![Container {
                    name: "app".to_string(),
                    image: "myapp:latest".to_string(),
                    ports: vec![ContainerPort {
                        name: Some("http".to_string()),
                        container_port: 8080,
                        protocol: Some("TCP".to_string()),
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            },
        }
    }

    #[test]
    fn test_should_inject_default() {
        let config = InjectionConfig::default();
        let injector = SidecarInjector::new(config);
        let pod = test_pod();

        assert!(injector.should_inject(&pod));
    }

    #[test]
    fn test_should_not_inject_already_injected() {
        let config = InjectionConfig::default();
        let injector = SidecarInjector::new(config);

        let mut pod = test_pod();
        pod.metadata.annotations.insert(
            "sidecar.mesh.io/injected".to_string(),
            "true".to_string(),
        );

        assert!(!injector.should_inject(&pod));
    }

    #[test]
    fn test_should_inject_explicit_annotation() {
        let config = InjectionConfig::default();
        let injector = SidecarInjector::new(config);

        let mut pod = test_pod();
        pod.metadata.annotations.insert(
            "sidecar.mesh.io/inject".to_string(),
            "true".to_string(),
        );

        assert!(injector.should_inject(&pod));
    }

    #[test]
    fn test_should_not_inject_explicit_false() {
        let config = InjectionConfig::default();
        let injector = SidecarInjector::new(config);

        let mut pod = test_pod();
        pod.metadata.annotations.insert(
            "sidecar.mesh.io/inject".to_string(),
            "false".to_string(),
        );

        assert!(!injector.should_inject(&pod));
    }

    #[test]
    fn test_generate_patch() {
        let config = InjectionConfig::default();
        let injector = SidecarInjector::new(config);
        let pod = test_pod();

        let patches = injector.generate_patch(&pod).unwrap();

        // Should have patches for:
        // 1. Init container
        // 2. Sidecar container
        // 3. Volumes (at least 2)
        // 4. Annotation
        assert!(patches.len() >= 4);

        // Check init container patch
        let init_patch = &patches[0];
        assert!(init_patch.path.contains("initContainers"));

        // Check sidecar patch
        let sidecar_patch = &patches[1];
        assert_eq!(sidecar_patch.path, "/spec/containers/-");
    }

    #[test]
    fn test_create_init_container() {
        let config = InjectionConfig::default();
        let injector = SidecarInjector::new(config);
        let pod = test_pod();

        let container = injector.create_init_container(&pod);

        assert_eq!(container.name, "mesh-init");
        assert!(container.security_context.is_some());

        let sec_ctx = container.security_context.unwrap();
        assert!(sec_ctx.capabilities.is_some());

        let caps = sec_ctx.capabilities.unwrap();
        assert!(caps.add.contains(&"NET_ADMIN".to_string()));
    }

    #[test]
    fn test_create_sidecar_container() {
        let config = InjectionConfig::default();
        let injector = SidecarInjector::new(config);
        let pod = test_pod();

        let container = injector.create_sidecar_container(&pod);

        assert_eq!(container.name, "mesh-proxy");
        assert!(!container.env.is_empty());
        assert!(!container.volume_mounts.is_empty());
        assert!(container.readiness_probe.is_some());
        assert!(container.liveness_probe.is_some());
    }

    #[test]
    fn test_injection_config_default() {
        let config = InjectionConfig::default();

        assert_eq!(config.inbound_port, 15006);
        assert_eq!(config.outbound_port, 15001);
        assert_eq!(config.admin_port, 15000);
        assert!(config.enable_mtls);
    }
}
