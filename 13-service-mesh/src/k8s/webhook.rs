//! Kubernetes mutating admission webhook server.

use super::sidecar::{InjectionConfig, SidecarInjector};
use super::types::*;
use crate::{Error, Result};

use bytes::Bytes;
use hyper::server::conn::Http;
use hyper::service::service_fn;
use hyper::{Body, Method, Request, Response, StatusCode};
use rustls::{Certificate, PrivateKey, ServerConfig};
use rustls::server::AllowAnyAuthenticatedClient;
use std::fs;
use std::io::BufReader;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;
use tracing::{debug, error, info, warn};

/// Webhook server for sidecar injection.
pub struct WebhookServer {
    /// Server address.
    address: SocketAddr,
    /// Sidecar injector.
    injector: Arc<SidecarInjector>,
    /// TLS certificate path.
    cert_path: Option<String>,
    /// TLS key path.
    key_path: Option<String>,
}

impl WebhookServer {
    /// Create a new webhook server.
    pub fn new(address: SocketAddr, config: InjectionConfig) -> Self {
        Self {
            address,
            injector: Arc::new(SidecarInjector::new(config)),
            cert_path: None,
            key_path: None,
        }
    }

    /// Set TLS configuration.
    pub fn with_tls(mut self, cert_path: String, key_path: String) -> Self {
        self.cert_path = Some(cert_path);
        self.key_path = Some(key_path);
        self
    }

    /// Get the server address.
    pub fn address(&self) -> SocketAddr {
        self.address
    }

    /// Handle an admission review request.
    pub fn handle_admission_review(&self, review: AdmissionReview) -> AdmissionReview {
        let request = match review.request {
            Some(req) => req,
            None => {
                error!("Admission review missing request");
                return AdmissionReview::reject("unknown".to_string(), "missing request");
            }
        };

        let uid = request.uid.clone();

        // Only handle Pod creation
        if request.kind.kind != "Pod" || request.operation != "CREATE" {
            debug!("Skipping non-Pod or non-CREATE request");
            return AdmissionReview::response(uid, true);
        }

        // Parse the pod object
        let pod: Pod = match request.object {
            Some(obj) => match serde_json::from_value(obj) {
                Ok(p) => p,
                Err(e) => {
                    error!("Failed to parse pod: {}", e);
                    return AdmissionReview::response(uid, true);
                }
            },
            None => {
                error!("Missing object in request");
                return AdmissionReview::response(uid, true);
            }
        };

        let pod_name = pod.metadata.name.clone().unwrap_or_else(|| "unknown".to_string());
        let namespace = pod.metadata.namespace.clone().unwrap_or_else(|| "default".to_string());

        info!("Processing pod {}/{}", namespace, pod_name);

        // Check if we should inject
        if !self.injector.should_inject(&pod) {
            debug!("Skipping injection for pod {}/{}", namespace, pod_name);
            return AdmissionReview::response(uid, true);
        }

        // Generate the patch
        match self.injector.generate_patch(&pod) {
            Ok(patches) => {
                info!(
                    "Injecting sidecar into pod {}/{} with {} patches",
                    namespace,
                    pod_name,
                    patches.len()
                );
                AdmissionReview::response_with_patch(uid, patches)
            }
            Err(e) => {
                error!("Failed to generate patch for pod {}/{}: {}", namespace, pod_name, e);
                // Allow the pod but log the error
                AdmissionReview::response(uid, true)
            }
        }
    }

    /// Handle HTTP request body and return response body.
    pub fn handle_request_body(&self, body: &[u8]) -> Result<Vec<u8>> {
        let review: AdmissionReview = serde_json::from_slice(body)
            .map_err(|e| Error::Serialization(format!("Failed to parse request: {}", e)))?;

        let response = self.handle_admission_review(review);

        serde_json::to_vec(&response)
            .map_err(|e| Error::Serialization(format!("Failed to serialize response: {}", e)))
    }

    /// Health check endpoint.
    pub fn health_check(&self) -> &'static str {
        "OK"
    }

    /// Ready check endpoint.
    pub fn ready_check(&self) -> &'static str {
        "READY"
    }

    /// Get the injector.
    pub fn injector(&self) -> &SidecarInjector {
        &self.injector
    }

    /// Run the webhook server with optional TLS.
    pub async fn run(self: Arc<Self>) -> Result<()> {
        let listener = TcpListener::bind(self.address).await?;
        info!("Webhook server listening on {}", self.address);

        // Setup TLS if configured
        let tls_acceptor = if let (Some(cert_path), Some(key_path)) = (&self.cert_path, &self.key_path) {
            Some(Self::create_tls_acceptor(cert_path, key_path)?)
        } else {
            None
        };

        loop {
            let (stream, remote_addr) = listener.accept().await?;
            debug!("Connection from {}", remote_addr);

            let server = Arc::clone(&self);
            let acceptor = tls_acceptor.clone();

            tokio::spawn(async move {
                let result = if let Some(acceptor) = acceptor {
                    // TLS connection
                    match acceptor.accept(stream).await {
                        Ok(tls_stream) => {
                            let service = service_fn(|req| {
                                let srv = Arc::clone(&server);
                                async move { srv.handle_http_request(req).await }
                            });
                            Http::new().serve_connection(tls_stream, service).await
                        }
                        Err(e) => {
                            error!("TLS handshake failed: {}", e);
                            return;
                        }
                    }
                } else {
                    // Plain HTTP (for testing)
                    let service = service_fn(|req| {
                        let srv = Arc::clone(&server);
                        async move { srv.handle_http_request(req).await }
                    });
                    Http::new().serve_connection(stream, service).await
                };

                if let Err(e) = result {
                    error!("Error serving connection: {}", e);
                }
            });
        }
    }

    /// Handle an HTTP request.
    async fn handle_http_request(&self, req: Request<Body>) -> std::result::Result<Response<Body>, hyper::Error> {
        let (parts, body) = req.into_parts();
        let path = parts.uri.path();
        let method = parts.method;

        debug!("Webhook request: {} {}", method, path);

        let response = match (method, path) {
            (Method::POST, "/inject") | (Method::POST, "/mutate") => {
                // Read body
                let body_bytes = hyper::body::to_bytes(body).await?;

                match self.handle_request_body(&body_bytes) {
                    Ok(response_body) => {
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("Content-Type", "application/json")
                            .body(Body::from(response_body))
                            .unwrap()
                    }
                    Err(e) => {
                        error!("Failed to handle admission request: {}", e);
                        Response::builder()
                            .status(StatusCode::BAD_REQUEST)
                            .body(Body::from(format!("Error: {}", e)))
                            .unwrap()
                    }
                }
            }
            (Method::GET, "/health") | (Method::GET, "/healthz") => {
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Body::from(self.health_check()))
                    .unwrap()
            }
            (Method::GET, "/ready") | (Method::GET, "/readyz") => {
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Body::from(self.ready_check()))
                    .unwrap()
            }
            _ => {
                Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .body(Body::from("Not Found"))
                    .unwrap()
            }
        };

        Ok(response)
    }

    /// Create a TLS acceptor from certificate and key files.
    fn create_tls_acceptor(cert_path: &str, key_path: &str) -> Result<TlsAcceptor> {
        // Read and parse certificate
        let cert_pem = fs::read(cert_path)
            .map_err(|e| Error::Tls(format!("Failed to read cert: {}", e)))?;
        let mut cert_reader = BufReader::new(cert_pem.as_slice());
        let certs: Vec<Certificate> = rustls_pemfile::certs(&mut cert_reader)
            .map_err(|e| Error::Tls(format!("Failed to parse cert: {}", e)))?
            .into_iter()
            .map(Certificate)
            .collect();

        if certs.is_empty() {
            return Err(Error::Tls("No certificates found".to_string()));
        }

        // Read and parse private key
        let key_pem = fs::read(key_path)
            .map_err(|e| Error::Tls(format!("Failed to read key: {}", e)))?;
        let mut key_reader = BufReader::new(key_pem.as_slice());

        let key = rustls_pemfile::pkcs8_private_keys(&mut key_reader)
            .map_err(|e| Error::Tls(format!("Failed to parse key: {}", e)))?
            .into_iter()
            .next()
            .map(PrivateKey)
            .ok_or_else(|| Error::Tls("No private key found".to_string()))?;

        // Build server config
        let config = ServerConfig::builder()
            .with_safe_defaults()
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .map_err(|e| Error::Tls(e.to_string()))?;

        Ok(TlsAcceptor::from(Arc::new(config)))
    }
}

/// Webhook configuration for Kubernetes.
#[derive(Debug, Clone)]
pub struct WebhookConfig {
    /// Webhook name.
    pub name: String,
    /// Namespace selector.
    pub namespace_selector: Option<LabelSelector>,
    /// Object selector.
    pub object_selector: Option<LabelSelector>,
    /// Failure policy.
    pub failure_policy: FailurePolicy,
    /// Timeout seconds.
    pub timeout_seconds: i32,
    /// CA bundle.
    pub ca_bundle: Option<Vec<u8>>,
    /// Service reference.
    pub service: ServiceReference,
}

/// Label selector.
#[derive(Debug, Clone, Default)]
pub struct LabelSelector {
    /// Match labels.
    pub match_labels: std::collections::HashMap<String, String>,
    /// Match expressions.
    pub match_expressions: Vec<LabelSelectorRequirement>,
}

/// Label selector requirement.
#[derive(Debug, Clone)]
pub struct LabelSelectorRequirement {
    /// Label key.
    pub key: String,
    /// Operator.
    pub operator: String,
    /// Values.
    pub values: Vec<String>,
}

/// Failure policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailurePolicy {
    /// Ignore failures and allow the request.
    Ignore,
    /// Fail closed and reject the request.
    Fail,
}

/// Service reference.
#[derive(Debug, Clone)]
pub struct ServiceReference {
    /// Namespace.
    pub namespace: String,
    /// Service name.
    pub name: String,
    /// Path.
    pub path: Option<String>,
    /// Port.
    pub port: Option<i32>,
}

impl Default for WebhookConfig {
    fn default() -> Self {
        Self {
            name: "sidecar-injector.mesh.io".to_string(),
            namespace_selector: Some(LabelSelector {
                match_labels: [("mesh-injection".to_string(), "enabled".to_string())]
                    .into_iter()
                    .collect(),
                match_expressions: vec![],
            }),
            object_selector: None,
            failure_policy: FailurePolicy::Ignore,
            timeout_seconds: 10,
            ca_bundle: None,
            service: ServiceReference {
                namespace: "mesh-system".to_string(),
                name: "sidecar-injector".to_string(),
                path: Some("/inject".to_string()),
                port: Some(443),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_admission_request() -> AdmissionReview {
        let pod = json!({
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": "test-pod",
                "namespace": "default"
            },
            "spec": {
                "containers": [{
                    "name": "app",
                    "image": "nginx:latest",
                    "ports": [{
                        "containerPort": 80
                    }]
                }]
            }
        });

        AdmissionReview {
            api_version: "admission.k8s.io/v1".to_string(),
            kind: "AdmissionReview".to_string(),
            request: Some(AdmissionRequest {
                uid: "12345-67890".to_string(),
                kind: GroupVersionKind {
                    group: "".to_string(),
                    version: "v1".to_string(),
                    kind: "Pod".to_string(),
                },
                resource: GroupVersionResource {
                    group: "".to_string(),
                    version: "v1".to_string(),
                    resource: "pods".to_string(),
                },
                sub_resource: None,
                request_kind: None,
                request_resource: None,
                name: Some("test-pod".to_string()),
                namespace: Some("default".to_string()),
                operation: "CREATE".to_string(),
                user_info: UserInfo::default(),
                object: Some(pod),
                old_object: None,
                dry_run: false,
                options: None,
            }),
            response: None,
        }
    }

    #[test]
    fn test_webhook_server_creation() {
        let addr = "0.0.0.0:8443".parse().unwrap();
        let server = WebhookServer::new(addr, InjectionConfig::default());

        assert_eq!(server.address(), addr);
    }

    #[test]
    fn test_handle_admission_review() {
        let addr = "0.0.0.0:8443".parse().unwrap();
        let server = WebhookServer::new(addr, InjectionConfig::default());

        let review = test_admission_request();
        let response = server.handle_admission_review(review);

        assert!(response.response.is_some());
        let resp = response.response.unwrap();
        assert!(resp.allowed);
        assert!(resp.patch.is_some());
    }

    #[test]
    fn test_handle_non_pod_request() {
        let addr = "0.0.0.0:8443".parse().unwrap();
        let server = WebhookServer::new(addr, InjectionConfig::default());

        let mut review = test_admission_request();
        review.request.as_mut().unwrap().kind.kind = "Service".to_string();

        let response = server.handle_admission_review(review);
        let resp = response.response.unwrap();

        assert!(resp.allowed);
        assert!(resp.patch.is_none());
    }

    #[test]
    fn test_handle_non_create_request() {
        let addr = "0.0.0.0:8443".parse().unwrap();
        let server = WebhookServer::new(addr, InjectionConfig::default());

        let mut review = test_admission_request();
        review.request.as_mut().unwrap().operation = "UPDATE".to_string();

        let response = server.handle_admission_review(review);
        let resp = response.response.unwrap();

        assert!(resp.allowed);
        assert!(resp.patch.is_none());
    }

    #[test]
    fn test_handle_request_body() {
        let addr = "0.0.0.0:8443".parse().unwrap();
        let server = WebhookServer::new(addr, InjectionConfig::default());

        let review = test_admission_request();
        let body = serde_json::to_vec(&review).unwrap();

        let response = server.handle_request_body(&body).unwrap();
        let resp_review: AdmissionReview = serde_json::from_slice(&response).unwrap();

        assert!(resp_review.response.is_some());
        assert!(resp_review.response.unwrap().allowed);
    }

    #[test]
    fn test_health_check() {
        let addr = "0.0.0.0:8443".parse().unwrap();
        let server = WebhookServer::new(addr, InjectionConfig::default());

        assert_eq!(server.health_check(), "OK");
        assert_eq!(server.ready_check(), "READY");
    }

    #[test]
    fn test_webhook_config_default() {
        let config = WebhookConfig::default();

        assert_eq!(config.failure_policy, FailurePolicy::Ignore);
        assert_eq!(config.timeout_seconds, 10);
        assert!(config.namespace_selector.is_some());
    }

    #[test]
    fn test_already_injected_pod() {
        let addr = "0.0.0.0:8443".parse().unwrap();
        let server = WebhookServer::new(addr, InjectionConfig::default());

        let pod = json!({
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": "test-pod",
                "namespace": "default",
                "annotations": {
                    "sidecar.mesh.io/injected": "true"
                }
            },
            "spec": {
                "containers": [{
                    "name": "app",
                    "image": "nginx:latest"
                }]
            }
        });

        let mut review = test_admission_request();
        review.request.as_mut().unwrap().object = Some(pod);

        let response = server.handle_admission_review(review);
        let resp = response.response.unwrap();

        // Should be allowed but no patch (already injected)
        assert!(resp.allowed);
        assert!(resp.patch.is_none());
    }
}
