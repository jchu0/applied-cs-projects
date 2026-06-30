//! xDS protocol types and messages.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Version info for xDS resources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    pub version: String,
    pub nonce: String,
}

/// Discovery request from proxy to control plane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryRequest {
    /// The version of resources being ACK/NACK'd.
    pub version_info: String,
    /// Node identifier.
    pub node: Node,
    /// Resource names to subscribe to.
    pub resource_names: Vec<String>,
    /// Type URL of resources.
    pub type_url: String,
    /// Response nonce.
    pub response_nonce: String,
    /// Error detail if NACK.
    pub error_detail: Option<Status>,
}

/// Discovery response from control plane to proxy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryResponse {
    /// Version of the resource set.
    pub version_info: String,
    /// List of resources.
    pub resources: Vec<Resource>,
    /// Type URL of resources.
    pub type_url: String,
    /// Nonce for ACK/NACK.
    pub nonce: String,
}

/// Node identifier for the proxy.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Node {
    /// Unique node identifier.
    pub id: String,
    /// Cluster the node belongs to.
    pub cluster: String,
    /// Node metadata.
    pub metadata: HashMap<String, String>,
    /// Locality information.
    pub locality: Option<Locality>,
}

/// Locality information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Locality {
    pub region: String,
    pub zone: String,
    pub sub_zone: String,
}

/// Status for error reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Status {
    pub code: i32,
    pub message: String,
}

/// Generic resource wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    pub type_url: String,
    pub value: Vec<u8>,
}

// xDS Type URLs
pub const TYPE_URL_CDS: &str = "type.googleapis.com/envoy.config.cluster.v3.Cluster";
pub const TYPE_URL_EDS: &str = "type.googleapis.com/envoy.config.endpoint.v3.ClusterLoadAssignment";
pub const TYPE_URL_LDS: &str = "type.googleapis.com/envoy.config.listener.v3.Listener";
pub const TYPE_URL_RDS: &str = "type.googleapis.com/envoy.config.route.v3.RouteConfiguration";

/// Cluster definition (CDS).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cluster {
    /// Cluster name.
    pub name: String,
    /// Connect timeout.
    #[serde(with = "duration_serde")]
    pub connect_timeout: Duration,
    /// Cluster type.
    pub cluster_type: ClusterType,
    /// Load balancing policy.
    pub lb_policy: LbPolicy,
    /// Health checks configuration.
    pub health_checks: Vec<HealthCheck>,
    /// Circuit breaker thresholds.
    pub circuit_breakers: Option<CircuitBreakerThresholds>,
    /// TLS context.
    pub tls_context: Option<TlsContext>,
}

/// Cluster discovery type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClusterType {
    /// Use static endpoint list.
    Static,
    /// Use strict DNS resolution.
    StrictDns,
    /// Use logical DNS resolution.
    LogicalDns,
    /// Use EDS for endpoint discovery.
    Eds,
    /// Original destination cluster.
    OriginalDst,
}

/// Load balancing policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LbPolicy {
    RoundRobin,
    LeastRequest,
    Random,
    RingHash,
    Maglev,
}

/// Health check configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    /// Health check timeout.
    #[serde(with = "duration_serde")]
    pub timeout: Duration,
    /// Health check interval.
    #[serde(with = "duration_serde")]
    pub interval: Duration,
    /// Unhealthy threshold.
    pub unhealthy_threshold: u32,
    /// Healthy threshold.
    pub healthy_threshold: u32,
    /// Health check type.
    pub check_type: HealthCheckType,
}

/// Health check type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthCheckType {
    Tcp,
    Http { path: String, expected_statuses: Vec<u16> },
    Grpc { service_name: Option<String> },
}

/// Circuit breaker thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerThresholds {
    /// Maximum connections.
    pub max_connections: u32,
    /// Maximum pending requests.
    pub max_pending_requests: u32,
    /// Maximum requests.
    pub max_requests: u32,
    /// Maximum retries.
    pub max_retries: u32,
}

/// TLS context configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsContext {
    /// Common TLS context.
    pub common_tls_context: CommonTlsContext,
    /// SNI to use.
    pub sni: Option<String>,
}

/// Common TLS context.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CommonTlsContext {
    /// TLS certificate chain.
    pub tls_certificates: Vec<TlsCertificate>,
    /// Validation context.
    pub validation_context: Option<ValidationContext>,
    /// ALPN protocols.
    pub alpn_protocols: Vec<String>,
}

/// TLS certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsCertificate {
    /// Certificate chain in PEM format.
    pub certificate_chain: String,
    /// Private key in PEM format.
    pub private_key: String,
}

/// TLS validation context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationContext {
    /// Trusted CA certificates.
    pub trusted_ca: String,
    /// Subject alt name matchers.
    pub match_subject_alt_names: Vec<StringMatcher>,
}

/// String matcher for SAN validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StringMatcher {
    Exact(String),
    Prefix(String),
    Suffix(String),
    Regex(String),
}

/// Cluster load assignment (EDS).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterLoadAssignment {
    /// Cluster name.
    pub cluster_name: String,
    /// Endpoints by locality.
    pub endpoints: Vec<LocalityLbEndpoints>,
    /// Load balancing policy overrides.
    pub policy: Option<Policy>,
}

/// Locality-based endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalityLbEndpoints {
    /// Locality.
    pub locality: Option<Locality>,
    /// Endpoints.
    pub lb_endpoints: Vec<LbEndpoint>,
    /// Load balancing weight.
    pub load_balancing_weight: Option<u32>,
    /// Priority.
    pub priority: u32,
}

/// Single load-balanced endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LbEndpoint {
    /// Endpoint address.
    pub endpoint: Endpoint,
    /// Health status.
    pub health_status: HealthStatus,
    /// Load balancing weight.
    pub load_balancing_weight: Option<u32>,
    /// Endpoint metadata.
    pub metadata: HashMap<String, String>,
}

/// Endpoint address.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Endpoint {
    /// Socket address.
    pub address: SocketAddress,
    /// Health check config.
    pub health_check_config: Option<HealthCheckConfig>,
}

/// Socket address.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocketAddress {
    /// IP address or hostname.
    pub address: String,
    /// Port number.
    pub port: u16,
    /// Protocol.
    pub protocol: Protocol,
}

/// Protocol type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Protocol {
    Tcp,
    Udp,
}

/// Health check config for endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    pub port: u16,
    pub hostname: Option<String>,
}

/// Endpoint health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Unknown,
    Healthy,
    Unhealthy,
    Draining,
    Timeout,
    Degraded,
}

/// Load balancing policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// Drop overloads.
    pub drop_overloads: Vec<DropOverload>,
    /// Overprovisioning factor.
    pub overprovisioning_factor: u32,
}

/// Drop overload configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropOverload {
    pub category: String,
    pub drop_percentage: f64,
}

/// Listener configuration (LDS).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Listener {
    /// Listener name.
    pub name: String,
    /// Address to bind.
    pub address: SocketAddress,
    /// Filter chains.
    pub filter_chains: Vec<FilterChain>,
    /// Listener filters.
    pub listener_filters: Vec<ListenerFilter>,
    /// Traffic direction.
    pub traffic_direction: TrafficDirection,
}

/// Traffic direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrafficDirection {
    Unspecified,
    Inbound,
    Outbound,
}

/// Filter chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterChain {
    /// Filter chain match.
    pub filter_chain_match: Option<FilterChainMatch>,
    /// Filters in the chain.
    pub filters: Vec<Filter>,
    /// Transport socket (TLS).
    pub transport_socket: Option<TransportSocket>,
    /// Name.
    pub name: String,
}

/// Filter chain match criteria.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FilterChainMatch {
    /// Destination port.
    pub destination_port: Option<u16>,
    /// Prefix ranges.
    pub prefix_ranges: Vec<CidrRange>,
    /// Server names (SNI).
    pub server_names: Vec<String>,
    /// Transport protocol.
    pub transport_protocol: String,
    /// Application protocols.
    pub application_protocols: Vec<String>,
}

/// CIDR range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CidrRange {
    pub address_prefix: String,
    pub prefix_len: u32,
}

/// Network filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Filter {
    pub name: String,
    pub typed_config: FilterConfig,
}

/// Filter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterConfig {
    HttpConnectionManager(HttpConnectionManager),
    TcpProxy(TcpProxy),
    Raw(Vec<u8>),
}

/// HTTP connection manager config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpConnectionManager {
    /// Stat prefix.
    pub stat_prefix: String,
    /// Route configuration.
    pub route_config: Option<RouteConfiguration>,
    /// RDS config source.
    pub rds: Option<Rds>,
    /// HTTP filters.
    pub http_filters: Vec<HttpFilter>,
}

/// RDS configuration source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rds {
    pub config_source: ConfigSource,
    pub route_config_name: String,
}

/// Configuration source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSource {
    pub resource_api_version: String,
    pub ads: bool,
}

/// HTTP filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpFilter {
    pub name: String,
    pub typed_config: Vec<u8>,
}

/// TCP proxy config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcpProxy {
    pub stat_prefix: String,
    pub cluster: String,
}

/// Listener filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListenerFilter {
    pub name: String,
    pub typed_config: Vec<u8>,
}

/// Transport socket (TLS).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportSocket {
    pub name: String,
    pub typed_config: TlsContext,
}

/// Route configuration (RDS).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteConfiguration {
    /// Route config name.
    pub name: String,
    /// Virtual hosts.
    pub virtual_hosts: Vec<VirtualHost>,
}

/// Virtual host configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualHost {
    /// Virtual host name.
    pub name: String,
    /// Domains to match.
    pub domains: Vec<String>,
    /// Routes.
    pub routes: Vec<Route>,
    /// Request headers to add.
    pub request_headers_to_add: Vec<HeaderValueOption>,
    /// Response headers to add.
    pub response_headers_to_add: Vec<HeaderValueOption>,
    /// Retry policy.
    pub retry_policy: Option<RetryPolicyConfig>,
}

/// Route configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route {
    /// Route name.
    pub name: String,
    /// Match criteria.
    pub match_criteria: RouteMatch,
    /// Route action.
    pub route: RouteAction,
}

/// Route match criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteMatch {
    /// Path match.
    pub path: PathMatch,
    /// Header matches.
    pub headers: Vec<HeaderMatcher>,
    /// Query parameter matches.
    pub query_parameters: Vec<QueryParameterMatcher>,
}

/// Path matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathMatch {
    Prefix(String),
    Exact(String),
    Regex(String),
}

/// Header matcher.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeaderMatcher {
    pub name: String,
    pub match_type: StringMatcher,
    pub invert_match: bool,
}

/// Query parameter matcher.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryParameterMatcher {
    pub name: String,
    pub string_match: StringMatcher,
}

/// Route action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteAction {
    /// Cluster to route to.
    pub cluster: Option<String>,
    /// Weighted clusters.
    pub weighted_clusters: Option<WeightedCluster>,
    /// Timeout.
    #[serde(with = "duration_serde")]
    pub timeout: Duration,
    /// Retry policy.
    pub retry_policy: Option<RetryPolicyConfig>,
    /// Request mirror policy.
    pub request_mirror_policies: Vec<RequestMirrorPolicy>,
}

/// Weighted cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedCluster {
    pub clusters: Vec<ClusterWeight>,
    pub total_weight: Option<u32>,
}

/// Cluster with weight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterWeight {
    pub name: String,
    pub weight: u32,
}

/// Retry policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicyConfig {
    pub retry_on: String,
    pub num_retries: u32,
    #[serde(with = "duration_serde")]
    pub per_try_timeout: Duration,
}

/// Request mirror policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMirrorPolicy {
    pub cluster: String,
    pub runtime_fraction: f64,
}

/// Header value option.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeaderValueOption {
    pub header: HeaderValue,
    pub append: bool,
}

/// Header value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeaderValue {
    pub key: String,
    pub value: String,
}

/// Custom duration serialization.
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let millis = duration.as_millis() as u64;
        millis.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}

impl Default for Cluster {
    fn default() -> Self {
        Self {
            name: String::new(),
            connect_timeout: Duration::from_secs(5),
            cluster_type: ClusterType::Eds,
            lb_policy: LbPolicy::RoundRobin,
            health_checks: vec![],
            circuit_breakers: None,
            tls_context: None,
        }
    }
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self::Unknown
    }
}

impl Default for Protocol {
    fn default() -> Self {
        Self::Tcp
    }
}
