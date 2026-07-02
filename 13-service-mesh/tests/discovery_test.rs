//! Unit tests for service discovery functionality

use service_mesh::discovery::{ConsistentHashRing, LoadBalancerType, ServiceKey};
use service_mesh::{
    Endpoint, EndpointHealth, LoadBalancer, ServiceEndpoints, ServiceIdentity, ServiceRegistry,
};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

fn default_identity() -> ServiceIdentity {
    ServiceIdentity::new("default", "test-service")
}

#[test]
fn test_service_registry_initialization() {
    let registry = ServiceRegistry::new();
    assert!(registry.get_by_name("nonexistent").is_none());
}

#[test]
fn test_service_registration() {
    let registry = ServiceRegistry::new();

    let endpoints = ServiceEndpoints {
        endpoints: vec![
            Endpoint {
                address: "10.0.0.1:8080".parse().unwrap(),
                weight: 100,
                health: EndpointHealth::Healthy,
                metadata: Default::default(),
                tls_identity: default_identity(),
            },
            Endpoint {
                address: "10.0.0.2:8080".parse().unwrap(),
                weight: 100,
                health: EndpointHealth::Healthy,
                metadata: Default::default(),
                tls_identity: default_identity(),
            },
        ],
        load_balancer: LoadBalancerType::RoundRobin,
        policy: Default::default(),
    };

    let key = ServiceKey::new("api-gateway", "default", 8080);
    registry.register(key, endpoints);

    let discovered = registry.get_by_name("api-gateway");
    assert!(discovered.is_some());
    assert_eq!(discovered.unwrap().endpoints.len(), 2);
}

#[test]
fn test_endpoint_health_filtering() {
    let endpoints = ServiceEndpoints {
        endpoints: vec![
            Endpoint {
                address: "10.0.0.1:5432".parse().unwrap(),
                weight: 100,
                health: EndpointHealth::Healthy,
                metadata: Default::default(),
                tls_identity: default_identity(),
            },
            Endpoint {
                address: "10.0.0.2:5432".parse().unwrap(),
                weight: 100,
                health: EndpointHealth::Unhealthy,
                metadata: Default::default(),
                tls_identity: default_identity(),
            },
            Endpoint {
                address: "10.0.0.3:5432".parse().unwrap(),
                weight: 100,
                health: EndpointHealth::Healthy,
                metadata: Default::default(),
                tls_identity: default_identity(),
            },
        ],
        load_balancer: LoadBalancerType::RoundRobin,
        policy: Default::default(),
    };

    let healthy: Vec<_> = endpoints
        .endpoints
        .iter()
        .filter(|e| e.health == EndpointHealth::Healthy)
        .collect();

    assert_eq!(healthy.len(), 2);
}

#[test]
fn test_load_balancer_round_robin() {
    let endpoints = vec![
        Endpoint {
            address: "127.0.0.1:8001".parse().unwrap(),
            weight: 100,
            health: EndpointHealth::Healthy,
            metadata: Default::default(),
            tls_identity: default_identity(),
        },
        Endpoint {
            address: "127.0.0.2:8002".parse().unwrap(),
            weight: 100,
            health: EndpointHealth::Healthy,
            metadata: Default::default(),
            tls_identity: default_identity(),
        },
        Endpoint {
            address: "127.0.0.3:8003".parse().unwrap(),
            weight: 100,
            health: EndpointHealth::Healthy,
            metadata: Default::default(),
            tls_identity: default_identity(),
        },
    ];

    let balancer = LoadBalancer::new(LoadBalancerType::RoundRobin);

    let mut selections = vec![];
    for _ in 0..9 {
        selections.push(balancer.select(&endpoints).unwrap().address);
    }

    // Verify round-robin pattern
    for i in 0..3 {
        assert_eq!(selections[i], endpoints[i].address);
        assert_eq!(selections[i + 3], endpoints[i].address);
        assert_eq!(selections[i + 6], endpoints[i].address);
    }
}

#[test]
fn test_load_balancer_random() {
    let endpoints = vec![
        Endpoint {
            address: "127.0.0.1:8001".parse().unwrap(),
            weight: 100,
            health: EndpointHealth::Healthy,
            metadata: Default::default(),
            tls_identity: default_identity(),
        },
        Endpoint {
            address: "127.0.0.2:8002".parse().unwrap(),
            weight: 100,
            health: EndpointHealth::Healthy,
            metadata: Default::default(),
            tls_identity: default_identity(),
        },
    ];

    let balancer = LoadBalancer::new(LoadBalancerType::Random);

    // Should be able to select endpoints
    for _ in 0..10 {
        let selected = balancer.select(&endpoints);
        assert!(selected.is_some());
    }
}

#[test]
fn test_concurrent_registry_access() {
    let registry = Arc::new(ServiceRegistry::new());
    let mut handles = vec![];

    // Writer thread
    let registry_clone = Arc::clone(&registry);
    let writer = std::thread::spawn(move || {
        for i in 0..10 {
            let key = ServiceKey::new(format!("service-{}", i), "default", 8080);
            let endpoints = ServiceEndpoints {
                endpoints: vec![Endpoint {
                    address: format!("10.0.0.{}:8080", i).parse().unwrap(),
                    weight: 100,
                    health: EndpointHealth::Healthy,
                    metadata: Default::default(),
                    tls_identity: default_identity(),
                }],
                load_balancer: LoadBalancerType::RoundRobin,
                policy: Default::default(),
            };
            registry_clone.register(key, endpoints);
        }
    });
    handles.push(writer);

    // Reader threads
    for _ in 0..5 {
        let registry_clone = Arc::clone(&registry);
        let reader = std::thread::spawn(move || {
            for i in 0..10 {
                let _ = registry_clone.get_by_name(&format!("service-{}", i));
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        });
        handles.push(reader);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_empty_endpoints() {
    let endpoints: Vec<Endpoint> = vec![];
    let balancer = LoadBalancer::new(LoadBalancerType::RoundRobin);

    let selected = balancer.select(&endpoints);
    assert!(selected.is_none());
}

#[test]
fn test_endpoint_metadata() {
    let mut metadata = HashMap::new();
    metadata.insert("region".to_string(), "us-west".to_string());
    metadata.insert("zone".to_string(), "a".to_string());

    let endpoint = Endpoint {
        address: "10.0.0.1:8080".parse().unwrap(),
        weight: 100,
        health: EndpointHealth::Healthy,
        metadata,
        tls_identity: default_identity(),
    };

    assert_eq!(
        endpoint.metadata.get("region"),
        Some(&"us-west".to_string())
    );
    assert_eq!(endpoint.metadata.get("zone"), Some(&"a".to_string()));
}

#[test]
fn test_service_key_creation() {
    let key = ServiceKey::new("my-service", "production", 443);

    assert_eq!(key.name, "my-service");
    assert_eq!(key.namespace, "production");
    assert_eq!(key.port, 443);
}

fn hashring_endpoints(n: usize) -> Vec<Endpoint> {
    (0..n)
        .map(|i| Endpoint {
            address: format!("10.1.0.{}:8080", i + 1).parse().unwrap(),
            weight: 100,
            health: EndpointHealth::Healthy,
            metadata: Default::default(),
            tls_identity: default_identity(),
        })
        .collect()
}

#[test]
fn test_ring_hash_stable_key_mapping() {
    let endpoints = hashring_endpoints(4);
    let lb = LoadBalancer::new(LoadBalancerType::RingHash);

    // A given key must map to the same backend across many calls.
    for k in 0..20 {
        let key = format!("customer-{k}");
        let first = lb.select_with_key(&endpoints, &key).unwrap().address;
        for _ in 0..10 {
            assert_eq!(lb.select_with_key(&endpoints, &key).unwrap().address, first);
        }
    }
}

#[test]
fn test_ring_hash_distributes_keys_across_backends() {
    let endpoints = hashring_endpoints(4);
    let lb = LoadBalancer::new(LoadBalancerType::RingHash);

    let mut seen: HashMap<SocketAddr, usize> = HashMap::new();
    for k in 0..2000 {
        let a = lb.select_with_key(&endpoints, &format!("k{k}")).unwrap().address;
        *seen.entry(a).or_default() += 1;
    }

    // Not all keys land on one backend.
    assert_eq!(seen.len(), 4, "keys should spread across all 4 backends");
    for (_, c) in seen {
        assert!(c > 100, "each backend should get a meaningful share");
    }
}

#[test]
fn test_ring_hash_bounded_reassignment_on_membership_change() {
    let addrs: Vec<SocketAddr> = hashring_endpoints(5).iter().map(|e| e.address).collect();
    let before = ConsistentHashRing::new(&addrs);

    // Drop one backend.
    let removed = addrs[3];
    let after_addrs: Vec<SocketAddr> = addrs.iter().copied().filter(|a| *a != removed).collect();
    let after = ConsistentHashRing::new(&after_addrs);

    let total = 4000;
    let mut moved = 0;
    for k in 0..total {
        let key = format!("k{k}");
        let b = before.lookup(&key).unwrap();
        let a = after.lookup(&key).unwrap();
        if b != a {
            moved += 1;
            assert_eq!(b, removed, "only keys on the removed node may move");
        }
    }
    let frac = moved as f64 / total as f64;
    assert!(frac < 0.35, "reassignment fraction too high: {frac}");
}

#[test]
fn test_unhealthy_endpoints_filtered() {
    let endpoints = vec![
        Endpoint {
            address: "127.0.0.1:8001".parse().unwrap(),
            weight: 100,
            health: EndpointHealth::Unhealthy,
            metadata: Default::default(),
            tls_identity: default_identity(),
        },
        Endpoint {
            address: "127.0.0.2:8002".parse().unwrap(),
            weight: 100,
            health: EndpointHealth::Healthy,
            metadata: Default::default(),
            tls_identity: default_identity(),
        },
    ];

    let balancer = LoadBalancer::new(LoadBalancerType::RoundRobin);

    // Should only select the healthy endpoint
    for _ in 0..10 {
        let selected = balancer.select(&endpoints).unwrap();
        assert_eq!(selected.address, "127.0.0.2:8002".parse::<SocketAddr>().unwrap());
    }
}
