//! Unit tests for certificate management and mTLS functionality

use service_mesh::{CertificateAuthority, CertManager, IssuedCert, ServiceIdentity};
use std::time::Duration;

#[test]
fn test_certificate_authority_creation() {
    let ca = CertificateAuthority::new(Duration::from_secs(365 * 24 * 3600));
    assert!(ca.is_ok());
}

#[test]
fn test_service_identity_creation() {
    let identity = ServiceIdentity::new("production", "frontend");

    assert_eq!(identity.namespace, "production");
    assert_eq!(identity.service_account, "frontend");
    assert!(identity.spiffe_id.contains("frontend"));
    assert!(identity.spiffe_id.contains("production"));
}

#[test]
fn test_certificate_issuance() {
    let ca = CertificateAuthority::new(Duration::from_secs(365 * 24 * 3600))
        .expect("Failed to create CA");

    let identity = ServiceIdentity::new("staging", "backend");
    let cert = ca.issue_certificate(&identity);

    assert!(cert.is_ok());
    let cert = cert.unwrap();
    assert!(!cert.cert_chain.is_empty());
    assert!(!cert.private_key.is_empty());
}

#[test]
fn test_cert_manager_creation() {
    let ca = CertificateAuthority::new(Duration::from_secs(365 * 24 * 3600))
        .expect("Failed to create CA");

    let identity = ServiceIdentity::new("default", "api-gateway");
    let cert = ca.issue_certificate(&identity).expect("Failed to issue certificate");

    let cert_manager = CertManager::new(identity, cert, Duration::from_secs(3600));
    assert!(!cert_manager.needs_rotation());
}

#[test]
fn test_cert_manager_get_current_cert() {
    let ca = CertificateAuthority::new(Duration::from_secs(365 * 24 * 3600))
        .expect("Failed to create CA");

    let identity = ServiceIdentity::new("default", "worker");
    let cert = ca.issue_certificate(&identity).expect("Failed to issue certificate");

    let cert_manager = CertManager::new(identity.clone(), cert, Duration::from_secs(3600));
    let current = cert_manager.current_cert();

    assert!(!current.cert_chain.is_empty());
    assert!(!current.private_key.is_empty());
}

#[test]
fn test_cert_manager_identity() {
    let ca = CertificateAuthority::new(Duration::from_secs(365 * 24 * 3600))
        .expect("Failed to create CA");

    let identity = ServiceIdentity::new("test-ns", "cache-service");
    let cert = ca.issue_certificate(&identity).expect("Failed to issue certificate");

    let cert_manager = CertManager::new(identity.clone(), cert, Duration::from_secs(3600));
    let retrieved_identity = cert_manager.identity();

    assert_eq!(retrieved_identity.namespace, "test-ns");
    assert_eq!(retrieved_identity.service_account, "cache-service");
}

#[test]
fn test_spiffe_id_format() {
    let identity = ServiceIdentity::new("production", "auth-service");
    let spiffe_id = &identity.spiffe_id;

    assert!(spiffe_id.starts_with("spiffe://"));
    assert!(spiffe_id.contains("production"));
    assert!(spiffe_id.contains("auth-service"));
}

#[test]
fn test_root_cert_pem_retrieval() {
    let ca = CertificateAuthority::new(Duration::from_secs(365 * 24 * 3600))
        .expect("Failed to create CA");

    let root_pem = ca.root_cert_pem();
    assert!(root_pem.is_ok());
    assert!(!root_pem.unwrap().is_empty());
}

#[test]
fn test_concurrent_certificate_operations() {
    use std::sync::Arc;
    use std::thread;

    let ca = Arc::new(
        CertificateAuthority::new(Duration::from_secs(365 * 24 * 3600))
            .expect("Failed to create CA"),
    );

    let mut handles = vec![];

    for i in 0..10 {
        let ca_clone = Arc::clone(&ca);
        let handle = thread::spawn(move || {
            let identity = ServiceIdentity::new("default", &format!("service-{}", i));
            ca_clone.issue_certificate(&identity)
        });
        handles.push(handle);
    }

    for handle in handles {
        let result = handle.join().expect("Thread panicked");
        assert!(result.is_ok());
    }
}

#[test]
fn test_cert_update() {
    let ca = CertificateAuthority::new(Duration::from_secs(365 * 24 * 3600))
        .expect("Failed to create CA");

    let identity = ServiceIdentity::new("default", "update-test");
    let cert1 = ca.issue_certificate(&identity).expect("Failed to issue certificate");
    let cert2 = ca.issue_certificate(&identity).expect("Failed to issue certificate");

    let cert_manager = CertManager::new(identity, cert1, Duration::from_secs(3600));

    // Update with new cert
    cert_manager.update_cert(cert2);

    let current = cert_manager.current_cert();
    assert!(!current.cert_chain.is_empty());
}
