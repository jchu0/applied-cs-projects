//! Certificate management for mTLS.

use crate::config::ServiceIdentity;
use crate::{Error, Result};
use rcgen::{Certificate, CertificateParams, DistinguishedName, DnType, SanType};
use std::time::{Duration, SystemTime};

/// Issued certificate.
#[derive(Debug, Clone)]
pub struct IssuedCert {
    /// Certificate chain in PEM format.
    pub cert_chain: Vec<Vec<u8>>,
    /// Private key in PEM format.
    pub private_key: Vec<u8>,
    /// Expiry time.
    pub expiry: SystemTime,
}

/// Certificate Authority for issuing certificates.
pub struct CertificateAuthority {
    /// Root certificate.
    root_cert: Certificate,
    /// Certificate TTL.
    cert_ttl: Duration,
}

impl CertificateAuthority {
    /// Create a new CA.
    pub fn new(cert_ttl: Duration) -> Result<Self> {
        let mut params = CertificateParams::default();
        params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);

        let mut dn = DistinguishedName::new();
        dn.push(DnType::CommonName, "Mesh Root CA");
        dn.push(DnType::OrganizationName, "Mesh");
        params.distinguished_name = dn;

        let root_cert =
            Certificate::from_params(params).map_err(|e| Error::Certificate(e.to_string()))?;

        Ok(Self { root_cert, cert_ttl })
    }

    /// Issue a certificate for a service identity.
    pub fn issue_certificate(&self, identity: &ServiceIdentity) -> Result<IssuedCert> {
        let mut params = CertificateParams::default();

        let mut dn = DistinguishedName::new();
        dn.push(DnType::CommonName, &identity.service_account);
        dn.push(DnType::OrganizationName, "mesh");
        params.distinguished_name = dn;

        // Add SPIFFE ID as SAN
        params
            .subject_alt_names
            .push(SanType::URI(identity.spiffe_id.clone()));

        let cert =
            Certificate::from_params(params).map_err(|e| Error::Certificate(e.to_string()))?;

        let cert_pem = cert
            .serialize_pem_with_signer(&self.root_cert)
            .map_err(|e| Error::Certificate(e.to_string()))?;
        let key_pem = cert.serialize_private_key_pem();
        let root_pem = self
            .root_cert
            .serialize_pem()
            .map_err(|e| Error::Certificate(e.to_string()))?;

        Ok(IssuedCert {
            cert_chain: vec![cert_pem.into_bytes(), root_pem.into_bytes()],
            private_key: key_pem.into_bytes(),
            expiry: SystemTime::now() + self.cert_ttl,
        })
    }

    /// Get root CA certificate in PEM format.
    pub fn root_cert_pem(&self) -> Result<Vec<u8>> {
        self.root_cert
            .serialize_pem()
            .map(|s| s.into_bytes())
            .map_err(|e| Error::Certificate(e.to_string()))
    }
}

/// Certificate manager for automatic rotation.
pub struct CertManager {
    /// Current certificate.
    current_cert: parking_lot::RwLock<IssuedCert>,
    /// Service identity.
    identity: ServiceIdentity,
    /// Rotation threshold (e.g., 0.8 means rotate at 80% of lifetime).
    rotation_threshold: f64,
    /// Certificate TTL.
    cert_ttl: Duration,
}

impl CertManager {
    /// Create a new certificate manager.
    pub fn new(identity: ServiceIdentity, cert: IssuedCert, cert_ttl: Duration) -> Self {
        Self {
            current_cert: parking_lot::RwLock::new(cert),
            identity,
            rotation_threshold: 0.8,
            cert_ttl,
        }
    }

    /// Get current certificate.
    pub fn current_cert(&self) -> IssuedCert {
        self.current_cert.read().clone()
    }

    /// Update certificate.
    pub fn update_cert(&self, cert: IssuedCert) {
        *self.current_cert.write() = cert;
    }

    /// Check if certificate needs rotation.
    pub fn needs_rotation(&self) -> bool {
        let cert = self.current_cert.read();
        let remaining = cert
            .expiry
            .duration_since(SystemTime::now())
            .unwrap_or(Duration::ZERO);

        let threshold = self.cert_ttl.mul_f64(1.0 - self.rotation_threshold);
        remaining < threshold
    }

    /// Get identity.
    pub fn identity(&self) -> &ServiceIdentity {
        &self.identity
    }
}
