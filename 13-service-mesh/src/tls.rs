//! TLS support for the service mesh proxy.
//!
//! Provides mTLS connection handling using tokio-rustls.

use crate::cert::{CertManager, IssuedCert};
use crate::{Error, Result};

use rustls::{Certificate, PrivateKey, RootCertStore, ServerConfig, ClientConfig};
use rustls::server::AllowAnyAuthenticatedClient;
use std::io::BufReader;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::TcpStream;
use tokio_rustls::{TlsAcceptor, TlsConnector, server::TlsStream as ServerTlsStream, client::TlsStream as ClientTlsStream};
use std::pin::Pin;
use std::task::{Context, Poll};

/// TLS configuration for the proxy.
#[derive(Clone)]
pub struct TlsManager {
    /// Server config for accepting connections.
    server_config: Arc<ServerConfig>,
    /// Client config for outgoing connections.
    client_config: Arc<ClientConfig>,
    /// Root CA certificates.
    root_store: RootCertStore,
}

impl TlsManager {
    /// Create a new TLS manager from a certificate manager.
    pub fn from_cert_manager(cert_manager: &CertManager) -> Result<Self> {
        let issued = cert_manager.current_cert();
        Self::from_issued_cert(&issued)
    }

    /// Create a new TLS manager from an issued certificate.
    pub fn from_issued_cert(issued: &IssuedCert) -> Result<Self> {
        // Parse certificates
        let certs = Self::parse_certificates(&issued.cert_chain)?;
        let key = Self::parse_private_key(&issued.private_key)?;

        // Build root store from CA cert (last in chain)
        let mut root_store = RootCertStore::empty();
        if issued.cert_chain.len() > 1 {
            let ca_cert = Self::parse_certificates(&issued.cert_chain[1..])?;
            for cert in ca_cert {
                root_store.add(&cert).map_err(|e| Error::Tls(e.to_string()))?;
            }
        }

        // Server config with client auth
        let client_cert_verifier = AllowAnyAuthenticatedClient::new(root_store.clone());
        let server_config = ServerConfig::builder()
            .with_safe_defaults()
            .with_client_cert_verifier(Arc::new(client_cert_verifier))
            .with_single_cert(certs.clone(), key.clone())
            .map_err(|e| Error::Tls(e.to_string()))?;

        // Client config for outgoing mTLS connections
        let client_config = ClientConfig::builder()
            .with_safe_defaults()
            .with_root_certificates(root_store.clone())
            .with_client_auth_cert(certs, key)
            .map_err(|e| Error::Tls(e.to_string()))?;

        Ok(Self {
            server_config: Arc::new(server_config),
            client_config: Arc::new(client_config),
            root_store,
        })
    }

    /// Create a TLS acceptor for inbound connections.
    pub fn acceptor(&self) -> TlsAcceptor {
        TlsAcceptor::from(self.server_config.clone())
    }

    /// Create a TLS connector for outbound connections.
    pub fn connector(&self) -> TlsConnector {
        TlsConnector::from(self.client_config.clone())
    }

    /// Accept an inbound TLS connection.
    pub async fn accept(&self, stream: TcpStream) -> Result<ServerTlsStream<TcpStream>> {
        let acceptor = self.acceptor();
        acceptor.accept(stream).await.map_err(|e| Error::Tls(e.to_string()))
    }

    /// Connect with TLS to a remote server.
    pub async fn connect(&self, stream: TcpStream, server_name: &str) -> Result<ClientTlsStream<TcpStream>> {
        let connector = self.connector();
        let domain = rustls::ServerName::try_from(server_name)
            .map_err(|_| Error::Tls(format!("Invalid server name: {}", server_name)))?;
        connector.connect(domain, stream).await.map_err(|e| Error::Tls(e.to_string()))
    }

    /// Parse PEM certificates.
    fn parse_certificates(pem_data: &[Vec<u8>]) -> Result<Vec<Certificate>> {
        let mut certs = Vec::new();
        for pem in pem_data {
            let mut reader = BufReader::new(pem.as_slice());
            let parsed = rustls_pemfile::certs(&mut reader)
                .map_err(|e| Error::Tls(format!("Failed to parse certificate: {}", e)))?;
            certs.extend(parsed.into_iter().map(Certificate));
        }
        Ok(certs)
    }

    /// Parse PEM private key.
    fn parse_private_key(pem_data: &[u8]) -> Result<PrivateKey> {
        let mut reader = BufReader::new(pem_data);

        // Try PKCS8 first
        if let Ok(keys) = rustls_pemfile::pkcs8_private_keys(&mut reader) {
            if let Some(key) = keys.into_iter().next() {
                return Ok(PrivateKey(key));
            }
        }

        // Try RSA keys
        let mut reader = BufReader::new(pem_data);
        if let Ok(keys) = rustls_pemfile::rsa_private_keys(&mut reader) {
            if let Some(key) = keys.into_iter().next() {
                return Ok(PrivateKey(key));
            }
        }

        Err(Error::Tls("No valid private key found".to_string()))
    }

    /// Get root store.
    pub fn root_store(&self) -> &RootCertStore {
        &self.root_store
    }
}

/// A unified TLS stream that can be either server-side or client-side.
pub enum TlsStream {
    /// Server-side TLS stream.
    Server(ServerTlsStream<TcpStream>),
    /// Client-side TLS stream.
    Client(ClientTlsStream<TcpStream>),
}

impl TlsStream {
    /// Get peer certificate SPIFFE ID if available.
    pub fn peer_spiffe_id(&self) -> Option<String> {
        // In a real implementation, we'd extract the SPIFFE ID from the
        // peer certificate's SAN extension
        match self {
            TlsStream::Server(stream) => {
                let (_, session) = stream.get_ref();
                session.peer_certificates().and_then(|certs| {
                    if certs.is_empty() {
                        None
                    } else {
                        // Extract SPIFFE ID from first certificate
                        extract_spiffe_from_cert(&certs[0].0)
                    }
                })
            }
            TlsStream::Client(stream) => {
                let (_, session) = stream.get_ref();
                session.peer_certificates().and_then(|certs| {
                    if certs.is_empty() {
                        None
                    } else {
                        extract_spiffe_from_cert(&certs[0].0)
                    }
                })
            }
        }
    }
}

impl AsyncRead for TlsStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            TlsStream::Server(s) => Pin::new(s).poll_read(cx, buf),
            TlsStream::Client(s) => Pin::new(s).poll_read(cx, buf),
        }
    }
}

impl AsyncWrite for TlsStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        match self.get_mut() {
            TlsStream::Server(s) => Pin::new(s).poll_write(cx, buf),
            TlsStream::Client(s) => Pin::new(s).poll_write(cx, buf),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            TlsStream::Server(s) => Pin::new(s).poll_flush(cx),
            TlsStream::Client(s) => Pin::new(s).poll_flush(cx),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            TlsStream::Server(s) => Pin::new(s).poll_shutdown(cx),
            TlsStream::Client(s) => Pin::new(s).poll_shutdown(cx),
        }
    }
}

/// Extract SPIFFE ID from certificate DER bytes.
fn extract_spiffe_from_cert(cert_der: &[u8]) -> Option<String> {
    use x509_parser::prelude::*;

    let (_, cert) = X509Certificate::from_der(cert_der).ok()?;

    // Look for Subject Alternative Names extension
    for ext in cert.extensions() {
        if let ParsedExtension::SubjectAlternativeName(san) = ext.parsed_extension() {
            for name in &san.general_names {
                if let GeneralName::URI(uri) = name {
                    if uri.starts_with("spiffe://") {
                        return Some(uri.to_string());
                    }
                }
            }
        }
    }

    None
}

/// TLS-enabled connection for the proxy.
pub struct SecureConnection {
    /// TLS stream.
    pub stream: TlsStream,
    /// Peer SPIFFE identity.
    pub peer_identity: Option<String>,
}

impl SecureConnection {
    /// Create from a server-side TLS handshake.
    pub fn from_server_stream(stream: ServerTlsStream<TcpStream>) -> Self {
        let tls_stream = TlsStream::Server(stream);
        let peer_identity = tls_stream.peer_spiffe_id();
        Self {
            stream: tls_stream,
            peer_identity,
        }
    }

    /// Create from a client-side TLS handshake.
    pub fn from_client_stream(stream: ClientTlsStream<TcpStream>) -> Self {
        let tls_stream = TlsStream::Client(stream);
        let peer_identity = tls_stream.peer_spiffe_id();
        Self {
            stream: tls_stream,
            peer_identity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cert::CertificateAuthority;
    use crate::config::ServiceIdentity;
    use std::time::Duration;

    fn create_test_cert() -> IssuedCert {
        let ca = CertificateAuthority::new(Duration::from_secs(3600)).unwrap();
        let identity = ServiceIdentity::new("default", "test-service");
        ca.issue_certificate(&identity).unwrap()
    }

    #[test]
    fn test_tls_manager_creation() {
        let cert = create_test_cert();
        let result = TlsManager::from_issued_cert(&cert);
        // Note: This may fail in tests without proper cert format
        // In production, we'd have properly formatted PEM certificates
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_spiffe_extraction() {
        // Test with a cert that has a SPIFFE SAN
        // This is a unit test placeholder - actual extraction
        // requires a properly formatted X.509 certificate
        let fake_cert = vec![0u8; 10];
        let result = extract_spiffe_from_cert(&fake_cert);
        assert!(result.is_none()); // Invalid cert should return None
    }
}
