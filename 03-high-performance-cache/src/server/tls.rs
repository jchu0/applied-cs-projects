//! TLS support for Redis-lite server
//!
//! Provides TLS encryption for client connections using rustls.

use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::sync::Arc;

use rustls::{
    self,
    pki_types::{CertificateDer, PrivateKeyDer},
    ServerConfig, ServerConnection,
};
use rustls_pemfile::{certs, private_key};

/// TLS configuration for the server
#[derive(Clone)]
pub struct TlsConfig {
    /// Path to certificate file (PEM format)
    pub cert_path: String,
    /// Path to private key file (PEM format)
    pub key_path: String,
    /// Path to CA certificate for client authentication (optional)
    pub ca_cert_path: Option<String>,
    /// Require client certificate
    pub require_client_cert: bool,
    /// Minimum TLS version (default: TLS 1.2)
    pub min_version: TlsVersion,
}

/// Supported TLS versions
#[derive(Clone, Copy, Debug)]
pub enum TlsVersion {
    Tls12,
    Tls13,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            cert_path: String::new(),
            key_path: String::new(),
            ca_cert_path: None,
            require_client_cert: false,
            min_version: TlsVersion::Tls12,
        }
    }
}

/// TLS acceptor for incoming connections
pub struct TlsAcceptor {
    config: Arc<ServerConfig>,
}

impl TlsAcceptor {
    /// Create a new TLS acceptor from configuration
    pub fn new(config: &TlsConfig) -> io::Result<Self> {
        let certs = load_certs(&config.cert_path)?;
        let key = load_private_key(&config.key_path)?;

        let server_config = ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        Ok(Self {
            config: Arc::new(server_config),
        })
    }

    /// Accept a TLS connection
    pub fn accept(&self, stream: TcpStream) -> io::Result<TlsStream> {
        let conn = ServerConnection::new(Arc::clone(&self.config))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

        Ok(TlsStream::new(stream, conn))
    }
}

/// TLS stream wrapper
pub struct TlsStream {
    socket: TcpStream,
    tls: ServerConnection,
    /// Buffer for reading encrypted data
    read_buf: Vec<u8>,
}

impl TlsStream {
    /// Create a new TLS stream
    fn new(socket: TcpStream, tls: ServerConnection) -> Self {
        Self {
            socket,
            tls,
            read_buf: vec![0u8; 16384],
        }
    }

    /// Complete the TLS handshake
    pub fn handshake(&mut self) -> io::Result<()> {
        while self.tls.is_handshaking() {
            self.do_io()?;
        }
        Ok(())
    }

    /// Perform TLS I/O
    fn do_io(&mut self) -> io::Result<()> {
        // Read encrypted data from socket
        if self.tls.wants_read() {
            match self.socket.read(&mut self.read_buf) {
                Ok(0) => {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "connection closed"));
                }
                Ok(n) => {
                    self.tls.read_tls(&mut &self.read_buf[..n])
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                    self.tls.process_new_packets()
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                }
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {}
                Err(e) => return Err(e),
            }
        }

        // Write encrypted data to socket
        if self.tls.wants_write() {
            self.tls.write_tls(&mut self.socket)?;
        }

        Ok(())
    }

    /// Get the underlying socket
    pub fn get_ref(&self) -> &TcpStream {
        &self.socket
    }

    /// Get mutable reference to underlying socket
    pub fn get_mut(&mut self) -> &mut TcpStream {
        &mut self.socket
    }

    /// Check if TLS handshake is complete
    pub fn is_handshaking(&self) -> bool {
        self.tls.is_handshaking()
    }
}

impl Read for TlsStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.do_io()?;

        let mut reader = self.tls.reader();
        match reader.read(buf) {
            Ok(n) => Ok(n),
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => Ok(0),
            Err(e) => Err(e),
        }
    }
}

impl Write for TlsStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut writer = self.tls.writer();
        let n = writer.write(buf)?;
        self.do_io()?;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        let mut writer = self.tls.writer();
        writer.flush()?;
        self.do_io()?;
        self.socket.flush()
    }
}

/// Load certificates from PEM file
fn load_certs(path: &str) -> io::Result<Vec<CertificateDer<'static>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let certs: Vec<_> = certs(&mut reader)
        .filter_map(|r| r.ok())
        .collect();

    if certs.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "no certificates found in file",
        ));
    }

    Ok(certs)
}

/// Load private key from PEM file
fn load_private_key(path: &str) -> io::Result<PrivateKeyDer<'static>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    private_key(&mut reader)?
        .ok_or_else(|| io::Error::new(
            io::ErrorKind::InvalidData,
            "no private key found in file",
        ))
}

/// Check if TLS is configured
pub fn is_tls_configured(config: &TlsConfig) -> bool {
    !config.cert_path.is_empty() && !config.key_path.is_empty()
}

/// Generate self-signed certificate for testing
#[cfg(feature = "test-certs")]
pub fn generate_test_certs(cert_path: &Path, key_path: &Path) -> io::Result<()> {
    use rcgen::{generate_simple_self_signed, CertifiedKey};

    let subject_alt_names = vec![
        "localhost".to_string(),
        "127.0.0.1".to_string(),
    ];

    let CertifiedKey { cert, key_pair } = generate_simple_self_signed(subject_alt_names)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    std::fs::write(cert_path, cert.pem())?;
    std::fs::write(key_path, key_pair.serialize_pem())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tls_config_default() {
        let config = TlsConfig::default();
        assert!(config.cert_path.is_empty());
        assert!(config.key_path.is_empty());
        assert!(!config.require_client_cert);
    }

    #[test]
    fn test_is_tls_configured() {
        let config = TlsConfig::default();
        assert!(!is_tls_configured(&config));

        let config = TlsConfig {
            cert_path: "/path/to/cert.pem".to_string(),
            key_path: "/path/to/key.pem".to_string(),
            ..Default::default()
        };
        assert!(is_tls_configured(&config));
    }
}
