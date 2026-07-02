//! Integration tests for mTLS on the sidecar proxy's inbound data path.
//!
//! These tests exercise the real proxy: an inbound connection is terminated as
//! a rustls server connection using the sidecar's CA-issued certificate, the
//! peer's SPIFFE identity is extracted from its client certificate, and the
//! request is forwarded to a local application. They verify that:
//!
//!   1. A TLS client with a CA-issued cert negotiates TLS successfully and the
//!      local app sees the client's SPIFFE identity (via the XFCC header).
//!   2. A plaintext client is rejected when the proxy runs in Strict mode.

// Building ProxyConfig by mutating fields after `default()` reads more clearly
// here than a large struct literal interleaved with `.await` port probes.
#![allow(clippy::field_reassign_with_default)]

use service_mesh::{
    CertManager, CertificateAuthority, MtlsMode, ProxyConfig, ServiceIdentity, ServiceRegistry,
    SidecarProxy, TlsManager,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// A minimal HTTP application that echoes the `x-forwarded-client-cert` header
/// value back in its response body, so tests can observe the peer identity the
/// proxy propagated. Returns the port it is listening on.
async fn spawn_echo_app() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();

    tokio::spawn(async move {
        loop {
            let (mut stream, _) = match listener.accept().await {
                Ok(v) => v,
                Err(_) => break,
            };
            tokio::spawn(async move {
                // Read the full request (headers up to blank line; ignore body).
                let mut buf = Vec::new();
                let mut tmp = [0u8; 1024];
                loop {
                    match stream.read(&mut tmp).await {
                        Ok(0) => break,
                        Ok(n) => {
                            buf.extend_from_slice(&tmp[..n]);
                            if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                                break;
                            }
                        }
                        Err(_) => return,
                    }
                }
                let req = String::from_utf8_lossy(&buf);
                let xfcc = req
                    .lines()
                    .find(|l| l.to_ascii_lowercase().starts_with("x-forwarded-client-cert:"))
                    .and_then(|l| l.split_once(':').map(|(_, v)| v.trim().to_string()))
                    .unwrap_or_else(|| "none".to_string());

                let body = format!("peer={}", xfcc);
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(resp.as_bytes()).await;
                let _ = stream.flush().await;
            });
        }
    });

    port
}

/// Start a sidecar proxy on a free inbound port and return its port + a handle.
async fn spawn_proxy(mode: MtlsMode, app_port: u16, server_identity: &ServiceIdentity) -> u16 {
    // Pick a free inbound port up front.
    let probe = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let inbound_port = probe.local_addr().unwrap().port();
    drop(probe);

    let ca = CertificateAuthority::new(Duration::from_secs(3600)).unwrap();
    let cert = ca.issue_certificate(server_identity).unwrap();
    let cert_manager = Arc::new(CertManager::new(
        server_identity.clone(),
        cert,
        Duration::from_secs(3600),
    ));
    let registry = Arc::new(ServiceRegistry::new());

    let mut config = ProxyConfig::default();
    config.mtls_mode = mode;
    config.inbound_port = inbound_port;
    config.app_port = app_port;
    // Use distinct, unused ports for outbound/admin so run() can bind them.
    config.outbound_port = free_port().await;
    config.admin_port = free_port().await;
    config.tracing_config.enabled = false;

    let proxy = SidecarProxy::new(config, registry, cert_manager);
    tokio::spawn(async move {
        let _ = proxy.run().await;
    });

    // Give the listeners a moment to bind.
    tokio::time::sleep(Duration::from_millis(150)).await;
    inbound_port
}

async fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let p = l.local_addr().unwrap().port();
    drop(l);
    p
}

/// Build a client-side TlsManager for `client_identity`, all issued by the same
/// CA the proxy uses. Because both certs must chain to the same root for mTLS,
/// we return the CA together with the client cert so the caller can also mint a
/// matching server cert. Here we simply reuse one CA per test.
fn client_tls_manager(ca: &CertificateAuthority, client_identity: &ServiceIdentity) -> TlsManager {
    let cert = ca.issue_certificate(client_identity).unwrap();
    TlsManager::from_issued_cert(&cert).unwrap()
}

/// Full end-to-end mTLS test: the proxy terminates TLS, extracts the client
/// SPIFFE id, and the app echoes it back.
#[tokio::test]
async fn test_mtls_inbound_negotiates_and_exposes_peer_identity() {
    // A single CA issues both the server (proxy) cert and the client cert so
    // the mutual handshake validates.
    let ca = CertificateAuthority::new(Duration::from_secs(3600)).unwrap();

    let server_identity = ServiceIdentity::new("default", "backend");
    let client_identity = ServiceIdentity::new("default", "frontend");

    let app_port = spawn_echo_app().await;

    // Build the proxy with the SAME CA as the client by constructing it here.
    let server_cert = ca.issue_certificate(&server_identity).unwrap();
    let cert_manager = Arc::new(CertManager::new(
        server_identity.clone(),
        server_cert,
        Duration::from_secs(3600),
    ));
    let registry = Arc::new(ServiceRegistry::new());
    let mut config = ProxyConfig::default();
    config.mtls_mode = MtlsMode::Strict;
    let inbound_port = free_port().await;
    config.inbound_port = inbound_port;
    config.app_port = app_port;
    config.outbound_port = free_port().await;
    config.admin_port = free_port().await;
    config.tracing_config.enabled = false;
    let proxy = SidecarProxy::new(config, registry, cert_manager);
    tokio::spawn(async move {
        let _ = proxy.run().await;
    });
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Client TLS with a cert from the same CA.
    let client_tls = client_tls_manager(&ca, &client_identity);
    let tcp = TcpStream::connect(("127.0.0.1", inbound_port))
        .await
        .expect("tcp connect");

    // Dial by the server's certificate DNS name so WebPKI validation succeeds.
    let server_name = server_identity.tls_server_name();
    let mut tls = client_tls
        .connect(tcp, &server_name)
        .await
        .expect("TLS handshake should succeed");

    let request = "GET /hello HTTP/1.1\r\nHost: backend\r\nContent-Length: 0\r\n\r\n";
    tls.write_all(request.as_bytes()).await.unwrap();
    tls.flush().await.unwrap();

    let mut resp = Vec::new();
    let mut tmp = [0u8; 1024];
    loop {
        match tls.read(&mut tmp).await {
            Ok(0) => break,
            Ok(n) => {
                resp.extend_from_slice(&tmp[..n]);
                if resp.windows(4).any(|w| w == b"\r\n\r\n")
                    && String::from_utf8_lossy(&resp).contains("peer=")
                {
                    break;
                }
            }
            Err(_) => break,
        }
    }

    let text = String::from_utf8_lossy(&resp);
    assert!(text.contains("200 OK"), "expected 200, got: {}", text);
    // The app must have seen the client's SPIFFE id via the XFCC header.
    assert!(
        text.contains(&client_identity.spiffe_id),
        "response should carry client SPIFFE id {}, got: {}",
        client_identity.spiffe_id,
        text
    );
}

/// In Strict mode, a plaintext (non-TLS) client must be rejected: the handshake
/// the proxy expects never happens, so the connection does not yield an HTTP
/// response.
#[tokio::test]
async fn test_mtls_strict_rejects_plaintext_client() {
    let app_port = spawn_echo_app().await;
    let server_identity = ServiceIdentity::new("default", "backend");
    let inbound_port = spawn_proxy(MtlsMode::Strict, app_port, &server_identity).await;

    let mut tcp = TcpStream::connect(("127.0.0.1", inbound_port))
        .await
        .expect("tcp connect");

    // Send a plaintext HTTP request; the proxy is expecting a TLS ClientHello.
    let request = "GET /hello HTTP/1.1\r\nHost: backend\r\nContent-Length: 0\r\n\r\n";
    let _ = tcp.write_all(request.as_bytes()).await;
    let _ = tcp.flush().await;

    // The proxy should not return a valid HTTP response over the plaintext
    // connection. Read with a timeout; expect EOF/reset or non-HTTP bytes.
    let mut buf = Vec::new();
    let mut tmp = [0u8; 512];
    let read = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            match tcp.read(&mut tmp).await {
                Ok(0) => break,
                Ok(n) => {
                    buf.extend_from_slice(&tmp[..n]);
                    if buf.len() > 256 {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    })
    .await;

    // Either the read timed out / closed with no data, or whatever came back is
    // NOT a successful HTTP response served by the app.
    let text = String::from_utf8_lossy(&buf);
    assert!(
        !text.contains("200 OK"),
        "plaintext client must not receive a served HTTP 200, got: {}",
        text
    );
    let _ = read;
}

/// A TLS client whose certificate is issued by a DIFFERENT (untrusted) CA must
/// fail the mutual handshake in Strict mode.
#[tokio::test]
async fn test_mtls_rejects_untrusted_client_ca() {
    let server_identity = ServiceIdentity::new("default", "backend");
    let app_port = spawn_echo_app().await;
    let inbound_port = spawn_proxy(MtlsMode::Strict, app_port, &server_identity).await;

    // Client cert from an unrelated CA -> should not chain to the proxy's root.
    let rogue_ca = CertificateAuthority::new(Duration::from_secs(3600)).unwrap();
    let client_identity = ServiceIdentity::new("default", "attacker");
    let client_tls = client_tls_manager(&rogue_ca, &client_identity);

    let tcp = TcpStream::connect(("127.0.0.1", inbound_port))
        .await
        .expect("tcp connect");
    let server_name = server_identity.tls_server_name();

    // The handshake may fail on connect, or the server may reject the client
    // cert and drop the connection during the first read/write.
    let handshake = client_tls.connect(tcp, &server_name).await;
    let rejected = match handshake {
        Err(_) => true,
        Ok(mut tls) => {
            let req = "GET / HTTP/1.1\r\nHost: backend\r\n\r\n";
            if tls.write_all(req.as_bytes()).await.is_err() {
                true
            } else {
                let mut tmp = [0u8; 64];
                matches!(tls.read(&mut tmp).await, Ok(0) | Err(_))
            }
        }
    };
    assert!(rejected, "client with untrusted CA must be rejected");
}
