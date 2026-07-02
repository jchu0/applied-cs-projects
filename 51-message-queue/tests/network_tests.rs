//! End-to-end integration tests for the wire protocol server + client.
//!
//! Each test binds a real TCP listener on an ephemeral port (`127.0.0.1:0`),
//! spawns the server on a background task, connects with [`MqClient`], and
//! exercises the protocol over an actual socket.

use message_queue::server::{serve_with_listener, ServerOptions};
use message_queue::{Broker, BrokerConfig, MqClient, Request, WireMessage};
use std::sync::Arc;
use tempfile::TempDir;
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};

/// Boot a broker on a temp dir and start serving on an ephemeral port.
/// Returns the bound address, the temp dir guard, and the server task handle.
async fn start_server(
    auth_token: Option<String>,
) -> (
    std::net::SocketAddr,
    TempDir,
    tokio::task::JoinHandle<()>,
) {
    let dir = TempDir::new().unwrap();
    let config = BrokerConfig::default()
        .with_data_dir(dir.path().join("data"))
        .with_log_dir(dir.path().join("logs"))
        .with_default_partitions(1);
    let broker = Arc::new(Broker::new(config).unwrap());
    broker.start().unwrap();

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let opts = ServerOptions {
        max_frame_bytes: 64 * 1024, // small cap so oversized-frame test is cheap
        auth_token,
        offset_dir: dir.path().join("data").join("consumer-offsets"),
    };

    let handle = tokio::spawn(async move {
        let _ = serve_with_listener(broker, listener, opts).await;
    });

    (addr, dir, handle)
}

#[tokio::test]
async fn server_up_and_client_connect_ping() {
    let (addr, _dir, _h) = start_server(None).await;
    let mut client = MqClient::connect(addr, None).await.unwrap();
    client.ping().await.unwrap();
}

#[tokio::test]
async fn produce_then_fetch_round_trips_payload() {
    let (addr, _dir, _h) = start_server(None).await;
    let mut client = MqClient::connect(addr, None).await.unwrap();

    client.create_topic("orders", Some(1)).await.unwrap();
    let (partition, offset) = client
        .produce_message("orders", Some(0), WireMessage::new(b"hello world".to_vec()))
        .await
        .unwrap();
    assert_eq!(partition, 0);
    assert_eq!(offset, 0);

    let msgs = client.fetch("orders", 0, 0, 10, 0).await.unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].payload, b"hello world");
}

#[tokio::test]
async fn multiple_messages_preserve_order_and_offsets() {
    let (addr, _dir, _h) = start_server(None).await;
    let mut client = MqClient::connect(addr, None).await.unwrap();
    client.create_topic("events", Some(1)).await.unwrap();

    for i in 0..5u64 {
        let payload = format!("msg-{}", i).into_bytes();
        let (_, offset) = client
            .produce_message("events", Some(0), WireMessage::new(payload))
            .await
            .unwrap();
        assert_eq!(offset, i);
    }

    let msgs = client.fetch("events", 0, 0, 100, 0).await.unwrap();
    assert_eq!(msgs.len(), 5);
    for (i, m) in msgs.iter().enumerate() {
        assert_eq!(m.offset, i as u64);
        assert_eq!(m.payload, format!("msg-{}", i).into_bytes());
    }
}

#[tokio::test]
async fn create_topic_and_list_topics() {
    let (addr, _dir, _h) = start_server(None).await;
    let mut client = MqClient::connect(addr, None).await.unwrap();

    client.create_topic("alpha", None).await.unwrap();
    client.create_topic("beta", None).await.unwrap();
    // idempotent create should not error
    client.create_topic("alpha", None).await.unwrap();

    let mut topics = client.list_topics().await.unwrap();
    topics.sort();
    assert!(topics.contains(&"alpha".to_string()));
    assert!(topics.contains(&"beta".to_string()));
}

#[tokio::test]
async fn commit_offset_then_fetch_offset() {
    let (addr, _dir, _h) = start_server(None).await;
    let mut client = MqClient::connect(addr, None).await.unwrap();
    client.create_topic("logs", Some(1)).await.unwrap();

    // No offset committed yet.
    let none = client.fetch_offset("group-a", "logs", 0).await.unwrap();
    assert_eq!(none, None);

    client
        .commit_offset("group-a", "logs", 0, 42)
        .await
        .unwrap();
    let committed = client.fetch_offset("group-a", "logs", 0).await.unwrap();
    assert_eq!(committed, Some(42));

    // Different group is independent.
    let other = client.fetch_offset("group-b", "logs", 0).await.unwrap();
    assert_eq!(other, None);
}

#[tokio::test]
async fn fetch_from_missing_topic_returns_typed_error() {
    let (addr, _dir, _h) = start_server(None).await;
    let mut client = MqClient::connect(addr, None).await.unwrap();
    let err = client.fetch("nope", 0, 0, 10, 0).await.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("TopicNotFound") || msg.contains("not found"), "got: {}", msg);
}

#[tokio::test]
async fn oversized_frame_is_rejected() {
    // Server cap is 64 KiB (set in start_server). Send a length prefix that
    // exceeds it and verify the server does not crash and closes the socket.
    let (addr, _dir, _h) = start_server(None).await;
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let oversized_len: u32 = 1_000_000; // ~1 MB > 64 KiB cap
    stream.write_all(&oversized_len.to_be_bytes()).await.unwrap();
    stream.flush().await.unwrap();

    // The server should reply with an error frame and/or close the connection.
    // Either way, a fresh client must still be able to connect and ping,
    // proving the listener survived the bad frame.
    drop(stream);
    let mut client = MqClient::connect(addr, None).await.unwrap();
    client.ping().await.unwrap();
}

#[tokio::test]
async fn auth_required_wrong_token_rejected() {
    let (addr, _dir, _h) = start_server(Some("s3cr3t".to_string())).await;
    let err = MqClient::connect(addr, Some("wrong")).await.unwrap_err();
    assert!(err.to_string().to_lowercase().contains("auth"), "got: {}", err);
}

#[tokio::test]
async fn auth_required_missing_token_rejected() {
    let (addr, _dir, _h) = start_server(Some("s3cr3t".to_string())).await;
    // Connect without sending any auth frame, then send a Ping directly.
    let mut client = MqClient::connect(addr, None).await.unwrap();
    // The first non-auth frame must be rejected; ping should surface an error
    // (either an Unauthorized response or a closed connection).
    let result = client.ping().await;
    assert!(result.is_err(), "expected ping to fail without auth");
}

#[tokio::test]
async fn auth_required_correct_token_works() {
    let (addr, _dir, _h) = start_server(Some("s3cr3t".to_string())).await;
    let mut client = MqClient::connect(addr, Some("s3cr3t")).await.unwrap();
    client.ping().await.unwrap();

    client.create_topic("secure", Some(1)).await.unwrap();
    let (_, offset) = client
        .produce_message("secure", Some(0), WireMessage::new(b"auth ok".to_vec()))
        .await
        .unwrap();
    assert_eq!(offset, 0);
    let msgs = client.fetch("secure", 0, 0, 10, 0).await.unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].payload, b"auth ok");
}

#[tokio::test]
async fn produce_with_key_and_headers_round_trips() {
    let (addr, _dir, _h) = start_server(None).await;
    let mut client = MqClient::connect(addr, None).await.unwrap();
    client.create_topic("kv", Some(1)).await.unwrap();

    let mut wm = WireMessage::new(b"payload".to_vec());
    wm.key = Some(b"my-key".to_vec());
    wm.headers.insert("trace".to_string(), b"abc123".to_vec());

    client.produce_message("kv", Some(0), wm).await.unwrap();
    let msgs = client.fetch("kv", 0, 0, 10, 0).await.unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].key, Some(b"my-key".to_vec()));
    assert_eq!(msgs[0].headers.get("trace"), Some(&b"abc123".to_vec()));
}

#[tokio::test]
async fn malformed_request_keeps_connection_open() {
    // A garbage (undecodable) body should get an error response but not close
    // the connection; a follow-up ping must still succeed.
    let (addr, _dir, _h) = start_server(None).await;
    let mut stream = TcpStream::connect(addr).await.unwrap();

    let garbage = b"\xff\xff\xff\xff\xff\xff";
    let len = garbage.len() as u32;
    stream.write_all(&len.to_be_bytes()).await.unwrap();
    stream.write_all(garbage).await.unwrap();
    stream.flush().await.unwrap();

    // Read and discard the error frame.
    use tokio::io::AsyncReadExt;
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await.unwrap();
    let rlen = u32::from_be_bytes(len_buf) as usize;
    let mut body = vec![0u8; rlen];
    stream.read_exact(&mut body).await.unwrap();

    // Now send a valid Ping on the same connection.
    let ping = message_queue::protocol::encode(&Request::Ping).unwrap();
    let plen = ping.len() as u32;
    stream.write_all(&plen.to_be_bytes()).await.unwrap();
    stream.write_all(&ping).await.unwrap();
    stream.flush().await.unwrap();

    stream.read_exact(&mut len_buf).await.unwrap();
    let rlen = u32::from_be_bytes(len_buf) as usize;
    let mut body = vec![0u8; rlen];
    stream.read_exact(&mut body).await.unwrap();
    let resp: message_queue::Response = message_queue::protocol::decode(&body).unwrap();
    assert!(matches!(resp, message_queue::Response::Pong));
}
