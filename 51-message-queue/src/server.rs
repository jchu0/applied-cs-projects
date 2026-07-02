//! Tokio TCP server that exposes a [`Broker`] over the wire protocol.
//!
//! The server accepts connections, reads length-prefixed [`Request`] frames,
//! dispatches them to a shared [`Broker`], and writes framed [`Response`]s
//! back. It is the network layer that turns the in-process broker into an
//! actual, connectable message queue.
//!
//! Design notes:
//! - Broker calls are synchronous and operate on in-memory / local-file state,
//!   so they are fast and non-blocking in practice. We call them directly from
//!   the async task rather than paying the overhead of `spawn_blocking` on
//!   every request. (If the storage layer ever grows genuinely blocking calls,
//!   individual handlers can be wrapped in `spawn_blocking`.)
//! - One malformed frame closes only that connection; the accept loop keeps
//!   running. A per-frame size cap rejects oversized frames before allocation.
//! - Consumer-group offsets are tracked by the server (the broker deliberately
//!   does not expose offset commit/fetch) using the crate's file-backed
//!   [`OffsetStore`], so the broker's semantics are left untouched.

use crate::broker::Broker;
use crate::error::{Error, Result};
use crate::offset::{OffsetStore, TopicPartition};
use crate::protocol::{
    decode, encode, ErrorKind, Request, Response, WireMessage, DEFAULT_MAX_FRAME_BYTES,
};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream, ToSocketAddrs};

/// Options controlling server behavior.
#[derive(Debug, Clone)]
pub struct ServerOptions {
    /// Maximum accepted frame body size, in bytes. Larger frames are rejected.
    pub max_frame_bytes: usize,
    /// Optional shared-secret auth token. When `Some`, the first frame on each
    /// connection must be `Request::Auth` with a matching token.
    pub auth_token: Option<String>,
    /// Directory under which consumer-group offsets are persisted.
    pub offset_dir: PathBuf,
}

impl ServerOptions {
    /// Build options rooted at a broker data directory, reading the auth token
    /// from the `MQ_AUTH_TOKEN` environment variable if set.
    pub fn from_env(data_dir: impl Into<PathBuf>) -> Self {
        let data_dir = data_dir.into();
        let auth_token = std::env::var("MQ_AUTH_TOKEN")
            .ok()
            .filter(|t| !t.is_empty());
        Self {
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
            auth_token,
            offset_dir: data_dir.join("consumer-offsets"),
        }
    }
}

/// Server-side registry of consumer-group offset stores.
///
/// Each group gets one file-backed [`OffsetStore`]; commits are flushed
/// immediately so they survive restarts.
struct OffsetRegistry {
    dir: PathBuf,
    stores: Mutex<HashMap<String, Arc<OffsetStore>>>,
}

impl OffsetRegistry {
    fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            stores: Mutex::new(HashMap::new()),
        }
    }

    fn store_for(&self, group: &str) -> Result<Arc<OffsetStore>> {
        let mut stores = self.stores.lock();
        if let Some(existing) = stores.get(group) {
            return Ok(existing.clone());
        }
        let store = Arc::new(OffsetStore::new(group, &self.dir)?);
        stores.insert(group.to_string(), store.clone());
        Ok(store)
    }

    fn commit(&self, group: &str, tp: TopicPartition, offset: u64) -> Result<()> {
        let store = self.store_for(group)?;
        store.commit(tp, offset)?;
        store.flush()
    }

    fn fetch(&self, group: &str, tp: &TopicPartition) -> Result<Option<u64>> {
        let store = self.store_for(group)?;
        Ok(store.get_offset(tp))
    }
}

/// Shared, cheaply-clonable server state handed to each connection task.
#[derive(Clone)]
struct ServerState {
    broker: Arc<Broker>,
    offsets: Arc<OffsetRegistry>,
    opts: Arc<ServerOptions>,
}

/// Serve the wire protocol for `broker` on `addr` until the process exits.
///
/// Binds a TCP listener and dispatches each connection to its own task. Returns
/// an error only if binding fails; per-connection errors are logged and do not
/// stop the accept loop.
pub async fn serve<A: ToSocketAddrs>(
    broker: Arc<Broker>,
    addr: A,
    opts: ServerOptions,
) -> Result<()> {
    let listener = TcpListener::bind(addr).await?;
    let local = listener.local_addr()?;
    tracing::info!("mq wire server listening on {}", local);
    serve_with_listener(broker, listener, opts).await
}

/// Serve using an already-bound [`TcpListener`].
///
/// Useful for tests that bind to an ephemeral port (`127.0.0.1:0`) and need the
/// concrete address before the loop starts.
pub async fn serve_with_listener(
    broker: Arc<Broker>,
    listener: TcpListener,
    opts: ServerOptions,
) -> Result<()> {
    std::fs::create_dir_all(&opts.offset_dir)?;
    let state = ServerState {
        broker,
        offsets: Arc::new(OffsetRegistry::new(opts.offset_dir.clone())),
        opts: Arc::new(opts),
    };

    loop {
        let (socket, peer) = match listener.accept().await {
            Ok(pair) => pair,
            Err(e) => {
                // Transient accept errors (e.g. per-process fd limits) must not
                // kill the listener.
                tracing::warn!("accept error: {}", e);
                continue;
            }
        };
        let state = state.clone();
        tokio::spawn(async move {
            state.broker.metrics().connection_opened();
            if let Err(e) = handle_connection(socket, peer, &state).await {
                tracing::debug!("connection {} closed: {}", peer, e);
            }
            state.broker.metrics().connection_closed();
        });
    }
}

/// Read one length-prefixed frame body. Returns `Ok(None)` on a clean EOF at a
/// frame boundary (peer hung up), `Err` on protocol/IO errors.
async fn read_frame(stream: &mut TcpStream, max: usize) -> Result<Option<Vec<u8>>> {
    let mut len_buf = [0u8; 4];
    match stream.read_exact(&mut len_buf).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(Error::Io(e)),
    }
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > max {
        return Err(Error::Protocol(format!(
            "frame too large: {} bytes (max {})",
            len, max
        )));
    }
    let mut body = vec![0u8; len];
    stream
        .read_exact(&mut body)
        .await
        .map_err(Error::Io)?;
    Ok(Some(body))
}

/// Write one length-prefixed frame body.
async fn write_frame(stream: &mut TcpStream, body: &[u8]) -> Result<()> {
    let len = body.len() as u32;
    stream.write_all(&len.to_be_bytes()).await.map_err(Error::Io)?;
    stream.write_all(body).await.map_err(Error::Io)?;
    stream.flush().await.map_err(Error::Io)?;
    Ok(())
}

/// Encode and write a [`Response`] frame.
async fn write_response(stream: &mut TcpStream, resp: &Response) -> Result<()> {
    let body = encode(resp)?;
    write_frame(stream, &body).await
}

/// Constant-time byte comparison for tokens.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Handle a single connection: optional auth handshake, then a request loop.
async fn handle_connection(
    mut socket: TcpStream,
    peer: SocketAddr,
    state: &ServerState,
) -> Result<()> {
    socket.set_nodelay(true).ok();
    let max = state.opts.max_frame_bytes;

    // Auth handshake, if configured.
    if let Some(expected) = state.opts.auth_token.as_deref() {
        let body = match read_frame(&mut socket, max).await? {
            Some(b) => b,
            None => return Ok(()), // peer disconnected before authenticating
        };
        let ok = match decode::<Request>(&body) {
            Ok(Request::Auth { token }) => {
                constant_time_eq(token.as_bytes(), expected.as_bytes())
            }
            _ => false,
        };
        if ok {
            write_response(&mut socket, &Response::AuthOk).await?;
        } else {
            let _ = write_response(
                &mut socket,
                &Response::error(ErrorKind::Unauthorized, "authentication required"),
            )
            .await;
            return Err(Error::Auth(format!("{} failed authentication", peer)));
        }
    }

    // Request loop.
    loop {
        let body = match read_frame(&mut socket, max).await {
            Ok(Some(b)) => b,
            Ok(None) => return Ok(()), // clean disconnect
            Err(Error::Protocol(msg)) => {
                // Bad frame: report it, then close this connection only.
                let _ = write_response(
                    &mut socket,
                    &Response::error(ErrorKind::InvalidRequest, msg.clone()),
                )
                .await;
                return Err(Error::Protocol(msg));
            }
            Err(e) => return Err(e),
        };

        let request: Request = match decode(&body) {
            Ok(req) => req,
            Err(e) => {
                // Undecodable body: reply with an error and keep the connection
                // open so a single bad request does not drop the client.
                write_response(
                    &mut socket,
                    &Response::error(
                        ErrorKind::InvalidRequest,
                        format!("malformed request: {}", e),
                    ),
                )
                .await?;
                continue;
            }
        };

        let response = dispatch(request, state);
        if matches!(response, Response::Error { .. }) {
            state.broker.metrics().record_failure();
        }
        write_response(&mut socket, &response).await?;
    }
}

/// Dispatch a decoded request against the shared broker/offset state.
fn dispatch(request: Request, state: &ServerState) -> Response {
    match request {
        // A stray Auth frame after the handshake is a harmless no-op.
        Request::Auth { .. } => Response::AuthOk,
        Request::Ping => Response::Pong,
        Request::CreateTopic { name, partitions } => {
            let config = partitions.map(|p| crate::config::TopicConfig {
                partition_count: p,
                ..crate::config::TopicConfig::default()
            });
            match state.broker.create_topic(&name, config) {
                Ok(_) => Response::TopicCreated { name },
                // Treat "already exists" as success (idempotent create).
                Err(Error::TopicAlreadyExists(_)) => Response::TopicCreated { name },
                Err(e) => Response::from_error(&e),
            }
        }
        Request::ListTopics => Response::Topics {
            names: state.broker.list_topics(),
        },
        Request::Produce {
            topic,
            partition,
            message,
        } => {
            let msg = message.into_message();
            let result = match partition {
                Some(p) => state
                    .broker
                    .produce_to_partition(&topic, p, msg)
                    .map(|offset| (p, offset)),
                None => state.broker.produce(&topic, msg),
            };
            match result {
                Ok((partition, offset)) => Response::Produced { partition, offset },
                Err(e) => Response::from_error(&e),
            }
        }
        Request::Fetch {
            topic,
            partition,
            offset,
            max_messages,
            max_bytes,
        } => {
            match state.broker.fetch(
                &topic,
                partition,
                offset,
                max_messages as usize,
                max_bytes as usize,
            ) {
                Ok(messages) => Response::Fetched {
                    messages: messages.iter().map(WireMessage::from_message).collect(),
                },
                Err(e) => Response::from_error(&e),
            }
        }
        Request::CommitOffset {
            group,
            topic,
            partition,
            offset,
        } => {
            let tp = TopicPartition::new(topic, partition);
            match state.offsets.commit(&group, tp, offset) {
                Ok(()) => Response::OffsetCommitted,
                Err(e) => Response::from_error(&e),
            }
        }
        Request::FetchOffset {
            group,
            topic,
            partition,
        } => {
            let tp = TopicPartition::new(topic, partition);
            match state.offsets.fetch(&group, &tp) {
                Ok(offset) => Response::FetchedOffset { offset },
                Err(e) => Response::from_error(&e),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_time_eq_matches() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreu"));
        assert!(!constant_time_eq(b"secret", b"secre"));
        assert!(!constant_time_eq(b"", b"x"));
        assert!(constant_time_eq(b"", b""));
    }
}
