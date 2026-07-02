//! Async client for the message-queue wire protocol.
//!
//! [`MqClient`] connects to a running [`serve`](crate::server::serve) endpoint,
//! performs the optional auth handshake, and offers one method per protocol
//! request. Each call writes a length-prefixed [`Request`] frame and awaits the
//! matching [`Response`] frame. The client is single-connection and issues one
//! request at a time (`&mut self`), which keeps request/response framing
//! unambiguous.

use crate::error::{Error, Result};
use crate::protocol::{
    decode, encode, ErrorKind, Request, Response, WireMessage, DEFAULT_MAX_FRAME_BYTES,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpStream, ToSocketAddrs};

/// A connected message-queue client.
#[derive(Debug)]
pub struct MqClient {
    stream: TcpStream,
    max_frame_bytes: usize,
}

impl MqClient {
    /// Connect to `addr`, optionally authenticating with `token`.
    ///
    /// When the server requires auth, pass `Some(token)`; the handshake is sent
    /// before any other request. When the server does not require auth, passing
    /// a token is harmless (the server accepts a stray `Auth` frame).
    pub async fn connect<A: ToSocketAddrs>(addr: A, token: Option<&str>) -> Result<Self> {
        let stream = TcpStream::connect(addr).await.map_err(Error::Io)?;
        stream.set_nodelay(true).ok();
        let mut client = Self {
            stream,
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
        };

        if let Some(token) = token {
            match client
                .request(&Request::Auth {
                    token: token.to_string(),
                })
                .await?
            {
                Response::AuthOk => {}
                Response::Error { kind, message } => {
                    return Err(Error::Auth(format!("{:?}: {}", kind, message)))
                }
                other => {
                    return Err(Error::Protocol(format!(
                        "unexpected auth response: {:?}",
                        other
                    )))
                }
            }
        }

        Ok(client)
    }

    /// Send a request and read the raw response frame.
    async fn request(&mut self, req: &Request) -> Result<Response> {
        let body = encode(req)?;
        let len = body.len() as u32;
        self.stream
            .write_all(&len.to_be_bytes())
            .await
            .map_err(Error::Io)?;
        self.stream.write_all(&body).await.map_err(Error::Io)?;
        self.stream.flush().await.map_err(Error::Io)?;

        let mut len_buf = [0u8; 4];
        self.stream
            .read_exact(&mut len_buf)
            .await
            .map_err(Error::Io)?;
        let resp_len = u32::from_be_bytes(len_buf) as usize;
        if resp_len > self.max_frame_bytes {
            return Err(Error::Protocol(format!(
                "response frame too large: {} bytes",
                resp_len
            )));
        }
        let mut resp_body = vec![0u8; resp_len];
        self.stream
            .read_exact(&mut resp_body)
            .await
            .map_err(Error::Io)?;
        decode(&resp_body)
    }

    /// Convert a `Response::Error` into a crate `Error::Server`.
    fn server_error(kind: ErrorKind, message: String) -> Error {
        Error::Server {
            kind: format!("{:?}", kind),
            message,
        }
    }

    /// Send a health-check ping.
    pub async fn ping(&mut self) -> Result<()> {
        match self.request(&Request::Ping).await? {
            Response::Pong => Ok(()),
            Response::Error { kind, message } => Err(Self::server_error(kind, message)),
            other => Err(Error::Protocol(format!("unexpected response: {:?}", other))),
        }
    }

    /// Create a topic (idempotent). `partitions` uses the server default when `None`.
    pub async fn create_topic(&mut self, name: &str, partitions: Option<u32>) -> Result<()> {
        match self
            .request(&Request::CreateTopic {
                name: name.to_string(),
                partitions,
            })
            .await?
        {
            Response::TopicCreated { .. } => Ok(()),
            Response::Error { kind, message } => Err(Self::server_error(kind, message)),
            other => Err(Error::Protocol(format!("unexpected response: {:?}", other))),
        }
    }

    /// List all topic names.
    pub async fn list_topics(&mut self) -> Result<Vec<String>> {
        match self.request(&Request::ListTopics).await? {
            Response::Topics { names } => Ok(names),
            Response::Error { kind, message } => Err(Self::server_error(kind, message)),
            other => Err(Error::Protocol(format!("unexpected response: {:?}", other))),
        }
    }

    /// Produce a raw payload, letting the broker choose the partition.
    /// Returns `(partition, offset)`.
    pub async fn produce(&mut self, topic: &str, payload: impl Into<Vec<u8>>) -> Result<(u32, u64)> {
        self.produce_message(topic, None, WireMessage::new(payload))
            .await
    }

    /// Produce a fully-specified [`WireMessage`], optionally to a fixed partition.
    /// Returns `(partition, offset)`.
    pub async fn produce_message(
        &mut self,
        topic: &str,
        partition: Option<u32>,
        message: WireMessage,
    ) -> Result<(u32, u64)> {
        match self
            .request(&Request::Produce {
                topic: topic.to_string(),
                partition,
                message,
            })
            .await?
        {
            Response::Produced { partition, offset } => Ok((partition, offset)),
            Response::Error { kind, message } => Err(Self::server_error(kind, message)),
            other => Err(Error::Protocol(format!("unexpected response: {:?}", other))),
        }
    }

    /// Fetch up to `max_messages` messages from `topic`/`partition` starting at `offset`.
    /// `max_bytes` of `0` means unbounded.
    pub async fn fetch(
        &mut self,
        topic: &str,
        partition: u32,
        offset: u64,
        max_messages: u32,
        max_bytes: u32,
    ) -> Result<Vec<WireMessage>> {
        match self
            .request(&Request::Fetch {
                topic: topic.to_string(),
                partition,
                offset,
                max_messages,
                max_bytes,
            })
            .await?
        {
            Response::Fetched { messages } => Ok(messages),
            Response::Error { kind, message } => Err(Self::server_error(kind, message)),
            other => Err(Error::Protocol(format!("unexpected response: {:?}", other))),
        }
    }

    /// Commit a consumer-group offset for a topic-partition.
    pub async fn commit_offset(
        &mut self,
        group: &str,
        topic: &str,
        partition: u32,
        offset: u64,
    ) -> Result<()> {
        match self
            .request(&Request::CommitOffset {
                group: group.to_string(),
                topic: topic.to_string(),
                partition,
                offset,
            })
            .await?
        {
            Response::OffsetCommitted => Ok(()),
            Response::Error { kind, message } => Err(Self::server_error(kind, message)),
            other => Err(Error::Protocol(format!("unexpected response: {:?}", other))),
        }
    }

    /// Fetch a committed consumer-group offset. `None` means never committed.
    pub async fn fetch_offset(
        &mut self,
        group: &str,
        topic: &str,
        partition: u32,
    ) -> Result<Option<u64>> {
        match self
            .request(&Request::FetchOffset {
                group: group.to_string(),
                topic: topic.to_string(),
                partition,
            })
            .await?
        {
            Response::FetchedOffset { offset } => Ok(offset),
            Response::Error { kind, message } => Err(Self::server_error(kind, message)),
            other => Err(Error::Protocol(format!("unexpected response: {:?}", other))),
        }
    }
}
