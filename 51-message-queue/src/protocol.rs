//! Wire protocol for the message-queue server.
//!
//! The protocol is a simple length-prefixed framing over TCP:
//!
//! ```text
//! +-------------------+-----------------------------+
//! | u32 length (BE)   | body (bincode-serialized)   |
//! +-------------------+-----------------------------+
//! ```
//!
//! The `length` prefix is the number of bytes in the body. Both requests and
//! responses share this framing. Bodies are serialized with [`bincode`], which
//! is already a dependency of this crate and produces compact binary output.
//!
//! [`Request`] and [`Response`] are the top-level enums exchanged over the
//! wire. To keep the protocol decoupled from the broker's internal
//! representation (`Message` holds `Bytes`/`MessageId`/`Headers`), messages are
//! carried as [`WireMessage`], a plain serde-friendly record.

use crate::error::Error;
use crate::message::{Headers, Message};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Maximum frame body size accepted or produced by default (16 MiB).
///
/// Frames larger than this are rejected before allocation to avoid
/// memory-exhaustion from malicious or buggy peers.
pub const DEFAULT_MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// Protocol version, bumped on incompatible wire changes.
pub const PROTOCOL_VERSION: u16 = 1;

/// A message as carried on the wire.
///
/// This is intentionally a flat, owned structure so the protocol does not
/// depend on the broker's internal `Message` layout. It round-trips to and
/// from [`Message`] via [`WireMessage::from_message`] / [`WireMessage::into_message`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireMessage {
    /// Optional partitioning/dedup key.
    pub key: Option<Vec<u8>>,
    /// Message payload.
    pub payload: Vec<u8>,
    /// Message headers (string key -> raw bytes).
    pub headers: HashMap<String, Vec<u8>>,
    /// Timestamp in milliseconds since the Unix epoch. `0` means "assign now".
    pub timestamp: u64,
    /// Offset within the partition (populated on fetch; ignored on produce).
    pub offset: u64,
    /// Partition id (populated on fetch; ignored on produce).
    pub partition: u32,
}

impl WireMessage {
    /// Build a wire message from a payload with no key or headers.
    pub fn new(payload: impl Into<Vec<u8>>) -> Self {
        Self {
            key: None,
            payload: payload.into(),
            headers: HashMap::new(),
            timestamp: 0,
            offset: 0,
            partition: 0,
        }
    }

    /// Convert an internal broker [`Message`] into a wire message.
    pub fn from_message(msg: &Message) -> Self {
        Self {
            key: msg.key.as_ref().map(|k| k.to_vec()),
            payload: msg.payload.to_vec(),
            headers: msg
                .headers
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            timestamp: msg.timestamp,
            offset: msg.offset,
            partition: msg.partition,
        }
    }

    /// Convert this wire message into an internal broker [`Message`].
    ///
    /// A fresh `MessageId` is generated and, when `timestamp` is `0`, the
    /// current time is used (matching `Message::new` semantics).
    pub fn into_message(self) -> Message {
        let mut msg = match self.key {
            Some(key) => Message::with_key(key, self.payload),
            None => Message::new(self.payload),
        };
        if self.timestamp != 0 {
            msg.timestamp = self.timestamp;
        }
        if !self.headers.is_empty() {
            msg.headers = Headers::from(self.headers);
        }
        msg
    }
}

/// A request sent from a client to the server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Request {
    /// Authentication handshake. Must be the first frame when the server
    /// requires auth. Ignored (but accepted) when auth is disabled.
    Auth {
        /// Shared secret token.
        token: String,
    },
    /// Liveness / handshake check.
    Ping,
    /// Create a topic with an optional partition count.
    CreateTopic {
        /// Topic name.
        name: String,
        /// Desired partition count (server default when `None`).
        partitions: Option<u32>,
    },
    /// List all known topic names.
    ListTopics,
    /// Produce a single message. When `partition` is `None` the broker's
    /// partitioner selects the partition.
    Produce {
        /// Target topic.
        topic: String,
        /// Optional explicit partition.
        partition: Option<u32>,
        /// The message to append.
        message: WireMessage,
    },
    /// Fetch messages from a partition starting at `offset`.
    Fetch {
        /// Source topic.
        topic: String,
        /// Source partition.
        partition: u32,
        /// Starting offset.
        offset: u64,
        /// Maximum number of messages to return.
        max_messages: u32,
        /// Maximum total bytes to return (`0` = unbounded).
        max_bytes: u32,
    },
    /// Commit a consumer-group offset for a topic-partition.
    CommitOffset {
        /// Consumer group id.
        group: String,
        /// Topic name.
        topic: String,
        /// Partition id.
        partition: u32,
        /// Offset to commit.
        offset: u64,
    },
    /// Fetch a committed consumer-group offset for a topic-partition.
    FetchOffset {
        /// Consumer group id.
        group: String,
        /// Topic name.
        topic: String,
        /// Partition id.
        partition: u32,
    },
}

/// A response returned from the server to a client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Response {
    /// Authentication accepted.
    AuthOk,
    /// Reply to [`Request::Ping`].
    Pong,
    /// Topic created (or already existed).
    TopicCreated {
        /// Topic name.
        name: String,
    },
    /// Reply to [`Request::ListTopics`].
    Topics {
        /// Known topic names.
        names: Vec<String>,
    },
    /// Reply to [`Request::Produce`] with the assigned coordinates.
    Produced {
        /// Partition the message landed in.
        partition: u32,
        /// Offset assigned to the message.
        offset: u64,
    },
    /// Reply to [`Request::Fetch`].
    Fetched {
        /// The messages read (may be empty).
        messages: Vec<WireMessage>,
    },
    /// Reply to [`Request::CommitOffset`].
    OffsetCommitted,
    /// Reply to [`Request::FetchOffset`]. `offset` is `None` when the group has
    /// never committed for that topic-partition.
    FetchedOffset {
        /// The committed offset, if any.
        offset: Option<u64>,
    },
    /// A typed error reply for any failed request.
    Error {
        /// Machine-readable error kind.
        kind: ErrorKind,
        /// Human-readable detail.
        message: String,
    },
}

/// A coarse, wire-stable classification of a server-side error.
///
/// This intentionally does not mirror every [`Error`] variant; it groups them
/// into categories a client can reasonably branch on while remaining forward
/// compatible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorKind {
    /// The requested topic does not exist.
    TopicNotFound,
    /// The requested partition does not exist.
    PartitionNotFound,
    /// The requested offset is outside the valid range.
    OffsetOutOfRange,
    /// The request was malformed or violated an invariant.
    InvalidRequest,
    /// Authentication was required and failed.
    Unauthorized,
    /// A storage/IO error occurred.
    Storage,
    /// Anything not covered above.
    Internal,
}

impl Response {
    /// Build an [`Response::Error`] from a broker [`Error`], mapping it to the
    /// closest [`ErrorKind`].
    pub fn from_error(err: &Error) -> Self {
        let kind = match err {
            Error::TopicNotFound(_) => ErrorKind::TopicNotFound,
            Error::PartitionNotFound { .. } => ErrorKind::PartitionNotFound,
            Error::OffsetOutOfRange { .. } => ErrorKind::OffsetOutOfRange,
            Error::InvalidConfig(_)
            | Error::InvalidPartitionCount(_)
            | Error::MessageTooLarge { .. } => ErrorKind::InvalidRequest,
            Error::Io(_) | Error::StorageFull | Error::SegmentCorrupted(_) => ErrorKind::Storage,
            _ => ErrorKind::Internal,
        };
        Response::Error {
            kind,
            message: err.to_string(),
        }
    }

    /// Build an [`Response::Error`] with an explicit kind and message.
    pub fn error(kind: ErrorKind, message: impl Into<String>) -> Self {
        Response::Error {
            kind,
            message: message.into(),
        }
    }
}

/// Serialize a value into a bincode body.
pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, Error> {
    bincode::serialize(value).map_err(Error::from)
}

/// Deserialize a value from a bincode body.
pub fn decode<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Result<T, Error> {
    bincode::deserialize(bytes).map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_message_round_trips_through_message() {
        let mut wm = WireMessage::new(b"hello".to_vec());
        wm.key = Some(b"k".to_vec());
        wm.headers.insert("h1".to_string(), b"v1".to_vec());

        let msg = wm.clone().into_message();
        assert_eq!(msg.payload.as_ref(), b"hello");
        assert_eq!(msg.key.as_ref().unwrap().as_ref(), b"k");
        assert_eq!(msg.headers.get("h1"), Some(&b"v1"[..]));

        let back = WireMessage::from_message(&msg);
        assert_eq!(back.payload, b"hello");
        assert_eq!(back.key, Some(b"k".to_vec()));
        assert_eq!(back.headers.get("h1"), Some(&b"v1".to_vec()));
    }

    #[test]
    fn request_bincode_round_trip() {
        let req = Request::Produce {
            topic: "t".to_string(),
            partition: Some(2),
            message: WireMessage::new(b"payload".to_vec()),
        };
        let bytes = encode(&req).unwrap();
        let decoded: Request = decode(&bytes).unwrap();
        assert_eq!(req, decoded);
    }

    #[test]
    fn response_bincode_round_trip() {
        let resp = Response::Fetched {
            messages: vec![WireMessage::new(b"a".to_vec()), WireMessage::new(b"b".to_vec())],
        };
        let bytes = encode(&resp).unwrap();
        let decoded: Response = decode(&bytes).unwrap();
        assert_eq!(resp, decoded);
    }

    #[test]
    fn error_mapping() {
        let e = Error::TopicNotFound("x".to_string());
        match Response::from_error(&e) {
            Response::Error { kind, .. } => assert_eq!(kind, ErrorKind::TopicNotFound),
            _ => panic!("expected error response"),
        }
    }
}
