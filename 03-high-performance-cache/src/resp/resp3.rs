//! RESP3 Protocol Support
//!
//! RESP3 extends RESP2 with additional data types:
//! - Null: `_\r\n`
//! - Boolean: `#t\r\n` or `#f\r\n`
//! - Double: `,1.23\r\n`
//! - Big number: `(3492890328409238509324850943850943825024385\r\n`
//! - Verbatim string: `=15\r\ntxt:Some string\r\n`
//! - Map: `%2\r\n+first\r\n:1\r\n+second\r\n:2\r\n`
//! - Set: `~3\r\n+orange\r\n+apple\r\n+banana\r\n`
//! - Attribute: `|1\r\n+key\r\n+value\r\n` (metadata)
//! - Push: `>3\r\n+pubsub\r\n+message\r\n+data\r\n` (async messages)

use bytes::{BufMut, BytesMut};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// RESP3 value types (superset of RESP2)
#[derive(Debug, Clone, PartialEq)]
pub enum Resp3Value {
    // RESP2 types
    /// Simple strings: +OK\r\n
    SimpleString(String),
    /// Errors: -ERR message\r\n
    Error(String),
    /// Integers: :1000\r\n
    Integer(i64),
    /// Bulk strings: $5\r\nhello\r\n
    BulkString(Option<Vec<u8>>),
    /// Arrays: *2\r\n...
    Array(Option<Vec<Resp3Value>>),

    // RESP3-only types
    /// Null: _\r\n
    Null,
    /// Boolean: #t\r\n or #f\r\n
    Boolean(bool),
    /// Double: ,1.23\r\n
    Double(f64),
    /// Big number: (12345678901234567890\r\n
    BigNumber(String),
    /// Verbatim string: =15\r\ntxt:Some string\r\n
    VerbatimString { encoding: String, data: Vec<u8> },
    /// Map: %2\r\n+key1\r\n:val1\r\n+key2\r\n:val2\r\n
    Map(BTreeMap<String, Resp3Value>),
    /// Set: ~3\r\n+item1\r\n+item2\r\n+item3\r\n
    Set(BTreeSet<String>),
    /// Attribute (metadata for next value): |1\r\n...
    Attribute {
        data: BTreeMap<String, Resp3Value>,
        value: Box<Resp3Value>,
    },
    /// Push (out-of-band data): >3\r\n...
    Push(Vec<Resp3Value>),
    /// Blob error: !21\r\nSYNTAX invalid syntax\r\n
    BlobError(Vec<u8>),
}

impl Resp3Value {
    /// Serialize the value to RESP3 protocol bytes
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = BytesMut::with_capacity(64);
        self.serialize_into(&mut buf);
        buf.to_vec()
    }

    /// Serialize into a buffer
    pub fn serialize_into(&self, buf: &mut BytesMut) {
        match self {
            // RESP2 compatible types
            Resp3Value::SimpleString(s) => {
                buf.put_u8(b'+');
                buf.put_slice(s.as_bytes());
                buf.put_slice(b"\r\n");
            }
            Resp3Value::Error(s) => {
                buf.put_u8(b'-');
                buf.put_slice(s.as_bytes());
                buf.put_slice(b"\r\n");
            }
            Resp3Value::Integer(i) => {
                buf.put_u8(b':');
                buf.put_slice(i.to_string().as_bytes());
                buf.put_slice(b"\r\n");
            }
            Resp3Value::BulkString(None) => {
                buf.put_slice(b"$-1\r\n");
            }
            Resp3Value::BulkString(Some(data)) => {
                buf.put_u8(b'$');
                buf.put_slice(data.len().to_string().as_bytes());
                buf.put_slice(b"\r\n");
                buf.put_slice(data);
                buf.put_slice(b"\r\n");
            }
            Resp3Value::Array(None) => {
                buf.put_slice(b"*-1\r\n");
            }
            Resp3Value::Array(Some(arr)) => {
                buf.put_u8(b'*');
                buf.put_slice(arr.len().to_string().as_bytes());
                buf.put_slice(b"\r\n");
                for item in arr {
                    item.serialize_into(buf);
                }
            }

            // RESP3-only types
            Resp3Value::Null => {
                buf.put_slice(b"_\r\n");
            }
            Resp3Value::Boolean(b) => {
                buf.put_u8(b'#');
                buf.put_u8(if *b { b't' } else { b'f' });
                buf.put_slice(b"\r\n");
            }
            Resp3Value::Double(d) => {
                buf.put_u8(b',');
                if d.is_infinite() {
                    if *d > 0.0 {
                        buf.put_slice(b"inf");
                    } else {
                        buf.put_slice(b"-inf");
                    }
                } else if d.is_nan() {
                    buf.put_slice(b"nan");
                } else {
                    buf.put_slice(d.to_string().as_bytes());
                }
                buf.put_slice(b"\r\n");
            }
            Resp3Value::BigNumber(n) => {
                buf.put_u8(b'(');
                buf.put_slice(n.as_bytes());
                buf.put_slice(b"\r\n");
            }
            Resp3Value::VerbatimString { encoding, data } => {
                buf.put_u8(b'=');
                let total_len = encoding.len() + 1 + data.len(); // encoding + ':' + data
                buf.put_slice(total_len.to_string().as_bytes());
                buf.put_slice(b"\r\n");
                buf.put_slice(encoding.as_bytes());
                buf.put_u8(b':');
                buf.put_slice(data);
                buf.put_slice(b"\r\n");
            }
            Resp3Value::Map(map) => {
                buf.put_u8(b'%');
                buf.put_slice(map.len().to_string().as_bytes());
                buf.put_slice(b"\r\n");
                for (key, value) in map {
                    // Keys are simple strings
                    buf.put_u8(b'+');
                    buf.put_slice(key.as_bytes());
                    buf.put_slice(b"\r\n");
                    value.serialize_into(buf);
                }
            }
            Resp3Value::Set(set) => {
                buf.put_u8(b'~');
                buf.put_slice(set.len().to_string().as_bytes());
                buf.put_slice(b"\r\n");
                for item in set {
                    buf.put_u8(b'+');
                    buf.put_slice(item.as_bytes());
                    buf.put_slice(b"\r\n");
                }
            }
            Resp3Value::Attribute { data, value } => {
                buf.put_u8(b'|');
                buf.put_slice(data.len().to_string().as_bytes());
                buf.put_slice(b"\r\n");
                for (key, val) in data {
                    buf.put_u8(b'+');
                    buf.put_slice(key.as_bytes());
                    buf.put_slice(b"\r\n");
                    val.serialize_into(buf);
                }
                value.serialize_into(buf);
            }
            Resp3Value::Push(items) => {
                buf.put_u8(b'>');
                buf.put_slice(items.len().to_string().as_bytes());
                buf.put_slice(b"\r\n");
                for item in items {
                    item.serialize_into(buf);
                }
            }
            Resp3Value::BlobError(data) => {
                buf.put_u8(b'!');
                buf.put_slice(data.len().to_string().as_bytes());
                buf.put_slice(b"\r\n");
                buf.put_slice(data);
                buf.put_slice(b"\r\n");
            }
        }
    }

    // Constructors
    pub fn ok() -> Self {
        Resp3Value::SimpleString("OK".to_string())
    }

    pub fn null() -> Self {
        Resp3Value::Null
    }

    pub fn error(msg: impl Into<String>) -> Self {
        Resp3Value::Error(msg.into())
    }

    pub fn integer(i: i64) -> Self {
        Resp3Value::Integer(i)
    }

    pub fn boolean(b: bool) -> Self {
        Resp3Value::Boolean(b)
    }

    pub fn double(d: f64) -> Self {
        Resp3Value::Double(d)
    }

    pub fn bulk(data: impl Into<Vec<u8>>) -> Self {
        Resp3Value::BulkString(Some(data.into()))
    }

    pub fn bulk_string(s: impl Into<String>) -> Self {
        Resp3Value::BulkString(Some(s.into().into_bytes()))
    }

    pub fn array(items: Vec<Resp3Value>) -> Self {
        Resp3Value::Array(Some(items))
    }

    pub fn map(items: BTreeMap<String, Resp3Value>) -> Self {
        Resp3Value::Map(items)
    }

    pub fn set(items: BTreeSet<String>) -> Self {
        Resp3Value::Set(items)
    }

    pub fn push(items: Vec<Resp3Value>) -> Self {
        Resp3Value::Push(items)
    }

    pub fn verbatim(encoding: impl Into<String>, data: impl Into<Vec<u8>>) -> Self {
        Resp3Value::VerbatimString {
            encoding: encoding.into(),
            data: data.into(),
        }
    }

    /// Check if this is a null value
    pub fn is_null(&self) -> bool {
        matches!(
            self,
            Resp3Value::Null | Resp3Value::BulkString(None) | Resp3Value::Array(None)
        )
    }

    /// Convert to string if possible
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Resp3Value::SimpleString(s) => Some(s),
            Resp3Value::BulkString(Some(data)) => std::str::from_utf8(data).ok(),
            _ => None,
        }
    }

    /// Convert to integer if possible
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Resp3Value::Integer(i) => Some(*i),
            Resp3Value::BulkString(Some(data)) => {
                std::str::from_utf8(data).ok()?.parse().ok()
            }
            _ => None,
        }
    }

    /// Convert to double if possible
    pub fn as_double(&self) -> Option<f64> {
        match self {
            Resp3Value::Double(d) => Some(*d),
            Resp3Value::Integer(i) => Some(*i as f64),
            Resp3Value::BulkString(Some(data)) => {
                std::str::from_utf8(data).ok()?.parse().ok()
            }
            _ => None,
        }
    }

    /// Convert to boolean if possible
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Resp3Value::Boolean(b) => Some(*b),
            Resp3Value::Integer(i) => Some(*i != 0),
            _ => None,
        }
    }
}

impl fmt::Display for Resp3Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Resp3Value::SimpleString(s) => write!(f, "{}", s),
            Resp3Value::Error(s) => write!(f, "(error) {}", s),
            Resp3Value::Integer(i) => write!(f, "(integer) {}", i),
            Resp3Value::BulkString(None) => write!(f, "(nil)"),
            Resp3Value::BulkString(Some(data)) => {
                if let Ok(s) = std::str::from_utf8(data) {
                    write!(f, "\"{}\"", s)
                } else {
                    write!(f, "{:?}", data)
                }
            }
            Resp3Value::Array(None) => write!(f, "(nil)"),
            Resp3Value::Array(Some(arr)) => {
                if arr.is_empty() {
                    write!(f, "(empty array)")
                } else {
                    for (i, item) in arr.iter().enumerate() {
                        if i > 0 {
                            writeln!(f)?;
                        }
                        write!(f, "{}) {}", i + 1, item)?;
                    }
                    Ok(())
                }
            }
            Resp3Value::Null => write!(f, "(nil)"),
            Resp3Value::Boolean(b) => write!(f, "(boolean) {}", b),
            Resp3Value::Double(d) => write!(f, "(double) {}", d),
            Resp3Value::BigNumber(n) => write!(f, "(big number) {}", n),
            Resp3Value::VerbatimString { encoding, data } => {
                if let Ok(s) = std::str::from_utf8(data) {
                    write!(f, "(verbatim {}) \"{}\"", encoding, s)
                } else {
                    write!(f, "(verbatim {}) {:?}", encoding, data)
                }
            }
            Resp3Value::Map(map) => {
                writeln!(f, "(map)")?;
                for (key, value) in map {
                    writeln!(f, "  {} => {}", key, value)?;
                }
                Ok(())
            }
            Resp3Value::Set(set) => {
                writeln!(f, "(set)")?;
                for item in set {
                    writeln!(f, "  {}", item)?;
                }
                Ok(())
            }
            Resp3Value::Attribute { data: _, value } => {
                write!(f, "{}", value)
            }
            Resp3Value::Push(items) => {
                write!(f, "(push) ")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                Ok(())
            }
            Resp3Value::BlobError(data) => {
                if let Ok(s) = std::str::from_utf8(data) {
                    write!(f, "(blob error) {}", s)
                } else {
                    write!(f, "(blob error) {:?}", data)
                }
            }
        }
    }
}

/// Protocol version negotiation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RespProtocol {
    Resp2,
    Resp3,
}

impl Default for RespProtocol {
    fn default() -> Self {
        RespProtocol::Resp2
    }
}

/// Client connection state including protocol version
#[derive(Debug, Clone)]
pub struct ClientState {
    pub protocol: RespProtocol,
    pub client_name: Option<String>,
    pub client_id: u64,
    pub authenticated: bool,
}

impl Default for ClientState {
    fn default() -> Self {
        Self {
            protocol: RespProtocol::Resp2,
            client_name: None,
            client_id: 0,
            authenticated: false,
        }
    }
}

/// HELLO command response for RESP3 negotiation
pub fn hello_response(proto_version: u8, server_name: &str) -> Resp3Value {
    let mut map = BTreeMap::new();
    map.insert("server".to_string(), Resp3Value::bulk_string(server_name));
    map.insert("version".to_string(), Resp3Value::bulk_string("7.0.0"));
    map.insert("proto".to_string(), Resp3Value::Integer(proto_version as i64));
    map.insert("id".to_string(), Resp3Value::Integer(0));
    map.insert("mode".to_string(), Resp3Value::bulk_string("standalone"));
    map.insert("role".to_string(), Resp3Value::bulk_string("master"));
    map.insert("modules".to_string(), Resp3Value::Array(Some(vec![])));

    Resp3Value::Map(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== RESP3 Type Tests ====================

    #[test]
    fn test_serialize_null() {
        let value = Resp3Value::Null;
        assert_eq!(value.serialize(), b"_\r\n");
    }

    #[test]
    fn test_serialize_boolean_true() {
        let value = Resp3Value::Boolean(true);
        assert_eq!(value.serialize(), b"#t\r\n");
    }

    #[test]
    fn test_serialize_boolean_false() {
        let value = Resp3Value::Boolean(false);
        assert_eq!(value.serialize(), b"#f\r\n");
    }

    #[test]
    fn test_serialize_double() {
        let value = Resp3Value::Double(3.14);
        let serialized = value.serialize();
        assert!(serialized.starts_with(b","));
        assert!(serialized.ends_with(b"\r\n"));
    }

    #[test]
    fn test_serialize_double_infinity() {
        let value = Resp3Value::Double(f64::INFINITY);
        assert_eq!(value.serialize(), b",inf\r\n");

        let value = Resp3Value::Double(f64::NEG_INFINITY);
        assert_eq!(value.serialize(), b",-inf\r\n");
    }

    #[test]
    fn test_serialize_big_number() {
        let value = Resp3Value::BigNumber("12345678901234567890".to_string());
        assert_eq!(value.serialize(), b"(12345678901234567890\r\n");
    }

    #[test]
    fn test_serialize_verbatim_string() {
        let value = Resp3Value::VerbatimString {
            encoding: "txt".to_string(),
            data: b"Hello".to_vec(),
        };
        assert_eq!(value.serialize(), b"=9\r\ntxt:Hello\r\n");
    }

    #[test]
    fn test_serialize_map() {
        let mut map = BTreeMap::new();
        map.insert("key1".to_string(), Resp3Value::Integer(1));
        map.insert("key2".to_string(), Resp3Value::Integer(2));
        let value = Resp3Value::Map(map);
        let serialized = value.serialize();
        // Map has 2 entries
        assert!(serialized.starts_with(b"%2\r\n"));
    }

    #[test]
    fn test_serialize_set() {
        let mut set = BTreeSet::new();
        set.insert("a".to_string());
        set.insert("b".to_string());
        set.insert("c".to_string());
        let value = Resp3Value::Set(set);
        let serialized = value.serialize();
        assert!(serialized.starts_with(b"~3\r\n"));
    }

    #[test]
    fn test_serialize_push() {
        let value = Resp3Value::Push(vec![
            Resp3Value::bulk_string("pubsub"),
            Resp3Value::bulk_string("message"),
            Resp3Value::bulk_string("channel"),
        ]);
        let serialized = value.serialize();
        assert!(serialized.starts_with(b">3\r\n"));
    }

    #[test]
    fn test_serialize_blob_error() {
        let value = Resp3Value::BlobError(b"SYNTAX invalid".to_vec());
        assert_eq!(value.serialize(), b"!14\r\nSYNTAX invalid\r\n");
    }

    // ==================== Constructor Tests ====================

    #[test]
    fn test_constructors() {
        assert_eq!(Resp3Value::ok(), Resp3Value::SimpleString("OK".to_string()));
        assert_eq!(Resp3Value::null(), Resp3Value::Null);
        assert_eq!(Resp3Value::boolean(true), Resp3Value::Boolean(true));
        assert_eq!(Resp3Value::double(1.5), Resp3Value::Double(1.5));
        assert_eq!(Resp3Value::integer(42), Resp3Value::Integer(42));
    }

    // ==================== Conversion Tests ====================

    #[test]
    fn test_is_null() {
        assert!(Resp3Value::Null.is_null());
        assert!(Resp3Value::BulkString(None).is_null());
        assert!(Resp3Value::Array(None).is_null());
        assert!(!Resp3Value::Integer(0).is_null());
    }

    #[test]
    fn test_as_str() {
        assert_eq!(Resp3Value::SimpleString("hello".to_string()).as_str(), Some("hello"));
        assert_eq!(Resp3Value::bulk_string("world").as_str(), Some("world"));
        assert_eq!(Resp3Value::Integer(42).as_str(), None);
    }

    #[test]
    fn test_as_int() {
        assert_eq!(Resp3Value::Integer(42).as_int(), Some(42));
        assert_eq!(Resp3Value::bulk_string("123").as_int(), Some(123));
        assert_eq!(Resp3Value::bulk_string("abc").as_int(), None);
    }

    #[test]
    fn test_as_double() {
        assert_eq!(Resp3Value::Double(3.14).as_double(), Some(3.14));
        assert_eq!(Resp3Value::Integer(42).as_double(), Some(42.0));
        assert_eq!(Resp3Value::bulk_string("1.5").as_double(), Some(1.5));
    }

    #[test]
    fn test_as_bool() {
        assert_eq!(Resp3Value::Boolean(true).as_bool(), Some(true));
        assert_eq!(Resp3Value::Boolean(false).as_bool(), Some(false));
        assert_eq!(Resp3Value::Integer(1).as_bool(), Some(true));
        assert_eq!(Resp3Value::Integer(0).as_bool(), Some(false));
    }

    // ==================== Protocol Tests ====================

    #[test]
    fn test_hello_response() {
        let resp = hello_response(3, "test-cache");
        match resp {
            Resp3Value::Map(map) => {
                assert!(map.contains_key("server"));
                assert!(map.contains_key("version"));
                assert!(map.contains_key("proto"));
            }
            _ => panic!("Expected map response"),
        }
    }

    #[test]
    fn test_client_state_default() {
        let state = ClientState::default();
        assert_eq!(state.protocol, RespProtocol::Resp2);
        assert!(!state.authenticated);
    }
}
