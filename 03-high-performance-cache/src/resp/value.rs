use bytes::{BufMut, BytesMut};
use std::fmt;

/// RESP (Redis Serialization Protocol) value types
#[derive(Debug, Clone, PartialEq)]
pub enum RespValue {
    /// Simple strings: +OK\r\n
    SimpleString(String),
    /// Errors: -ERR message\r\n
    Error(String),
    /// Integers: :1000\r\n
    Integer(i64),
    /// Bulk strings: $5\r\nhello\r\n or $-1\r\n for null
    BulkString(Option<Vec<u8>>),
    /// Arrays: *2\r\n$3\r\nfoo\r\n$3\r\nbar\r\n
    Array(Option<Vec<RespValue>>),
}

impl RespValue {
    /// Serialize the value to RESP protocol bytes
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = BytesMut::with_capacity(64);
        self.serialize_into(&mut buf);
        buf.to_vec()
    }

    /// Serialize into a buffer
    pub fn serialize_into(&self, buf: &mut BytesMut) {
        match self {
            RespValue::SimpleString(s) => {
                buf.put_u8(b'+');
                buf.put_slice(s.as_bytes());
                buf.put_slice(b"\r\n");
            }
            RespValue::Error(s) => {
                buf.put_u8(b'-');
                buf.put_slice(s.as_bytes());
                buf.put_slice(b"\r\n");
            }
            RespValue::Integer(i) => {
                buf.put_u8(b':');
                buf.put_slice(i.to_string().as_bytes());
                buf.put_slice(b"\r\n");
            }
            RespValue::BulkString(None) => {
                buf.put_slice(b"$-1\r\n");
            }
            RespValue::BulkString(Some(data)) => {
                buf.put_u8(b'$');
                buf.put_slice(data.len().to_string().as_bytes());
                buf.put_slice(b"\r\n");
                buf.put_slice(data);
                buf.put_slice(b"\r\n");
            }
            RespValue::Array(None) => {
                buf.put_slice(b"*-1\r\n");
            }
            RespValue::Array(Some(arr)) => {
                buf.put_u8(b'*');
                buf.put_slice(arr.len().to_string().as_bytes());
                buf.put_slice(b"\r\n");
                for item in arr {
                    item.serialize_into(buf);
                }
            }
        }
    }

    /// Create a simple OK response
    pub fn ok() -> Self {
        RespValue::SimpleString("OK".to_string())
    }

    /// Create a null bulk string
    pub fn null() -> Self {
        RespValue::BulkString(None)
    }

    /// Create a null array
    pub fn null_array() -> Self {
        RespValue::Array(None)
    }

    /// Create an error response
    pub fn error(msg: impl Into<String>) -> Self {
        RespValue::Error(msg.into())
    }

    /// Create a bulk string from bytes
    pub fn bulk(data: impl Into<Vec<u8>>) -> Self {
        RespValue::BulkString(Some(data.into()))
    }

    /// Create a bulk string from a string
    pub fn bulk_string(s: impl Into<String>) -> Self {
        RespValue::BulkString(Some(s.into().into_bytes()))
    }

    /// Create an integer response
    pub fn integer(i: i64) -> Self {
        RespValue::Integer(i)
    }

    /// Create an array from values
    pub fn array(items: Vec<RespValue>) -> Self {
        RespValue::Array(Some(items))
    }

    /// Try to convert to string
    pub fn as_str(&self) -> Option<&str> {
        match self {
            RespValue::SimpleString(s) => Some(s),
            RespValue::BulkString(Some(data)) => std::str::from_utf8(data).ok(),
            _ => None,
        }
    }

    /// Try to convert to bytes
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            RespValue::SimpleString(s) => Some(s.as_bytes()),
            RespValue::BulkString(Some(data)) => Some(data),
            _ => None,
        }
    }

    /// Try to convert to integer
    pub fn as_int(&self) -> Option<i64> {
        match self {
            RespValue::Integer(i) => Some(*i),
            RespValue::BulkString(Some(data)) => {
                std::str::from_utf8(data).ok()?.parse().ok()
            }
            _ => None,
        }
    }

    /// Check if this is a null value
    pub fn is_null(&self) -> bool {
        matches!(
            self,
            RespValue::BulkString(None) | RespValue::Array(None)
        )
    }

    /// Get array items if this is an array
    pub fn into_array(self) -> Option<Vec<RespValue>> {
        match self {
            RespValue::Array(arr) => arr,
            _ => None,
        }
    }
}

impl fmt::Display for RespValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RespValue::SimpleString(s) => write!(f, "{}", s),
            RespValue::Error(s) => write!(f, "(error) {}", s),
            RespValue::Integer(i) => write!(f, "(integer) {}", i),
            RespValue::BulkString(None) => write!(f, "(nil)"),
            RespValue::BulkString(Some(data)) => {
                if let Ok(s) = std::str::from_utf8(data) {
                    write!(f, "\"{}\"", s)
                } else {
                    write!(f, "{:?}", data)
                }
            }
            RespValue::Array(None) => write!(f, "(nil)"),
            RespValue::Array(Some(arr)) => {
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
        }
    }
}

impl From<String> for RespValue {
    fn from(s: String) -> Self {
        RespValue::BulkString(Some(s.into_bytes()))
    }
}

impl From<&str> for RespValue {
    fn from(s: &str) -> Self {
        RespValue::BulkString(Some(s.as_bytes().to_vec()))
    }
}

impl From<i64> for RespValue {
    fn from(i: i64) -> Self {
        RespValue::Integer(i)
    }
}

impl From<Vec<u8>> for RespValue {
    fn from(data: Vec<u8>) -> Self {
        RespValue::BulkString(Some(data))
    }
}

impl From<Vec<RespValue>> for RespValue {
    fn from(arr: Vec<RespValue>) -> Self {
        RespValue::Array(Some(arr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Serialization Tests ====================

    #[test]
    fn test_serialize_simple_string() {
        let value = RespValue::SimpleString("OK".to_string());
        assert_eq!(value.serialize(), b"+OK\r\n");
    }

    #[test]
    fn test_serialize_simple_string_empty() {
        let value = RespValue::SimpleString("".to_string());
        assert_eq!(value.serialize(), b"+\r\n");
    }

    #[test]
    fn test_serialize_error() {
        let value = RespValue::Error("ERR unknown command".to_string());
        assert_eq!(value.serialize(), b"-ERR unknown command\r\n");
    }

    #[test]
    fn test_serialize_integer() {
        let value = RespValue::Integer(1000);
        assert_eq!(value.serialize(), b":1000\r\n");
    }

    #[test]
    fn test_serialize_integer_zero() {
        let value = RespValue::Integer(0);
        assert_eq!(value.serialize(), b":0\r\n");
    }

    #[test]
    fn test_serialize_integer_negative() {
        let value = RespValue::Integer(-123);
        assert_eq!(value.serialize(), b":-123\r\n");
    }

    #[test]
    fn test_serialize_bulk_string() {
        let value = RespValue::BulkString(Some(b"hello".to_vec()));
        assert_eq!(value.serialize(), b"$5\r\nhello\r\n");
    }

    #[test]
    fn test_serialize_bulk_string_empty() {
        let value = RespValue::BulkString(Some(Vec::new()));
        assert_eq!(value.serialize(), b"$0\r\n\r\n");
    }

    #[test]
    fn test_serialize_null_bulk_string() {
        let value = RespValue::BulkString(None);
        assert_eq!(value.serialize(), b"$-1\r\n");
    }

    #[test]
    fn test_serialize_array() {
        let value = RespValue::Array(Some(vec![
            RespValue::BulkString(Some(b"foo".to_vec())),
            RespValue::BulkString(Some(b"bar".to_vec())),
        ]));
        assert_eq!(value.serialize(), b"*2\r\n$3\r\nfoo\r\n$3\r\nbar\r\n");
    }

    #[test]
    fn test_serialize_empty_array() {
        let value = RespValue::Array(Some(Vec::new()));
        assert_eq!(value.serialize(), b"*0\r\n");
    }

    #[test]
    fn test_serialize_null_array() {
        let value = RespValue::Array(None);
        assert_eq!(value.serialize(), b"*-1\r\n");
    }

    #[test]
    fn test_serialize_nested_array() {
        let value = RespValue::Array(Some(vec![
            RespValue::Array(Some(vec![
                RespValue::Integer(1),
                RespValue::Integer(2),
            ])),
            RespValue::Array(Some(vec![
                RespValue::Integer(3),
                RespValue::Integer(4),
            ])),
        ]));
        assert_eq!(value.serialize(), b"*2\r\n*2\r\n:1\r\n:2\r\n*2\r\n:3\r\n:4\r\n");
    }

    #[test]
    fn test_serialize_mixed_array() {
        let value = RespValue::Array(Some(vec![
            RespValue::SimpleString("OK".to_string()),
            RespValue::Error("ERR".to_string()),
            RespValue::Integer(42),
            RespValue::BulkString(Some(b"test".to_vec())),
            RespValue::BulkString(None),
        ]));
        assert_eq!(value.serialize(), b"*5\r\n+OK\r\n-ERR\r\n:42\r\n$4\r\ntest\r\n$-1\r\n");
    }

    // ==================== Constructor Tests ====================

    #[test]
    fn test_ok() {
        let value = RespValue::ok();
        assert_eq!(value, RespValue::SimpleString("OK".to_string()));
    }

    #[test]
    fn test_null() {
        let value = RespValue::null();
        assert_eq!(value, RespValue::BulkString(None));
    }

    #[test]
    fn test_null_array() {
        let value = RespValue::null_array();
        assert_eq!(value, RespValue::Array(None));
    }

    #[test]
    fn test_error() {
        let value = RespValue::error("test error");
        assert_eq!(value, RespValue::Error("test error".to_string()));
    }

    #[test]
    fn test_bulk() {
        let value = RespValue::bulk(b"hello".to_vec());
        assert_eq!(value, RespValue::BulkString(Some(b"hello".to_vec())));
    }

    #[test]
    fn test_bulk_string() {
        let value = RespValue::bulk_string("hello");
        assert_eq!(value, RespValue::BulkString(Some(b"hello".to_vec())));
    }

    #[test]
    fn test_integer() {
        let value = RespValue::integer(42);
        assert_eq!(value, RespValue::Integer(42));
    }

    #[test]
    fn test_array() {
        let value = RespValue::array(vec![RespValue::Integer(1), RespValue::Integer(2)]);
        assert_eq!(
            value,
            RespValue::Array(Some(vec![RespValue::Integer(1), RespValue::Integer(2)]))
        );
    }

    // ==================== Conversion Tests ====================

    #[test]
    fn test_as_str_simple_string() {
        let value = RespValue::SimpleString("hello".to_string());
        assert_eq!(value.as_str(), Some("hello"));
    }

    #[test]
    fn test_as_str_bulk_string() {
        let value = RespValue::BulkString(Some(b"hello".to_vec()));
        assert_eq!(value.as_str(), Some("hello"));
    }

    #[test]
    fn test_as_str_null_bulk_string() {
        let value = RespValue::BulkString(None);
        assert_eq!(value.as_str(), None);
    }

    #[test]
    fn test_as_str_integer() {
        let value = RespValue::Integer(42);
        assert_eq!(value.as_str(), None);
    }

    #[test]
    fn test_as_bytes_simple_string() {
        let value = RespValue::SimpleString("hello".to_string());
        assert_eq!(value.as_bytes(), Some(b"hello".as_slice()));
    }

    #[test]
    fn test_as_bytes_bulk_string() {
        let value = RespValue::BulkString(Some(b"hello".to_vec()));
        assert_eq!(value.as_bytes(), Some(b"hello".as_slice()));
    }

    #[test]
    fn test_as_bytes_null_bulk_string() {
        let value = RespValue::BulkString(None);
        assert_eq!(value.as_bytes(), None);
    }

    #[test]
    fn test_as_int_integer() {
        let value = RespValue::Integer(42);
        assert_eq!(value.as_int(), Some(42));
    }

    #[test]
    fn test_as_int_bulk_string_numeric() {
        let value = RespValue::BulkString(Some(b"123".to_vec()));
        assert_eq!(value.as_int(), Some(123));
    }

    #[test]
    fn test_as_int_bulk_string_non_numeric() {
        let value = RespValue::BulkString(Some(b"abc".to_vec()));
        assert_eq!(value.as_int(), None);
    }

    #[test]
    fn test_as_int_null_bulk_string() {
        let value = RespValue::BulkString(None);
        assert_eq!(value.as_int(), None);
    }

    #[test]
    fn test_is_null_bulk_string() {
        assert!(RespValue::BulkString(None).is_null());
        assert!(!RespValue::BulkString(Some(vec![])).is_null());
    }

    #[test]
    fn test_is_null_array() {
        assert!(RespValue::Array(None).is_null());
        assert!(!RespValue::Array(Some(vec![])).is_null());
    }

    #[test]
    fn test_into_array() {
        let value = RespValue::Array(Some(vec![RespValue::Integer(1)]));
        let arr = value.into_array();
        assert_eq!(arr, Some(vec![RespValue::Integer(1)]));
    }

    #[test]
    fn test_into_array_null() {
        let value = RespValue::Array(None);
        let arr = value.into_array();
        assert_eq!(arr, None);
    }

    #[test]
    fn test_into_array_non_array() {
        let value = RespValue::Integer(42);
        let arr = value.into_array();
        assert_eq!(arr, None);
    }

    // ==================== From Trait Tests ====================

    #[test]
    fn test_from_string() {
        let value: RespValue = String::from("hello").into();
        assert_eq!(value, RespValue::BulkString(Some(b"hello".to_vec())));
    }

    #[test]
    fn test_from_str() {
        let value: RespValue = "hello".into();
        assert_eq!(value, RespValue::BulkString(Some(b"hello".to_vec())));
    }

    #[test]
    fn test_from_i64() {
        let value: RespValue = 42i64.into();
        assert_eq!(value, RespValue::Integer(42));
    }

    #[test]
    fn test_from_vec_u8() {
        let value: RespValue = vec![1, 2, 3].into();
        assert_eq!(value, RespValue::BulkString(Some(vec![1, 2, 3])));
    }

    #[test]
    fn test_from_vec_resp_value() {
        let value: RespValue = vec![RespValue::Integer(1), RespValue::Integer(2)].into();
        assert_eq!(
            value,
            RespValue::Array(Some(vec![RespValue::Integer(1), RespValue::Integer(2)]))
        );
    }

    // ==================== Display Tests ====================

    #[test]
    fn test_display_simple_string() {
        let value = RespValue::SimpleString("OK".to_string());
        assert_eq!(format!("{}", value), "OK");
    }

    #[test]
    fn test_display_error() {
        let value = RespValue::Error("ERR test".to_string());
        assert_eq!(format!("{}", value), "(error) ERR test");
    }

    #[test]
    fn test_display_integer() {
        let value = RespValue::Integer(42);
        assert_eq!(format!("{}", value), "(integer) 42");
    }

    #[test]
    fn test_display_bulk_string() {
        let value = RespValue::BulkString(Some(b"hello".to_vec()));
        assert_eq!(format!("{}", value), "\"hello\"");
    }

    #[test]
    fn test_display_null_bulk_string() {
        let value = RespValue::BulkString(None);
        assert_eq!(format!("{}", value), "(nil)");
    }

    #[test]
    fn test_display_empty_array() {
        let value = RespValue::Array(Some(Vec::new()));
        assert_eq!(format!("{}", value), "(empty array)");
    }

    #[test]
    fn test_display_null_array() {
        let value = RespValue::Array(None);
        assert_eq!(format!("{}", value), "(nil)");
    }
}
