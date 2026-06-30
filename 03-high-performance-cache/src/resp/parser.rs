use bytes::BytesMut;
use thiserror::Error;

use super::RespValue;

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("Invalid RESP prefix: {0}")]
    InvalidPrefix(u8),
    #[error("Invalid integer: {0}")]
    InvalidInteger(String),
    #[error("Incomplete data")]
    Incomplete,
    #[error("Invalid UTF-8")]
    InvalidUtf8,
    #[error("Protocol error: {0}")]
    Protocol(String),
}

/// RESP protocol parser
pub struct RespParser {
    buffer: BytesMut,
}

impl RespParser {
    /// Create a new parser
    pub fn new() -> Self {
        Self {
            buffer: BytesMut::with_capacity(4096),
        }
    }

    /// Add data to the buffer
    pub fn feed(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    /// Get mutable reference to buffer for direct writing
    pub fn buffer_mut(&mut self) -> &mut BytesMut {
        &mut self.buffer
    }

    /// Try to parse a complete RESP value
    pub fn parse(&mut self) -> Result<Option<RespValue>, ParseError> {
        if self.buffer.is_empty() {
            return Ok(None);
        }

        let (value, consumed) = match self.parse_value(0)? {
            Some(result) => result,
            None => return Ok(None),
        };

        // Remove consumed bytes from buffer
        let _ = self.buffer.split_to(consumed);

        Ok(Some(value))
    }

    /// Parse a RESP value starting at the given offset
    /// Returns (value, bytes_consumed) or None if incomplete
    fn parse_value(&self, offset: usize) -> Result<Option<(RespValue, usize)>, ParseError> {
        if offset >= self.buffer.len() {
            return Ok(None);
        }

        let prefix = self.buffer[offset];
        match prefix {
            b'+' => self.parse_simple_string(offset),
            b'-' => self.parse_error(offset),
            b':' => self.parse_integer(offset),
            b'$' => self.parse_bulk_string(offset),
            b'*' => self.parse_array(offset),
            _ => Err(ParseError::InvalidPrefix(prefix)),
        }
    }

    /// Find CRLF starting from offset
    fn find_crlf(&self, offset: usize) -> Option<usize> {
        if offset >= self.buffer.len() {
            return None;
        }

        for i in offset..self.buffer.len().saturating_sub(1) {
            if self.buffer[i] == b'\r' && self.buffer[i + 1] == b'\n' {
                return Some(i);
            }
        }
        None
    }

    /// Parse a simple string (+OK\r\n)
    fn parse_simple_string(&self, offset: usize) -> Result<Option<(RespValue, usize)>, ParseError> {
        let crlf = match self.find_crlf(offset + 1) {
            Some(pos) => pos,
            None => return Ok(None),
        };

        let data = &self.buffer[offset + 1..crlf];
        let s = std::str::from_utf8(data)
            .map_err(|_| ParseError::InvalidUtf8)?
            .to_string();

        Ok(Some((RespValue::SimpleString(s), crlf + 2)))
    }

    /// Parse an error (-ERR message\r\n)
    fn parse_error(&self, offset: usize) -> Result<Option<(RespValue, usize)>, ParseError> {
        let crlf = match self.find_crlf(offset + 1) {
            Some(pos) => pos,
            None => return Ok(None),
        };

        let data = &self.buffer[offset + 1..crlf];
        let s = std::str::from_utf8(data)
            .map_err(|_| ParseError::InvalidUtf8)?
            .to_string();

        Ok(Some((RespValue::Error(s), crlf + 2)))
    }

    /// Parse an integer (:1000\r\n)
    fn parse_integer(&self, offset: usize) -> Result<Option<(RespValue, usize)>, ParseError> {
        let crlf = match self.find_crlf(offset + 1) {
            Some(pos) => pos,
            None => return Ok(None),
        };

        let data = &self.buffer[offset + 1..crlf];
        let s = std::str::from_utf8(data)
            .map_err(|_| ParseError::InvalidUtf8)?;

        let i: i64 = s.parse()
            .map_err(|_| ParseError::InvalidInteger(s.to_string()))?;

        Ok(Some((RespValue::Integer(i), crlf + 2)))
    }

    /// Parse a bulk string ($5\r\nhello\r\n)
    fn parse_bulk_string(&self, offset: usize) -> Result<Option<(RespValue, usize)>, ParseError> {
        let crlf = match self.find_crlf(offset + 1) {
            Some(pos) => pos,
            None => return Ok(None),
        };

        let len_str = std::str::from_utf8(&self.buffer[offset + 1..crlf])
            .map_err(|_| ParseError::InvalidUtf8)?;

        let len: i64 = len_str.parse()
            .map_err(|_| ParseError::InvalidInteger(len_str.to_string()))?;

        // Null bulk string
        if len == -1 {
            return Ok(Some((RespValue::BulkString(None), crlf + 2)));
        }

        if len < 0 {
            return Err(ParseError::Protocol("Invalid bulk string length".to_string()));
        }

        let data_start = crlf + 2;
        let data_end = data_start + len as usize;
        let total_end = data_end + 2; // Include trailing \r\n

        // Check if we have all the data
        if self.buffer.len() < total_end {
            return Ok(None);
        }

        let data = self.buffer[data_start..data_end].to_vec();
        Ok(Some((RespValue::BulkString(Some(data)), total_end)))
    }

    /// Parse an array (*2\r\n$3\r\nfoo\r\n$3\r\nbar\r\n)
    fn parse_array(&self, offset: usize) -> Result<Option<(RespValue, usize)>, ParseError> {
        let crlf = match self.find_crlf(offset + 1) {
            Some(pos) => pos,
            None => return Ok(None),
        };

        let len_str = std::str::from_utf8(&self.buffer[offset + 1..crlf])
            .map_err(|_| ParseError::InvalidUtf8)?;

        let len: i64 = len_str.parse()
            .map_err(|_| ParseError::InvalidInteger(len_str.to_string()))?;

        // Null array
        if len == -1 {
            return Ok(Some((RespValue::Array(None), crlf + 2)));
        }

        if len < 0 {
            return Err(ParseError::Protocol("Invalid array length".to_string()));
        }

        let mut items = Vec::with_capacity(len as usize);
        let mut current_offset = crlf + 2;

        for _ in 0..len {
            match self.parse_value(current_offset)? {
                Some((value, consumed)) => {
                    items.push(value);
                    current_offset = consumed;
                }
                None => return Ok(None),
            }
        }

        Ok(Some((RespValue::Array(Some(items)), current_offset)))
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get buffer length
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

impl Default for RespParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Simple String Tests ====================

    #[test]
    fn test_parse_simple_string() {
        let mut parser = RespParser::new();
        parser.feed(b"+OK\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(result, RespValue::SimpleString("OK".to_string()));
    }

    #[test]
    fn test_parse_simple_string_empty() {
        let mut parser = RespParser::new();
        parser.feed(b"+\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(result, RespValue::SimpleString("".to_string()));
    }

    #[test]
    fn test_parse_simple_string_with_spaces() {
        let mut parser = RespParser::new();
        parser.feed(b"+Hello World\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(result, RespValue::SimpleString("Hello World".to_string()));
    }

    #[test]
    fn test_parse_simple_string_incomplete() {
        let mut parser = RespParser::new();
        parser.feed(b"+OK");
        let result = parser.parse().unwrap();
        assert!(result.is_none());
    }

    // ==================== Error Tests ====================

    #[test]
    fn test_parse_error() {
        let mut parser = RespParser::new();
        parser.feed(b"-ERR unknown command\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(result, RespValue::Error("ERR unknown command".to_string()));
    }

    #[test]
    fn test_parse_error_empty() {
        let mut parser = RespParser::new();
        parser.feed(b"-\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(result, RespValue::Error("".to_string()));
    }

    #[test]
    fn test_parse_error_wrongtype() {
        let mut parser = RespParser::new();
        parser.feed(b"-WRONGTYPE Operation against a key holding the wrong kind of value\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(
            result,
            RespValue::Error("WRONGTYPE Operation against a key holding the wrong kind of value".to_string())
        );
    }

    // ==================== Integer Tests ====================

    #[test]
    fn test_parse_integer() {
        let mut parser = RespParser::new();
        parser.feed(b":1000\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(result, RespValue::Integer(1000));
    }

    #[test]
    fn test_parse_integer_zero() {
        let mut parser = RespParser::new();
        parser.feed(b":0\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(result, RespValue::Integer(0));
    }

    #[test]
    fn test_parse_integer_negative() {
        let mut parser = RespParser::new();
        parser.feed(b":-123\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(result, RespValue::Integer(-123));
    }

    #[test]
    fn test_parse_integer_large() {
        let mut parser = RespParser::new();
        parser.feed(b":9223372036854775807\r\n"); // i64::MAX
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(result, RespValue::Integer(i64::MAX));
    }

    #[test]
    fn test_parse_integer_negative_large() {
        let mut parser = RespParser::new();
        parser.feed(b":-9223372036854775808\r\n"); // i64::MIN
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(result, RespValue::Integer(i64::MIN));
    }

    #[test]
    fn test_parse_integer_invalid() {
        let mut parser = RespParser::new();
        parser.feed(b":abc\r\n");
        let result = parser.parse();
        assert!(result.is_err());
    }

    // ==================== Bulk String Tests ====================

    #[test]
    fn test_parse_bulk_string() {
        let mut parser = RespParser::new();
        parser.feed(b"$5\r\nhello\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(result, RespValue::BulkString(Some(b"hello".to_vec())));
    }

    #[test]
    fn test_parse_null_bulk_string() {
        let mut parser = RespParser::new();
        parser.feed(b"$-1\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(result, RespValue::BulkString(None));
    }

    #[test]
    fn test_parse_bulk_string_empty() {
        let mut parser = RespParser::new();
        parser.feed(b"$0\r\n\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(result, RespValue::BulkString(Some(Vec::new())));
    }

    #[test]
    fn test_parse_bulk_string_with_binary() {
        let mut parser = RespParser::new();
        parser.feed(b"$4\r\n\x00\x01\x02\x03\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(result, RespValue::BulkString(Some(vec![0, 1, 2, 3])));
    }

    #[test]
    fn test_parse_bulk_string_with_crlf() {
        let mut parser = RespParser::new();
        parser.feed(b"$7\r\nhello\r\n\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(result, RespValue::BulkString(Some(b"hello\r\n".to_vec())));
    }

    #[test]
    fn test_parse_bulk_string_long() {
        let mut parser = RespParser::new();
        let long_string = "x".repeat(10000);
        let input = format!("${}\r\n{}\r\n", long_string.len(), long_string);
        parser.feed(input.as_bytes());
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(result, RespValue::BulkString(Some(long_string.into_bytes())));
    }

    #[test]
    fn test_parse_bulk_string_incomplete_length() {
        let mut parser = RespParser::new();
        parser.feed(b"$5\r\nhel");
        let result = parser.parse().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_bulk_string_incomplete_crlf() {
        let mut parser = RespParser::new();
        parser.feed(b"$5\r\nhello");
        let result = parser.parse().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_bulk_string_invalid_length() {
        let mut parser = RespParser::new();
        parser.feed(b"$-2\r\n");
        let result = parser.parse();
        assert!(result.is_err());
    }

    // ==================== Array Tests ====================

    #[test]
    fn test_parse_array() {
        let mut parser = RespParser::new();
        parser.feed(b"*2\r\n$3\r\nfoo\r\n$3\r\nbar\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(
            result,
            RespValue::Array(Some(vec![
                RespValue::BulkString(Some(b"foo".to_vec())),
                RespValue::BulkString(Some(b"bar".to_vec())),
            ]))
        );
    }

    #[test]
    fn test_parse_null_array() {
        let mut parser = RespParser::new();
        parser.feed(b"*-1\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(result, RespValue::Array(None));
    }

    #[test]
    fn test_parse_empty_array() {
        let mut parser = RespParser::new();
        parser.feed(b"*0\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(result, RespValue::Array(Some(Vec::new())));
    }

    #[test]
    fn test_parse_array_of_integers() {
        let mut parser = RespParser::new();
        parser.feed(b"*3\r\n:1\r\n:2\r\n:3\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(
            result,
            RespValue::Array(Some(vec![
                RespValue::Integer(1),
                RespValue::Integer(2),
                RespValue::Integer(3),
            ]))
        );
    }

    #[test]
    fn test_parse_nested_array() {
        let mut parser = RespParser::new();
        parser.feed(b"*2\r\n*2\r\n$3\r\nfoo\r\n$3\r\nbar\r\n*1\r\n:42\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(
            result,
            RespValue::Array(Some(vec![
                RespValue::Array(Some(vec![
                    RespValue::BulkString(Some(b"foo".to_vec())),
                    RespValue::BulkString(Some(b"bar".to_vec())),
                ])),
                RespValue::Array(Some(vec![RespValue::Integer(42)])),
            ]))
        );
    }

    #[test]
    fn test_parse_mixed_array() {
        let mut parser = RespParser::new();
        parser.feed(b"*4\r\n+OK\r\n-ERR\r\n:100\r\n$4\r\ntest\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(
            result,
            RespValue::Array(Some(vec![
                RespValue::SimpleString("OK".to_string()),
                RespValue::Error("ERR".to_string()),
                RespValue::Integer(100),
                RespValue::BulkString(Some(b"test".to_vec())),
            ]))
        );
    }

    #[test]
    fn test_parse_array_with_null_bulk_string() {
        let mut parser = RespParser::new();
        parser.feed(b"*3\r\n$3\r\nfoo\r\n$-1\r\n$3\r\nbar\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(
            result,
            RespValue::Array(Some(vec![
                RespValue::BulkString(Some(b"foo".to_vec())),
                RespValue::BulkString(None),
                RespValue::BulkString(Some(b"bar".to_vec())),
            ]))
        );
    }

    #[test]
    fn test_parse_array_incomplete() {
        let mut parser = RespParser::new();
        parser.feed(b"*2\r\n$3\r\nfoo\r\n");
        let result = parser.parse().unwrap();
        assert!(result.is_none());
    }

    // ==================== Multiple Messages Tests ====================

    #[test]
    fn test_parse_incomplete() {
        let mut parser = RespParser::new();
        parser.feed(b"$5\r\nhel");
        let result = parser.parse().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_multiple() {
        let mut parser = RespParser::new();
        parser.feed(b"+OK\r\n:100\r\n");

        let result1 = parser.parse().unwrap().unwrap();
        assert_eq!(result1, RespValue::SimpleString("OK".to_string()));

        let result2 = parser.parse().unwrap().unwrap();
        assert_eq!(result2, RespValue::Integer(100));
    }

    #[test]
    fn test_parse_multiple_arrays() {
        let mut parser = RespParser::new();
        parser.feed(b"*1\r\n$4\r\nPING\r\n*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$5\r\nvalue\r\n");

        let result1 = parser.parse().unwrap().unwrap();
        assert_eq!(
            result1,
            RespValue::Array(Some(vec![RespValue::BulkString(Some(b"PING".to_vec()))]))
        );

        let result2 = parser.parse().unwrap().unwrap();
        assert_eq!(
            result2,
            RespValue::Array(Some(vec![
                RespValue::BulkString(Some(b"SET".to_vec())),
                RespValue::BulkString(Some(b"key".to_vec())),
                RespValue::BulkString(Some(b"value".to_vec())),
            ]))
        );
    }

    #[test]
    fn test_parse_incremental_feed() {
        let mut parser = RespParser::new();

        // Feed data incrementally
        parser.feed(b"*2\r\n");
        assert!(parser.parse().unwrap().is_none());

        parser.feed(b"$3\r\n");
        assert!(parser.parse().unwrap().is_none());

        parser.feed(b"foo\r\n");
        assert!(parser.parse().unwrap().is_none());

        parser.feed(b"$3\r\nbar\r\n");
        let result = parser.parse().unwrap().unwrap();

        assert_eq!(
            result,
            RespValue::Array(Some(vec![
                RespValue::BulkString(Some(b"foo".to_vec())),
                RespValue::BulkString(Some(b"bar".to_vec())),
            ]))
        );
    }

    // ==================== Invalid Prefix Tests ====================

    #[test]
    fn test_parse_invalid_prefix() {
        let mut parser = RespParser::new();
        parser.feed(b"?invalid\r\n");
        let result = parser.parse();
        assert!(result.is_err());
    }

    // ==================== Buffer Utility Tests ====================

    #[test]
    fn test_parser_clear() {
        let mut parser = RespParser::new();
        parser.feed(b"+OK\r\n:100\r\n");
        parser.clear();
        assert!(parser.is_empty());
        assert_eq!(parser.len(), 0);
    }

    #[test]
    fn test_parser_len() {
        let mut parser = RespParser::new();
        parser.feed(b"+OK\r\n");
        assert_eq!(parser.len(), 5);
        parser.parse().unwrap();
        assert_eq!(parser.len(), 0);
    }

    #[test]
    fn test_parser_default() {
        let parser: RespParser = Default::default();
        assert!(parser.is_empty());
    }

    // ==================== Redis Command Tests ====================

    #[test]
    fn test_parse_redis_get_command() {
        let mut parser = RespParser::new();
        parser.feed(b"*2\r\n$3\r\nGET\r\n$3\r\nkey\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(
            result,
            RespValue::Array(Some(vec![
                RespValue::BulkString(Some(b"GET".to_vec())),
                RespValue::BulkString(Some(b"key".to_vec())),
            ]))
        );
    }

    #[test]
    fn test_parse_redis_set_command_with_options() {
        let mut parser = RespParser::new();
        parser.feed(b"*5\r\n$3\r\nSET\r\n$3\r\nkey\r\n$5\r\nvalue\r\n$2\r\nEX\r\n$2\r\n60\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(
            result,
            RespValue::Array(Some(vec![
                RespValue::BulkString(Some(b"SET".to_vec())),
                RespValue::BulkString(Some(b"key".to_vec())),
                RespValue::BulkString(Some(b"value".to_vec())),
                RespValue::BulkString(Some(b"EX".to_vec())),
                RespValue::BulkString(Some(b"60".to_vec())),
            ]))
        );
    }

    #[test]
    fn test_parse_redis_mset_command() {
        let mut parser = RespParser::new();
        parser.feed(b"*5\r\n$4\r\nMSET\r\n$2\r\nk1\r\n$2\r\nv1\r\n$2\r\nk2\r\n$2\r\nv2\r\n");
        let result = parser.parse().unwrap().unwrap();
        assert_eq!(
            result,
            RespValue::Array(Some(vec![
                RespValue::BulkString(Some(b"MSET".to_vec())),
                RespValue::BulkString(Some(b"k1".to_vec())),
                RespValue::BulkString(Some(b"v1".to_vec())),
                RespValue::BulkString(Some(b"k2".to_vec())),
                RespValue::BulkString(Some(b"v2".to_vec())),
            ]))
        );
    }
}
