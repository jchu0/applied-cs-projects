//! Variable-length integer encoding (varint)
//!
//! Uses the same encoding as Protocol Buffers:
//! - Each byte uses 7 bits for data and 1 bit (MSB) as continuation flag
//! - Unsigned integers are encoded directly
//! - Signed integers use ZigZag encoding to handle negative numbers efficiently

use crate::error::{Result, TsdbError};

/// Encode an unsigned 64-bit integer as a varint
pub fn encode_varint(mut value: u64) -> Vec<u8> {
    let mut result = Vec::with_capacity(10); // Max 10 bytes for u64

    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;

        if value != 0 {
            byte |= 0x80; // Set continuation bit
        }

        result.push(byte);

        if value == 0 {
            break;
        }
    }

    result
}

/// Encode a varint into a pre-allocated buffer, returning bytes written
pub fn encode_varint_to_buf(mut value: u64, buf: &mut [u8]) -> Result<usize> {
    let mut i = 0;

    loop {
        if i >= buf.len() {
            return Err(TsdbError::BufferOverflow {
                expected: i + 1,
                actual: buf.len(),
            });
        }

        let mut byte = (value & 0x7F) as u8;
        value >>= 7;

        if value != 0 {
            byte |= 0x80;
        }

        buf[i] = byte;
        i += 1;

        if value == 0 {
            break;
        }
    }

    Ok(i)
}

/// Decode a varint from a byte slice, returning (value, bytes_read)
pub fn decode_varint(data: &[u8]) -> Result<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift = 0;

    for (i, &byte) in data.iter().enumerate() {
        if shift >= 64 {
            return Err(TsdbError::Decompression("Varint overflow".into()));
        }

        result |= ((byte & 0x7F) as u64) << shift;
        shift += 7;

        if byte & 0x80 == 0 {
            return Ok((result, i + 1));
        }
    }

    Err(TsdbError::Decompression("Incomplete varint".into()))
}

/// Encode a signed 64-bit integer using ZigZag encoding
pub fn encode_signed_varint(value: i64) -> Vec<u8> {
    let zigzag = ((value << 1) ^ (value >> 63)) as u64;
    encode_varint(zigzag)
}

/// Encode a signed varint into a buffer
pub fn encode_signed_varint_to_buf(value: i64, buf: &mut [u8]) -> Result<usize> {
    let zigzag = ((value << 1) ^ (value >> 63)) as u64;
    encode_varint_to_buf(zigzag, buf)
}

/// Decode a signed varint (ZigZag encoded)
pub fn decode_signed_varint(data: &[u8]) -> Result<(i64, usize)> {
    let (zigzag, bytes_read) = decode_varint(data)?;
    let value = ((zigzag >> 1) as i64) ^ (-((zigzag & 1) as i64));
    Ok((value, bytes_read))
}

/// Calculate the encoded length of a varint without encoding it
pub fn varint_length(value: u64) -> usize {
    if value == 0 {
        return 1;
    }

    let bits = 64 - value.leading_zeros() as usize;
    (bits + 6) / 7
}

/// Calculate the encoded length of a signed varint
pub fn signed_varint_length(value: i64) -> usize {
    let zigzag = ((value << 1) ^ (value >> 63)) as u64;
    varint_length(zigzag)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_small() {
        for value in 0u64..128 {
            let encoded = encode_varint(value);
            assert_eq!(encoded.len(), 1);
            let (decoded, bytes_read) = decode_varint(&encoded).unwrap();
            assert_eq!(decoded, value);
            assert_eq!(bytes_read, 1);
        }
    }

    #[test]
    fn test_encode_decode_medium() {
        for value in [128u64, 255, 256, 16383, 16384] {
            let encoded = encode_varint(value);
            let (decoded, bytes_read) = decode_varint(&encoded).unwrap();
            assert_eq!(decoded, value);
            assert_eq!(bytes_read, encoded.len());
        }
    }

    #[test]
    fn test_encode_decode_large() {
        let values = [
            u64::MAX,
            u64::MAX / 2,
            1u64 << 63,
            (1u64 << 56) - 1,
            1_000_000_000_000u64,
        ];

        for value in values {
            let encoded = encode_varint(value);
            let (decoded, _) = decode_varint(&encoded).unwrap();
            assert_eq!(decoded, value);
        }
    }

    #[test]
    fn test_signed_varint() {
        let values = [0i64, 1, -1, 127, -128, 10000, -10000, i64::MAX, i64::MIN];

        for value in values {
            let encoded = encode_signed_varint(value);
            let (decoded, _) = decode_signed_varint(&encoded).unwrap();
            assert_eq!(decoded, value, "Failed for value {}", value);
        }
    }

    #[test]
    fn test_zigzag_encoding_efficiency() {
        // Small negative numbers should encode efficiently
        let encoded_minus_1 = encode_signed_varint(-1);
        let encoded_1 = encode_signed_varint(1);
        assert_eq!(encoded_minus_1.len(), encoded_1.len());
    }

    #[test]
    fn test_varint_length() {
        assert_eq!(varint_length(0), 1);
        assert_eq!(varint_length(127), 1);
        assert_eq!(varint_length(128), 2);
        assert_eq!(varint_length(16383), 2);
        assert_eq!(varint_length(16384), 3);
    }

    #[test]
    fn test_encode_to_buffer() {
        let mut buf = [0u8; 10];

        let bytes = encode_varint_to_buf(300, &mut buf).unwrap();
        assert_eq!(bytes, 2);

        let (decoded, _) = decode_varint(&buf[..bytes]).unwrap();
        assert_eq!(decoded, 300);
    }

    #[test]
    fn test_buffer_overflow() {
        let mut buf = [0u8; 1];
        let result = encode_varint_to_buf(300, &mut buf);
        assert!(matches!(result, Err(TsdbError::BufferOverflow { .. })));
    }

    #[test]
    fn test_incomplete_varint() {
        let data = [0x80, 0x80]; // Continuation bits set but no terminating byte
        let result = decode_varint(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_input() {
        let result = decode_varint(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_in_larger_buffer() {
        // Value followed by other data
        let mut data = encode_varint(12345);
        data.extend_from_slice(&[0xFF, 0xFF, 0xFF]);

        let (decoded, bytes_read) = decode_varint(&data).unwrap();
        assert_eq!(decoded, 12345);
        assert_eq!(bytes_read, 2); // 12345 takes 2 bytes
    }
}
