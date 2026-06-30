//! Run-Length Encoding (RLE) for repeated values
//!
//! RLE is effective when consecutive values are identical, which is common in:
//! - Status/state metrics (e.g., health = 1, 1, 1, 1, ...)
//! - Boolean flags
//! - Discrete values with low cardinality

use crate::error::{Result, TsdbError};
use super::varint::{encode_varint, decode_varint};

/// Run-length encoder for u64 values
#[derive(Debug, Default)]
pub struct RleEncoder {
    /// Current run value
    current_value: Option<u64>,
    /// Current run length
    run_length: u64,
    /// Output buffer
    buffer: Vec<u8>,
}

impl RleEncoder {
    /// Create a new RLE encoder
    pub fn new() -> Self {
        Self::default()
    }

    /// Encode a single value
    pub fn encode(&mut self, value: u64) {
        match self.current_value {
            None => {
                self.current_value = Some(value);
                self.run_length = 1;
            }
            Some(v) if v == value => {
                self.run_length += 1;
            }
            Some(_) => {
                self.flush_run();
                self.current_value = Some(value);
                self.run_length = 1;
            }
        }
    }

    /// Encode multiple values
    pub fn encode_all(&mut self, values: &[u64]) {
        for &value in values {
            self.encode(value);
        }
    }

    /// Flush the current run to the buffer
    fn flush_run(&mut self) {
        if let Some(value) = self.current_value {
            self.buffer.extend_from_slice(&encode_varint(value));
            self.buffer.extend_from_slice(&encode_varint(self.run_length));
        }
    }

    /// Get the encoded data
    pub fn finish(mut self) -> Vec<u8> {
        self.flush_run();
        self.buffer
    }

    /// Get current buffer length
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if no values encoded
    pub fn is_empty(&self) -> bool {
        self.current_value.is_none() && self.buffer.is_empty()
    }

    /// Reset the encoder
    pub fn reset(&mut self) {
        self.current_value = None;
        self.run_length = 0;
        self.buffer.clear();
    }
}

/// Run-length decoder
#[derive(Debug, Default)]
pub struct RleDecoder {
    /// Current value being repeated
    current_value: u64,
    /// Remaining repeats
    remaining: u64,
}

impl RleDecoder {
    /// Create a new decoder
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode all values from the buffer
    pub fn decode_all(&mut self, data: &[u8]) -> Result<Vec<u64>> {
        let mut result = Vec::new();
        let mut offset = 0;

        while offset < data.len() {
            let (value, bytes_read) = decode_varint(&data[offset..])?;
            offset += bytes_read;

            if offset >= data.len() {
                return Err(TsdbError::Decompression("Incomplete RLE data".into()));
            }

            let (count, bytes_read) = decode_varint(&data[offset..])?;
            offset += bytes_read;

            for _ in 0..count {
                result.push(value);
            }
        }

        Ok(result)
    }

    /// Decode values one at a time from a stateful iterator
    pub fn decode_next(&mut self, data: &[u8], offset: &mut usize) -> Result<Option<u64>> {
        if self.remaining > 0 {
            self.remaining -= 1;
            return Ok(Some(self.current_value));
        }

        if *offset >= data.len() {
            return Ok(None);
        }

        let (value, bytes_read) = decode_varint(&data[*offset..])?;
        *offset += bytes_read;

        if *offset >= data.len() {
            return Err(TsdbError::Decompression("Incomplete RLE data".into()));
        }

        let (count, bytes_read) = decode_varint(&data[*offset..])?;
        *offset += bytes_read;

        self.current_value = value;
        self.remaining = count - 1;

        Ok(Some(value))
    }

    /// Reset the decoder
    pub fn reset(&mut self) {
        self.current_value = 0;
        self.remaining = 0;
    }
}

/// RLE encoder for floating-point values (uses bit representation)
#[derive(Debug, Default)]
pub struct RleF64Encoder {
    inner: RleEncoder,
}

impl RleF64Encoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn encode(&mut self, value: f64) {
        self.inner.encode(value.to_bits());
    }

    pub fn encode_all(&mut self, values: &[f64]) {
        for &value in values {
            self.encode(value);
        }
    }

    pub fn finish(self) -> Vec<u8> {
        self.inner.finish()
    }
}

/// RLE decoder for floating-point values
#[derive(Debug, Default)]
pub struct RleF64Decoder {
    inner: RleDecoder,
}

impl RleF64Decoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn decode_all(&mut self, data: &[u8]) -> Result<Vec<f64>> {
        let values = self.inner.decode_all(data)?;
        Ok(values.into_iter().map(f64::from_bits).collect())
    }
}

/// Compress u64 values with RLE
pub fn rle_compress(values: &[u64]) -> Vec<u8> {
    let mut encoder = RleEncoder::new();
    encoder.encode_all(values);
    encoder.finish()
}

/// Decompress RLE-encoded u64 values
pub fn rle_decompress(data: &[u8]) -> Result<Vec<u64>> {
    let mut decoder = RleDecoder::new();
    decoder.decode_all(data)
}

/// Compress f64 values with RLE
pub fn rle_compress_f64(values: &[f64]) -> Vec<u8> {
    let mut encoder = RleF64Encoder::new();
    encoder.encode_all(values);
    encoder.finish()
}

/// Decompress RLE-encoded f64 values
pub fn rle_decompress_f64(data: &[u8]) -> Result<Vec<f64>> {
    let mut decoder = RleF64Decoder::new();
    decoder.decode_all(data)
}

/// Calculate if RLE would be beneficial
pub fn should_use_rle(values: &[u64]) -> bool {
    if values.len() < 4 {
        return false;
    }

    // Count runs
    let mut runs = 1;
    for i in 1..values.len() {
        if values[i] != values[i - 1] {
            runs += 1;
        }
    }

    // RLE is beneficial if average run length > 4
    values.len() / runs > 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rle_repeated_values() {
        let values = vec![5u64; 1000];

        let compressed = rle_compress(&values);
        let decompressed = rle_decompress(&compressed).unwrap();

        assert_eq!(values, decompressed);
        // Should be very small: just value (1 byte) + count (2 bytes)
        assert!(compressed.len() < 10);
    }

    #[test]
    fn test_rle_alternating() {
        let values: Vec<u64> = (0..100).map(|i| i % 2).collect();

        let compressed = rle_compress(&values);
        let decompressed = rle_decompress(&compressed).unwrap();

        assert_eq!(values, decompressed);
    }

    #[test]
    fn test_rle_runs() {
        let mut values = Vec::new();
        for i in 0..10 {
            for _ in 0..(i + 1) * 10 {
                values.push(i as u64);
            }
        }

        let compressed = rle_compress(&values);
        let decompressed = rle_decompress(&compressed).unwrap();

        assert_eq!(values, decompressed);
    }

    #[test]
    fn test_rle_single_value() {
        let values = vec![42u64];

        let compressed = rle_compress(&values);
        let decompressed = rle_decompress(&compressed).unwrap();

        assert_eq!(values, decompressed);
    }

    #[test]
    fn test_rle_empty() {
        let values: Vec<u64> = vec![];

        let compressed = rle_compress(&values);
        let decompressed = rle_decompress(&compressed).unwrap();

        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_rle_f64() {
        let values = vec![1.5f64; 100];

        let compressed = rle_compress_f64(&values);
        let decompressed = rle_decompress_f64(&compressed).unwrap();

        assert_eq!(values, decompressed);
    }

    #[test]
    fn test_rle_mixed_f64() {
        let values: Vec<f64> = vec![
            1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0,
        ];

        let compressed = rle_compress_f64(&values);
        let decompressed = rle_decompress_f64(&compressed).unwrap();

        assert_eq!(values, decompressed);
    }

    #[test]
    fn test_should_use_rle() {
        // Long runs - should use RLE
        let long_runs = vec![1u64; 100];
        assert!(should_use_rle(&long_runs));

        // Short runs - shouldn't use RLE
        let short_runs: Vec<u64> = (0..100).collect();
        assert!(!should_use_rle(&short_runs));

        // Too short
        let short = vec![1u64, 1, 1];
        assert!(!should_use_rle(&short));
    }

    #[test]
    fn test_rle_compression_ratio() {
        // Highly repetitive data
        let values = vec![42u64; 10000];
        let original_size = values.len() * 8;
        let compressed = rle_compress(&values);

        let ratio = original_size as f64 / compressed.len() as f64;
        assert!(ratio > 1000.0, "Ratio was only {:.0}x", ratio);
    }

    #[test]
    fn test_rle_decoder_stateful() {
        let values: Vec<u64> = vec![1, 1, 1, 2, 2, 3];
        let compressed = rle_compress(&values);

        let mut decoder = RleDecoder::new();
        let mut offset = 0;
        let mut result = Vec::new();

        while let Some(value) = decoder.decode_next(&compressed, &mut offset).unwrap() {
            result.push(value);
        }

        assert_eq!(values, result);
    }

    #[test]
    fn test_rle_large_values() {
        let values = vec![u64::MAX; 100];

        let compressed = rle_compress(&values);
        let decompressed = rle_decompress(&compressed).unwrap();

        assert_eq!(values, decompressed);
    }

    #[test]
    fn test_rle_encoder_reset() {
        let mut encoder = RleEncoder::new();

        encoder.encode_all(&[1, 1, 2, 2]);
        let first = encoder.finish();

        let mut encoder = RleEncoder::new();
        encoder.encode_all(&[1, 1, 2, 2]);
        let second = encoder.finish();

        assert_eq!(first, second);
    }
}
