//! Delta encoding for timestamps
//!
//! Delta encoding stores the difference between consecutive values instead of
//! absolute values. This is highly effective for timestamps in time-series data
//! because the intervals are often regular or semi-regular.
//!
//! We use delta-of-delta encoding for even better compression:
//! - First value: stored as-is
//! - Second value: stored as delta from first
//! - Subsequent values: stored as delta-of-delta (change in delta)

use crate::error::{Result, TsdbError};
use super::varint::{encode_signed_varint, decode_signed_varint, encode_varint, decode_varint};

/// Delta encoder for timestamp sequences
#[derive(Debug, Default)]
pub struct DeltaEncoder {
    /// Previous value
    prev_value: i64,
    /// Previous delta (for delta-of-delta)
    prev_delta: i64,
    /// Number of values encoded
    count: usize,
    /// Output buffer
    buffer: Vec<u8>,
}

impl DeltaEncoder {
    /// Create a new delta encoder
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an encoder with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            ..Default::default()
        }
    }

    /// Encode a single value
    pub fn encode(&mut self, value: i64) {
        match self.count {
            0 => {
                // First value: store as signed varint
                self.buffer.extend_from_slice(&encode_signed_varint(value));
                self.prev_value = value;
            }
            1 => {
                // Second value: store delta
                let delta = value - self.prev_value;
                self.buffer.extend_from_slice(&encode_signed_varint(delta));
                self.prev_delta = delta;
                self.prev_value = value;
            }
            _ => {
                // Subsequent values: store delta-of-delta
                let delta = value - self.prev_value;
                let delta_of_delta = delta - self.prev_delta;
                self.buffer.extend_from_slice(&encode_signed_varint(delta_of_delta));
                self.prev_delta = delta;
                self.prev_value = value;
            }
        }
        self.count += 1;
    }

    /// Encode multiple values
    pub fn encode_all(&mut self, values: &[i64]) {
        for &value in values {
            self.encode(value);
        }
    }

    /// Get the encoded data
    pub fn finish(self) -> Vec<u8> {
        self.buffer
    }

    /// Get the current encoded data without consuming the encoder
    pub fn data(&self) -> &[u8] {
        &self.buffer
    }

    /// Reset the encoder for reuse
    pub fn reset(&mut self) {
        self.prev_value = 0;
        self.prev_delta = 0;
        self.count = 0;
        self.buffer.clear();
    }
}

/// Delta decoder for timestamp sequences
#[derive(Debug, Default)]
pub struct DeltaDecoder {
    /// Previous value
    prev_value: i64,
    /// Previous delta
    prev_delta: i64,
    /// Number of values decoded
    count: usize,
}

impl DeltaDecoder {
    /// Create a new decoder
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode a single value, advancing the offset
    pub fn decode(&mut self, data: &[u8], offset: &mut usize) -> Result<i64> {
        let (raw_value, bytes_read) = decode_signed_varint(&data[*offset..])?;
        *offset += bytes_read;

        let value = match self.count {
            0 => {
                // First value: stored as-is
                self.prev_value = raw_value;
                raw_value
            }
            1 => {
                // Second value: stored as delta
                let value = self.prev_value + raw_value;
                self.prev_delta = raw_value;
                self.prev_value = value;
                value
            }
            _ => {
                // Subsequent values: stored as delta-of-delta
                let delta = self.prev_delta + raw_value;
                let value = self.prev_value + delta;
                self.prev_delta = delta;
                self.prev_value = value;
                value
            }
        };

        self.count += 1;
        Ok(value)
    }

    /// Decode all values from the buffer
    pub fn decode_all(&mut self, data: &[u8], count: usize) -> Result<Vec<i64>> {
        let mut result = Vec::with_capacity(count);
        let mut offset = 0;

        for _ in 0..count {
            if offset >= data.len() {
                return Err(TsdbError::Decompression("Unexpected end of data".into()));
            }
            result.push(self.decode(data, &mut offset)?);
        }

        Ok(result)
    }

    /// Reset the decoder for reuse
    pub fn reset(&mut self) {
        self.prev_value = 0;
        self.prev_delta = 0;
        self.count = 0;
    }
}

/// Compress a slice of timestamps using delta encoding
pub fn compress_timestamps(timestamps: &[i64]) -> Result<Vec<u8>> {
    if timestamps.is_empty() {
        return Ok(Vec::new());
    }

    // Write count first
    let mut result = encode_varint(timestamps.len() as u64);

    // Delta encode timestamps
    let mut encoder = DeltaEncoder::with_capacity(timestamps.len() * 2);
    encoder.encode_all(timestamps);
    result.extend_from_slice(&encoder.finish());

    Ok(result)
}

/// Decompress timestamps
pub fn decompress_timestamps(data: &[u8]) -> Result<Vec<i64>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    // Read count
    let (count, bytes_read) = decode_varint(data)?;

    // Decode timestamps
    let mut decoder = DeltaDecoder::new();
    decoder.decode_all(&data[bytes_read..], count as usize)
}

/// Simple delta encoding (not delta-of-delta) for cases where it's more appropriate
pub fn simple_delta_encode(values: &[i64]) -> Vec<u8> {
    if values.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(values.len() * 2);

    // Store first value
    result.extend_from_slice(&encode_signed_varint(values[0]));

    // Store deltas
    for i in 1..values.len() {
        let delta = values[i] - values[i - 1];
        result.extend_from_slice(&encode_signed_varint(delta));
    }

    result
}

/// Simple delta decoding
pub fn simple_delta_decode(data: &[u8], count: usize) -> Result<Vec<i64>> {
    if data.is_empty() || count == 0 {
        return Ok(Vec::new());
    }

    let mut result = Vec::with_capacity(count);
    let mut offset = 0;

    // Decode first value
    let (first_value, bytes_read) = decode_signed_varint(data)?;
    offset += bytes_read;
    result.push(first_value);

    // Decode deltas
    let mut prev = first_value;
    for _ in 1..count {
        if offset >= data.len() {
            return Err(TsdbError::Decompression("Unexpected end of data".into()));
        }
        let (delta, bytes_read) = decode_signed_varint(&data[offset..])?;
        offset += bytes_read;
        prev += delta;
        result.push(prev);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_encoder_regular_intervals() {
        let mut encoder = DeltaEncoder::new();
        let timestamps: Vec<i64> = (0..100).map(|i| 1000 + i * 60).collect();

        encoder.encode_all(&timestamps);
        let encoded = encoder.finish();

        // Regular intervals should compress very well
        // Original: 100 * 8 = 800 bytes
        // Encoded should be much smaller due to delta-of-delta being mostly 0
        assert!(encoded.len() < 200);
    }

    #[test]
    fn test_delta_roundtrip() {
        let timestamps: Vec<i64> = (0..100).map(|i| 1000000 + i * 60000).collect();

        let compressed = compress_timestamps(&timestamps).unwrap();
        let decompressed = decompress_timestamps(&compressed).unwrap();

        assert_eq!(timestamps, decompressed);
    }

    #[test]
    fn test_delta_irregular_intervals() {
        let timestamps: Vec<i64> = vec![
            1000, 1060, 1120, 1200, 1300, 1450, 1600, 1850, 2100, 2500,
        ];

        let compressed = compress_timestamps(&timestamps).unwrap();
        let decompressed = decompress_timestamps(&compressed).unwrap();

        assert_eq!(timestamps, decompressed);
    }

    #[test]
    fn test_delta_negative_deltas() {
        // Out of order timestamps (shouldn't happen but should still work)
        let timestamps: Vec<i64> = vec![1000, 900, 800, 700, 600];

        let compressed = compress_timestamps(&timestamps).unwrap();
        let decompressed = decompress_timestamps(&compressed).unwrap();

        assert_eq!(timestamps, decompressed);
    }

    #[test]
    fn test_delta_single_value() {
        let timestamps = vec![1000i64];

        let compressed = compress_timestamps(&timestamps).unwrap();
        let decompressed = decompress_timestamps(&compressed).unwrap();

        assert_eq!(timestamps, decompressed);
    }

    #[test]
    fn test_delta_two_values() {
        let timestamps = vec![1000i64, 2000];

        let compressed = compress_timestamps(&timestamps).unwrap();
        let decompressed = decompress_timestamps(&compressed).unwrap();

        assert_eq!(timestamps, decompressed);
    }

    #[test]
    fn test_delta_empty() {
        let timestamps: Vec<i64> = vec![];

        let compressed = compress_timestamps(&timestamps).unwrap();
        let decompressed = decompress_timestamps(&compressed).unwrap();

        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_simple_delta_encode_decode() {
        let values: Vec<i64> = vec![100, 150, 180, 250, 320, 400];

        let encoded = simple_delta_encode(&values);
        let decoded = simple_delta_decode(&encoded, values.len()).unwrap();

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_encoder_reset() {
        let mut encoder = DeltaEncoder::new();

        // First batch
        encoder.encode_all(&[100, 200, 300]);
        let first_data = encoder.data().to_vec();

        // Reset and encode different data
        encoder.reset();
        encoder.encode_all(&[100, 200, 300]);
        let second_data = encoder.data().to_vec();

        // Should produce identical output
        assert_eq!(first_data, second_data);
    }

    #[test]
    fn test_large_values() {
        let timestamps: Vec<i64> = (0..100)
            .map(|i| 1_700_000_000_000_000_000i64 + i * 1_000_000_000)
            .collect();

        let compressed = compress_timestamps(&timestamps).unwrap();
        let decompressed = decompress_timestamps(&compressed).unwrap();

        assert_eq!(timestamps, decompressed);
    }

    #[test]
    fn test_compression_ratio() {
        // Regular 1-minute intervals for a day
        let timestamps: Vec<i64> = (0..1440)
            .map(|i| 1_700_000_000_000_000_000i64 + i * 60_000_000_000)
            .collect();

        let original_size = timestamps.len() * 8;
        let compressed = compress_timestamps(&timestamps).unwrap();

        let ratio = original_size as f64 / compressed.len() as f64;
        // Expect at least 5x compression for regular intervals
        assert!(ratio > 5.0, "Compression ratio was only {:.2}x", ratio);
    }
}
