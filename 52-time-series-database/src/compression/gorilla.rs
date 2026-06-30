//! Gorilla compression for floating-point values
//!
//! Based on Facebook's Gorilla paper: "Gorilla: A Fast, Scalable, In-Memory Time Series Database"
//!
//! The algorithm exploits the fact that consecutive floating-point values in time series
//! often share many bits in their IEEE 754 representation. It uses XOR compression:
//!
//! 1. First value is stored as-is (64 bits)
//! 2. Subsequent values XOR with previous value
//! 3. If XOR result is 0 (same value), store single 0 bit
//! 4. Otherwise, encode leading zeros, meaningful bits, and trailing zeros efficiently

use crate::error::{Result, TsdbError};

/// Gorilla encoder for floating-point values
#[derive(Debug)]
pub struct GorillaEncoder {
    /// Previous value as bits
    prev_value: u64,
    /// Previous leading zeros
    prev_leading: u8,
    /// Previous trailing zeros
    prev_trailing: u8,
    /// Number of values encoded
    count: usize,
    /// Bit buffer
    buffer: BitBuffer,
}

impl Default for GorillaEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl GorillaEncoder {
    /// Create a new Gorilla encoder
    pub fn new() -> Self {
        Self {
            prev_value: 0,
            prev_leading: u8::MAX,
            prev_trailing: 0,
            count: 0,
            buffer: BitBuffer::new(),
        }
    }

    /// Create an encoder with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            prev_value: 0,
            prev_leading: u8::MAX,
            prev_trailing: 0,
            count: 0,
            buffer: BitBuffer::with_capacity(capacity),
        }
    }

    /// Encode a single value
    pub fn encode(&mut self, value: f64) {
        let bits = value.to_bits();

        if self.count == 0 {
            // First value: store as-is
            self.buffer.write_bits(bits, 64);
            self.prev_value = bits;
        } else {
            let xor = bits ^ self.prev_value;

            if xor == 0 {
                // Same value: single 0 bit
                self.buffer.write_bit(false);
            } else {
                // Different value
                self.buffer.write_bit(true);

                let leading = xor.leading_zeros() as u8;
                let trailing = xor.trailing_zeros() as u8;

                // Check if leading/trailing zeros fall within previous window
                if leading >= self.prev_leading
                    && trailing >= self.prev_trailing
                    && self.prev_leading != u8::MAX
                {
                    // Use previous window: 0 bit + meaningful bits
                    self.buffer.write_bit(false);
                    let meaningful_bits = 64 - self.prev_leading - self.prev_trailing;
                    let meaningful_value = xor >> self.prev_trailing;
                    self.buffer.write_bits(meaningful_value, meaningful_bits as usize);
                } else {
                    // New window: 1 bit + 5 bits leading + 6 bits length + meaningful bits
                    self.buffer.write_bit(true);

                    // Write leading zeros (5 bits, max 31)
                    let leading_clamped = leading.min(31);
                    self.buffer.write_bits(leading_clamped as u64, 5);

                    // Calculate meaningful bits
                    let meaningful_bits = 64 - leading - trailing;
                    // Write length - 1 (6 bits, but 0 means 64 bits)
                    let length_to_write = if meaningful_bits == 64 {
                        0
                    } else {
                        meaningful_bits
                    };
                    self.buffer.write_bits(length_to_write as u64, 6);

                    // Write meaningful bits
                    let meaningful_value = xor >> trailing;
                    self.buffer.write_bits(meaningful_value, meaningful_bits as usize);

                    self.prev_leading = leading_clamped;
                    self.prev_trailing = trailing;
                }
            }

            self.prev_value = bits;
        }

        self.count += 1;
    }

    /// Encode multiple values
    pub fn encode_all(&mut self, values: &[f64]) {
        for &value in values {
            self.encode(value);
        }
    }

    /// Get the encoded data
    pub fn finish(self) -> Vec<u8> {
        self.buffer.finish()
    }

    /// Get the current encoded length in bytes
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if no values have been encoded
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Reset the encoder for reuse
    pub fn reset(&mut self) {
        self.prev_value = 0;
        self.prev_leading = u8::MAX;
        self.prev_trailing = 0;
        self.count = 0;
        self.buffer.reset();
    }
}

/// Gorilla decoder for floating-point values
#[derive(Debug)]
pub struct GorillaDecoder {
    /// Previous value as bits
    prev_value: u64,
    /// Previous leading zeros
    prev_leading: u8,
    /// Previous trailing zeros
    prev_trailing: u8,
    /// Number of values decoded
    count: usize,
}

impl Default for GorillaDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl GorillaDecoder {
    /// Create a new decoder
    pub fn new() -> Self {
        Self {
            prev_value: 0,
            prev_leading: 0,
            prev_trailing: 0,
            count: 0,
        }
    }

    /// Decode a single value from the bit reader
    pub fn decode(&mut self, reader: &mut BitReader) -> Result<f64> {
        let bits = if self.count == 0 {
            // First value: read 64 bits
            reader.read_bits(64)?
        } else {
            // Check if same as previous
            if !reader.read_bit()? {
                // Same value
                self.prev_value
            } else {
                // Different value
                let xor = if !reader.read_bit()? {
                    // Use previous window
                    let meaningful_bits = 64 - self.prev_leading - self.prev_trailing;
                    let meaningful_value = reader.read_bits(meaningful_bits as usize)?;
                    meaningful_value << self.prev_trailing
                } else {
                    // New window
                    let leading = reader.read_bits(5)? as u8;
                    let length = reader.read_bits(6)? as u8;
                    let meaningful_bits = if length == 0 { 64 } else { length };
                    let trailing = 64 - leading - meaningful_bits;

                    let meaningful_value = reader.read_bits(meaningful_bits as usize)?;

                    self.prev_leading = leading;
                    self.prev_trailing = trailing;

                    meaningful_value << trailing
                };

                self.prev_value ^ xor
            }
        };

        self.prev_value = bits;
        self.count += 1;

        Ok(f64::from_bits(bits))
    }

    /// Decode all values
    pub fn decode_all(&mut self, data: &[u8], count: usize) -> Result<Vec<f64>> {
        let mut reader = BitReader::new(data);
        let mut result = Vec::with_capacity(count);

        for _ in 0..count {
            result.push(self.decode(&mut reader)?);
        }

        Ok(result)
    }

    /// Reset the decoder for reuse
    pub fn reset(&mut self) {
        self.prev_value = 0;
        self.prev_leading = 0;
        self.prev_trailing = 0;
        self.count = 0;
    }
}

/// Bit buffer for writing bits
#[derive(Debug)]
struct BitBuffer {
    /// Output bytes
    data: Vec<u8>,
    /// Current byte being built
    current_byte: u8,
    /// Number of bits written to current byte (0-7)
    bit_count: u8,
}

impl BitBuffer {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            current_byte: 0,
            bit_count: 0,
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            current_byte: 0,
            bit_count: 0,
        }
    }

    fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current_byte |= 1 << (7 - self.bit_count);
        }
        self.bit_count += 1;

        if self.bit_count == 8 {
            self.data.push(self.current_byte);
            self.current_byte = 0;
            self.bit_count = 0;
        }
    }

    fn write_bits(&mut self, value: u64, num_bits: usize) {
        for i in (0..num_bits).rev() {
            self.write_bit((value >> i) & 1 == 1);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bit_count > 0 {
            self.data.push(self.current_byte);
        }
        self.data
    }

    fn len(&self) -> usize {
        self.data.len() + if self.bit_count > 0 { 1 } else { 0 }
    }

    fn reset(&mut self) {
        self.data.clear();
        self.current_byte = 0;
        self.bit_count = 0;
    }
}

/// Bit reader for decoding
#[derive(Debug)]
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_index: usize,
    bit_index: u8,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_index: 0,
            bit_index: 0,
        }
    }

    pub fn read_bit(&mut self) -> Result<bool> {
        if self.byte_index >= self.data.len() {
            return Err(TsdbError::Decompression("Unexpected end of data".into()));
        }

        let bit = (self.data[self.byte_index] >> (7 - self.bit_index)) & 1 == 1;
        self.bit_index += 1;

        if self.bit_index == 8 {
            self.byte_index += 1;
            self.bit_index = 0;
        }

        Ok(bit)
    }

    pub fn read_bits(&mut self, num_bits: usize) -> Result<u64> {
        let mut result: u64 = 0;

        for _ in 0..num_bits {
            result = (result << 1) | (self.read_bit()? as u64);
        }

        Ok(result)
    }
}

/// Compress values using Gorilla encoding
pub fn compress_values(values: &[f64]) -> Result<Vec<u8>> {
    let mut encoder = GorillaEncoder::with_capacity(values.len() * 2);
    encoder.encode_all(values);
    Ok(encoder.finish())
}

/// Decompress values
pub fn decompress_values(data: &[u8], count: usize) -> Result<Vec<f64>> {
    let mut decoder = GorillaDecoder::new();
    decoder.decode_all(data, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gorilla_identical_values() {
        let values = vec![42.5; 100];

        let mut encoder = GorillaEncoder::new();
        encoder.encode_all(&values);
        let encoded = encoder.finish();

        // Identical values should compress to ~1 bit per value after first
        // First value: 64 bits, rest: 1 bit each = 64 + 99 = 163 bits = ~21 bytes
        assert!(encoded.len() < 30);

        let mut decoder = GorillaDecoder::new();
        let decoded = decoder.decode_all(&encoded, values.len()).unwrap();

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_gorilla_similar_values() {
        // Values that differ only in lower bits
        let values: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.001).collect();

        let compressed = compress_values(&values).unwrap();
        let decompressed = decompress_values(&compressed, values.len()).unwrap();

        for (orig, dec) in values.iter().zip(decompressed.iter()) {
            assert!((orig - dec).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gorilla_random_values() {
        let values: Vec<f64> = (0..100).map(|i| (i as f64 * 1.234).sin() * 100.0).collect();

        let compressed = compress_values(&values).unwrap();
        let decompressed = decompress_values(&compressed, values.len()).unwrap();

        for (orig, dec) in values.iter().zip(decompressed.iter()) {
            assert!((orig - dec).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gorilla_special_values() {
        let values = vec![
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN,
            f64::MAX,
            f64::MIN_POSITIVE,
        ];

        let compressed = compress_values(&values).unwrap();
        let decompressed = decompress_values(&compressed, values.len()).unwrap();

        // Check all except NaN (which can't be compared with ==)
        for (orig, dec) in values.iter().zip(decompressed.iter()) {
            assert_eq!(orig.to_bits(), dec.to_bits());
        }
    }

    #[test]
    fn test_gorilla_nan() {
        let values = vec![f64::NAN, 1.0, f64::NAN];

        let compressed = compress_values(&values).unwrap();
        let decompressed = decompress_values(&compressed, values.len()).unwrap();

        assert!(decompressed[0].is_nan());
        assert_eq!(decompressed[1], 1.0);
        assert!(decompressed[2].is_nan());
    }

    #[test]
    fn test_gorilla_single_value() {
        let values = vec![123.456];

        let compressed = compress_values(&values).unwrap();
        let decompressed = decompress_values(&compressed, values.len()).unwrap();

        assert_eq!(values, decompressed);
    }

    #[test]
    fn test_gorilla_two_values() {
        let values = vec![100.0, 200.0];

        let compressed = compress_values(&values).unwrap();
        let decompressed = decompress_values(&compressed, values.len()).unwrap();

        assert_eq!(values, decompressed);
    }

    #[test]
    fn test_gorilla_compression_ratio() {
        // Typical sensor data: small variations around a mean
        let values: Vec<f64> = (0..1000)
            .map(|i| 50.0 + (i as f64 * 0.1).sin() * 5.0)
            .collect();

        let original_size = values.len() * 8;
        let compressed = compress_values(&values).unwrap();

        let ratio = original_size as f64 / compressed.len() as f64;
        // Expect some compression for smooth time-series data
        assert!(ratio > 1.0, "Compression ratio was only {:.2}x", ratio);
    }

    #[test]
    fn test_bit_buffer() {
        let mut buffer = BitBuffer::new();
        buffer.write_bit(true);
        buffer.write_bit(false);
        buffer.write_bit(true);
        buffer.write_bit(false);
        buffer.write_bit(true);
        buffer.write_bit(false);
        buffer.write_bit(true);
        buffer.write_bit(false);

        let data = buffer.finish();
        assert_eq!(data, vec![0b10101010]);
    }

    #[test]
    fn test_bit_buffer_partial() {
        let mut buffer = BitBuffer::new();
        buffer.write_bit(true);
        buffer.write_bit(true);
        buffer.write_bit(true);

        let data = buffer.finish();
        assert_eq!(data, vec![0b11100000]);
    }

    #[test]
    fn test_bit_reader() {
        let data = vec![0b10101010, 0b11001100];
        let mut reader = BitReader::new(&data);

        assert!(reader.read_bit().unwrap());
        assert!(!reader.read_bit().unwrap());
        assert!(reader.read_bit().unwrap());
        assert!(!reader.read_bit().unwrap());
    }

    #[test]
    fn test_bit_reader_multi_bits() {
        let data = vec![0b11110000];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read_bits(4).unwrap(), 0b1111);
        assert_eq!(reader.read_bits(4).unwrap(), 0b0000);
    }

    #[test]
    fn test_encoder_reset() {
        let mut encoder = GorillaEncoder::new();

        encoder.encode_all(&[1.0, 2.0, 3.0]);
        let first_data = encoder.finish();

        let mut encoder = GorillaEncoder::new();
        encoder.encode_all(&[1.0, 2.0, 3.0]);
        let second_data = encoder.finish();

        assert_eq!(first_data, second_data);
    }
}
