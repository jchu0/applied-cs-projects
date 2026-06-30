//! Dictionary encoding for string values
//!
//! Dictionary encoding is effective for columns with low cardinality (few unique values).
//! Instead of storing the full string, we store an integer index into a dictionary.
//! This is particularly useful for:
//! - Tag keys and values in time-series data
//! - Categorical data
//! - Status/state strings

use std::collections::HashMap;
use crate::error::{Result, TsdbError};
use super::varint::{encode_varint, decode_varint};

/// Dictionary encoder for string values
#[derive(Debug, Default)]
pub struct DictionaryEncoder {
    /// Mapping from string to index
    dictionary: HashMap<String, u32>,
    /// Reverse mapping from index to string (for serialization)
    strings: Vec<String>,
    /// Encoded indices
    indices: Vec<u32>,
}

impl DictionaryEncoder {
    /// Create a new dictionary encoder
    pub fn new() -> Self {
        Self::default()
    }

    /// Encode a single value
    pub fn encode(&mut self, value: &str) -> u32 {
        if let Some(&idx) = self.dictionary.get(value) {
            self.indices.push(idx);
            idx
        } else {
            let idx = self.strings.len() as u32;
            self.dictionary.insert(value.to_string(), idx);
            self.strings.push(value.to_string());
            self.indices.push(idx);
            idx
        }
    }

    /// Encode multiple values
    pub fn encode_all(&mut self, values: &[&str]) {
        for value in values {
            self.encode(value);
        }
    }

    /// Get the dictionary size (number of unique values)
    pub fn dictionary_size(&self) -> usize {
        self.strings.len()
    }

    /// Get the number of encoded values
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Check if no values have been encoded
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Serialize the encoded data
    pub fn finish(self) -> Vec<u8> {
        let mut result = Vec::new();

        // Write dictionary size
        result.extend_from_slice(&encode_varint(self.strings.len() as u64));

        // Write each dictionary entry
        for s in &self.strings {
            result.extend_from_slice(&encode_varint(s.len() as u64));
            result.extend_from_slice(s.as_bytes());
        }

        // Write number of indices
        result.extend_from_slice(&encode_varint(self.indices.len() as u64));

        // Write indices
        for &idx in &self.indices {
            result.extend_from_slice(&encode_varint(idx as u64));
        }

        result
    }

    /// Reset the encoder for reuse
    pub fn reset(&mut self) {
        self.dictionary.clear();
        self.strings.clear();
        self.indices.clear();
    }
}

/// Dictionary decoder for string values
#[derive(Debug, Default)]
pub struct DictionaryDecoder {
    /// Dictionary mapping index to string
    dictionary: Vec<String>,
}

impl DictionaryDecoder {
    /// Create a new decoder
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode all values from the buffer
    pub fn decode_all(&mut self, data: &[u8]) -> Result<Vec<String>> {
        let mut offset = 0;

        // Read dictionary size
        let (dict_size, bytes_read) = decode_varint(data)?;
        offset += bytes_read;

        // Read dictionary entries
        self.dictionary.clear();
        for _ in 0..dict_size {
            let (str_len, bytes_read) = decode_varint(&data[offset..])?;
            offset += bytes_read;

            if offset + str_len as usize > data.len() {
                return Err(TsdbError::Decompression("Incomplete dictionary entry".into()));
            }

            let s = String::from_utf8(data[offset..offset + str_len as usize].to_vec())
                .map_err(|e| TsdbError::Decompression(format!("Invalid UTF-8: {}", e)))?;
            self.dictionary.push(s);
            offset += str_len as usize;
        }

        // Read number of indices
        let (num_indices, bytes_read) = decode_varint(&data[offset..])?;
        offset += bytes_read;

        // Read and decode indices
        let mut result = Vec::with_capacity(num_indices as usize);
        for _ in 0..num_indices {
            let (idx, bytes_read) = decode_varint(&data[offset..])?;
            offset += bytes_read;

            if idx as usize >= self.dictionary.len() {
                return Err(TsdbError::Decompression(format!(
                    "Invalid dictionary index: {} (dict size: {})",
                    idx,
                    self.dictionary.len()
                )));
            }

            result.push(self.dictionary[idx as usize].clone());
        }

        Ok(result)
    }

    /// Get a value by index from the current dictionary
    pub fn get(&self, index: u32) -> Option<&str> {
        self.dictionary.get(index as usize).map(|s| s.as_str())
    }

    /// Get the dictionary size
    pub fn dictionary_size(&self) -> usize {
        self.dictionary.len()
    }
}

/// Compress strings using dictionary encoding
pub fn dict_compress(values: &[&str]) -> Vec<u8> {
    let mut encoder = DictionaryEncoder::new();
    encoder.encode_all(values);
    encoder.finish()
}

/// Decompress dictionary-encoded strings
pub fn dict_decompress(data: &[u8]) -> Result<Vec<String>> {
    let mut decoder = DictionaryDecoder::new();
    decoder.decode_all(data)
}

/// Calculate compression ratio for dictionary encoding
pub fn estimate_dict_compression(values: &[&str]) -> f64 {
    if values.is_empty() {
        return 1.0;
    }

    // Original size: sum of string lengths
    let original_size: usize = values.iter().map(|s| s.len()).sum();

    // Build dictionary
    let mut unique: HashMap<&str, u32> = HashMap::new();
    let mut dict_size = 0usize;

    for &s in values {
        if !unique.contains_key(s) {
            unique.insert(s, unique.len() as u32);
            dict_size += s.len() + 1; // +1 for length prefix
        }
    }

    // Compressed size: dictionary + indices
    let index_size = values.len(); // 1 byte per index for small dictionaries

    let compressed_size = dict_size + index_size;

    original_size as f64 / compressed_size as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_basic() {
        let values = vec!["apple", "banana", "apple", "cherry", "banana", "apple"];

        let compressed = dict_compress(&values);
        let decompressed = dict_decompress(&compressed).unwrap();

        let expected: Vec<String> = values.iter().map(|s| s.to_string()).collect();
        assert_eq!(expected, decompressed);
    }

    #[test]
    fn test_dictionary_single_value() {
        let values = vec!["test"];

        let compressed = dict_compress(&values);
        let decompressed = dict_decompress(&compressed).unwrap();

        assert_eq!(vec!["test".to_string()], decompressed);
    }

    #[test]
    fn test_dictionary_empty() {
        let values: Vec<&str> = vec![];

        let compressed = dict_compress(&values);
        let decompressed = dict_decompress(&compressed).unwrap();

        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_dictionary_all_unique() {
        let values = vec!["a", "b", "c", "d", "e"];

        let compressed = dict_compress(&values);
        let decompressed = dict_decompress(&compressed).unwrap();

        let expected: Vec<String> = values.iter().map(|s| s.to_string()).collect();
        assert_eq!(expected, decompressed);
    }

    #[test]
    fn test_dictionary_all_same() {
        let values = vec!["repeated"; 1000];

        let compressed = dict_compress(&values);
        let decompressed = dict_decompress(&compressed).unwrap();

        assert_eq!(decompressed.len(), 1000);
        assert!(decompressed.iter().all(|s| s == "repeated"));

        // Should be compressed (dictionary encoding reduces repeated values)
        let original_size = values.len() * values[0].len();
        assert!(compressed.len() < original_size);
    }

    #[test]
    fn test_dictionary_special_chars() {
        let values = vec!["hello world", "foo\nbar", "unicode: \u{1F600}", ""];

        let compressed = dict_compress(&values);
        let decompressed = dict_decompress(&compressed).unwrap();

        let expected: Vec<String> = values.iter().map(|s| s.to_string()).collect();
        assert_eq!(expected, decompressed);
    }

    #[test]
    fn test_dictionary_encoder_methods() {
        let mut encoder = DictionaryEncoder::new();

        assert_eq!(encoder.encode("a"), 0);
        assert_eq!(encoder.encode("b"), 1);
        assert_eq!(encoder.encode("a"), 0); // Reuses index

        assert_eq!(encoder.dictionary_size(), 2);
        assert_eq!(encoder.len(), 3);
    }

    #[test]
    fn test_dictionary_decoder_get() {
        let values = vec!["apple", "banana"];
        let compressed = dict_compress(&values);

        let mut decoder = DictionaryDecoder::new();
        let _ = decoder.decode_all(&compressed).unwrap();

        assert_eq!(decoder.get(0), Some("apple"));
        assert_eq!(decoder.get(1), Some("banana"));
        assert_eq!(decoder.get(2), None);
    }

    #[test]
    fn test_estimate_compression() {
        // High repetition = good compression
        let repeated: Vec<&str> = vec!["status_ok"; 100];
        let ratio = estimate_dict_compression(&repeated);
        assert!(ratio > 5.0);

        // All unique = poor compression
        let unique: Vec<&str> = (0..10).map(|_| "x").collect();
        let ratio = estimate_dict_compression(&unique);
        assert!(ratio > 0.5);
    }

    #[test]
    fn test_dictionary_long_strings() {
        let long_string = "a".repeat(1000);
        let values: Vec<&str> = vec![&long_string; 100];

        let compressed = dict_compress(&values);
        let decompressed = dict_decompress(&compressed).unwrap();

        assert_eq!(decompressed.len(), 100);
        assert!(decompressed.iter().all(|s| s == &long_string));

        // Should be highly compressed
        let original_size = 1000 * 100;
        assert!(compressed.len() < original_size / 50);
    }

    #[test]
    fn test_dictionary_encoder_reset() {
        let mut encoder = DictionaryEncoder::new();

        encoder.encode("test1");
        encoder.encode("test2");
        encoder.reset();

        assert!(encoder.is_empty());
        assert_eq!(encoder.dictionary_size(), 0);

        // Should start fresh
        assert_eq!(encoder.encode("test1"), 0);
    }

    #[test]
    fn test_invalid_index() {
        // Manually create invalid data with an out-of-bounds index
        let mut data = Vec::new();
        data.extend_from_slice(&encode_varint(1)); // dict size = 1
        data.extend_from_slice(&encode_varint(4)); // string length = 4
        data.extend_from_slice(b"test");
        data.extend_from_slice(&encode_varint(1)); // 1 index
        data.extend_from_slice(&encode_varint(5)); // index 5 (out of bounds!)

        let result = dict_decompress(&data);
        assert!(result.is_err());
    }
}
