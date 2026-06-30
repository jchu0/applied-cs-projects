//! Compression algorithms for time-series data
//!
//! This module provides various compression techniques optimized for time-series data:
//! - Delta encoding for timestamps
//! - Gorilla/XOR compression for floating-point values
//! - Run-length encoding for repeated values
//! - Variable-length integer encoding (varint)
//! - Dictionary encoding for string tags

pub mod delta;
pub mod gorilla;
pub mod rle;
pub mod varint;
pub mod dictionary;
pub mod block;

pub use delta::DeltaEncoder;
pub use gorilla::GorillaEncoder;
pub use rle::RleEncoder;
pub use varint::{encode_varint, decode_varint, encode_signed_varint, decode_signed_varint};
pub use dictionary::DictionaryEncoder;
pub use block::{CompressedBlock, BlockCompressor};

use crate::error::Result;
use crate::types::DataPoint;

/// Compression type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CompressionType {
    None = 0,
    Delta = 1,
    Gorilla = 2,
    Rle = 3,
    DeltaGorilla = 4, // Combined delta + gorilla
}

impl TryFrom<u8> for CompressionType {
    type Error = crate::error::TsdbError;

    fn try_from(value: u8) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(CompressionType::None),
            1 => Ok(CompressionType::Delta),
            2 => Ok(CompressionType::Gorilla),
            3 => Ok(CompressionType::Rle),
            4 => Ok(CompressionType::DeltaGorilla),
            _ => Err(crate::error::TsdbError::InvalidFormat(format!(
                "Unknown compression type: {}",
                value
            ))),
        }
    }
}

/// Compress a series of data points using the optimal strategy
pub fn compress_points(points: &[DataPoint]) -> Result<Vec<u8>> {
    if points.is_empty() {
        return Ok(Vec::new());
    }

    let mut result = Vec::new();

    // Compress timestamps with delta encoding
    let timestamps: Vec<i64> = points.iter().map(|p| p.timestamp).collect();
    let compressed_timestamps = delta::compress_timestamps(&timestamps)?;

    // Compress values with Gorilla encoding
    let values: Vec<f64> = points.iter().map(|p| p.value).collect();
    let compressed_values = gorilla::compress_values(&values)?;

    // Write header: compression type
    result.push(CompressionType::DeltaGorilla as u8);

    // Write number of points
    let count_bytes = encode_varint(points.len() as u64);
    result.extend_from_slice(&count_bytes);

    // Write compressed timestamps length and data
    let ts_len_bytes = encode_varint(compressed_timestamps.len() as u64);
    result.extend_from_slice(&ts_len_bytes);
    result.extend_from_slice(&compressed_timestamps);

    // Write compressed values
    result.extend_from_slice(&compressed_values);

    Ok(result)
}

/// Decompress a series of data points
pub fn decompress_points(data: &[u8]) -> Result<Vec<DataPoint>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let mut offset = 0;

    // Read compression type
    let compression_type = CompressionType::try_from(data[offset])?;
    offset += 1;

    if compression_type != CompressionType::DeltaGorilla {
        return Err(crate::error::TsdbError::InvalidFormat(format!(
            "Unsupported compression type: {:?}",
            compression_type
        )));
    }

    // Read number of points
    let (count, bytes_read) = decode_varint(&data[offset..])?;
    offset += bytes_read;

    // Read compressed timestamps length
    let (ts_len, bytes_read) = decode_varint(&data[offset..])?;
    offset += bytes_read;

    // Decompress timestamps
    let timestamps = delta::decompress_timestamps(&data[offset..offset + ts_len as usize])?;
    offset += ts_len as usize;

    // Decompress values
    let values = gorilla::decompress_values(&data[offset..], count as usize)?;

    // Combine into data points
    let points: Vec<DataPoint> = timestamps
        .into_iter()
        .zip(values.into_iter())
        .map(|(timestamp, value)| DataPoint { timestamp, value })
        .collect();

    Ok(points)
}

/// Calculate compression ratio
pub fn compression_ratio(original_size: usize, compressed_size: usize) -> f64 {
    if compressed_size == 0 {
        return 0.0;
    }
    original_size as f64 / compressed_size as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_decompress_points() {
        let points: Vec<DataPoint> = (0..100)
            .map(|i| DataPoint::new(1000 + i * 60, i as f64 * 1.5))
            .collect();

        let compressed = compress_points(&points).unwrap();
        let decompressed = decompress_points(&compressed).unwrap();

        assert_eq!(points.len(), decompressed.len());
        for (orig, dec) in points.iter().zip(decompressed.iter()) {
            assert_eq!(orig.timestamp, dec.timestamp);
            assert!((orig.value - dec.value).abs() < 1e-10);
        }
    }

    #[test]
    fn test_empty_points() {
        let points: Vec<DataPoint> = vec![];
        let compressed = compress_points(&points).unwrap();
        let decompressed = decompress_points(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_compression_ratio() {
        let points: Vec<DataPoint> = (0..1000)
            .map(|i| DataPoint::new(1000000 + i * 60000, 50.0 + (i as f64 * 0.01).sin()))
            .collect();

        let original_size = points.len() * std::mem::size_of::<DataPoint>();
        let compressed = compress_points(&points).unwrap();
        let ratio = compression_ratio(original_size, compressed.len());

        // Expect at least 2x compression for regular time-series data
        assert!(ratio > 2.0, "Compression ratio was only {:.2}x", ratio);
    }
}
