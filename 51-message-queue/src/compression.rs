//! Compression support for messages.

use crate::error::{Error, Result};
use bytes::{Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

/// Compression types supported by the message queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[repr(u8)]
pub enum Compression {
    /// No compression
    #[default]
    None = 0,
    /// Gzip compression
    Gzip = 1,
    /// LZ4 compression (fast)
    Lz4 = 2,
    /// Snappy compression (balanced)
    Snappy = 3,
}

impl Compression {
    /// Get compression type from byte value.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Compression::None),
            1 => Some(Compression::Gzip),
            2 => Some(Compression::Lz4),
            3 => Some(Compression::Snappy),
            _ => None,
        }
    }

    /// Convert to byte value.
    pub fn as_u8(&self) -> u8 {
        *self as u8
    }

    /// Compress data using this compression type.
    pub fn compress(&self, data: &[u8]) -> Result<Bytes> {
        match self {
            Compression::None => Ok(Bytes::copy_from_slice(data)),
            Compression::Gzip => compress_gzip(data),
            Compression::Lz4 => compress_lz4(data),
            Compression::Snappy => compress_snappy(data),
        }
    }

    /// Decompress data using this compression type.
    pub fn decompress(&self, data: &[u8]) -> Result<Bytes> {
        match self {
            Compression::None => Ok(Bytes::copy_from_slice(data)),
            Compression::Gzip => decompress_gzip(data),
            Compression::Lz4 => decompress_lz4(data),
            Compression::Snappy => decompress_snappy(data),
        }
    }

    /// Get the name of this compression type.
    pub fn name(&self) -> &'static str {
        match self {
            Compression::None => "none",
            Compression::Gzip => "gzip",
            Compression::Lz4 => "lz4",
            Compression::Snappy => "snappy",
        }
    }

    /// Estimate compressed size (rough estimate for buffer allocation).
    pub fn estimate_compressed_size(&self, uncompressed_size: usize) -> usize {
        match self {
            Compression::None => uncompressed_size,
            // Gzip typically achieves 60-80% compression on text
            Compression::Gzip => uncompressed_size / 2 + 64,
            // LZ4 is less aggressive
            Compression::Lz4 => uncompressed_size * 3 / 4 + 64,
            // Snappy is similar to LZ4
            Compression::Snappy => uncompressed_size * 3 / 4 + 64,
        }
    }
}

impl std::fmt::Display for Compression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for Compression {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "none" => Ok(Compression::None),
            "gzip" => Ok(Compression::Gzip),
            "lz4" => Ok(Compression::Lz4),
            "snappy" => Ok(Compression::Snappy),
            _ => Err(Error::InvalidConfig(format!(
                "Unknown compression type: {}",
                s
            ))),
        }
    }
}

/// Compress using gzip.
fn compress_gzip(data: &[u8]) -> Result<Bytes> {
    use flate2::write::GzEncoder;
    use flate2::Compression;

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(data)
        .map_err(|e| Error::Compression(e.to_string()))?;
    let compressed = encoder
        .finish()
        .map_err(|e| Error::Compression(e.to_string()))?;
    Ok(Bytes::from(compressed))
}

/// Decompress gzip data.
fn decompress_gzip(data: &[u8]) -> Result<Bytes> {
    use flate2::read::GzDecoder;

    let mut decoder = GzDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .map_err(|e| Error::Decompression(e.to_string()))?;
    Ok(Bytes::from(decompressed))
}

/// Compress using LZ4.
fn compress_lz4(data: &[u8]) -> Result<Bytes> {
    let compressed = lz4_flex::compress_prepend_size(data);
    Ok(Bytes::from(compressed))
}

/// Decompress LZ4 data.
fn decompress_lz4(data: &[u8]) -> Result<Bytes> {
    let decompressed = lz4_flex::decompress_size_prepended(data)
        .map_err(|e| Error::Decompression(e.to_string()))?;
    Ok(Bytes::from(decompressed))
}

/// Compress using Snappy.
fn compress_snappy(data: &[u8]) -> Result<Bytes> {
    let mut encoder = snap::raw::Encoder::new();
    let compressed = encoder
        .compress_vec(data)
        .map_err(|e| Error::Compression(e.to_string()))?;
    Ok(Bytes::from(compressed))
}

/// Decompress Snappy data.
fn decompress_snappy(data: &[u8]) -> Result<Bytes> {
    let mut decoder = snap::raw::Decoder::new();
    let decompressed = decoder
        .decompress_vec(data)
        .map_err(|e| Error::Decompression(e.to_string()))?;
    Ok(Bytes::from(decompressed))
}

/// Compression statistics.
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Number of bytes before compression
    pub uncompressed_bytes: u64,
    /// Number of bytes after compression
    pub compressed_bytes: u64,
    /// Number of compression operations
    pub compression_count: u64,
    /// Number of decompression operations
    pub decompression_count: u64,
}

impl CompressionStats {
    /// Create new empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        if self.uncompressed_bytes == 0 {
            1.0
        } else {
            self.compressed_bytes as f64 / self.uncompressed_bytes as f64
        }
    }

    /// Calculate space savings percentage.
    pub fn space_savings(&self) -> f64 {
        (1.0 - self.compression_ratio()) * 100.0
    }

    /// Record a compression operation.
    pub fn record_compression(&mut self, uncompressed: usize, compressed: usize) {
        self.uncompressed_bytes += uncompressed as u64;
        self.compressed_bytes += compressed as u64;
        self.compression_count += 1;
    }

    /// Record a decompression operation.
    pub fn record_decompression(&mut self) {
        self.decompression_count += 1;
    }

    /// Merge stats from another instance.
    pub fn merge(&mut self, other: &CompressionStats) {
        self.uncompressed_bytes += other.uncompressed_bytes;
        self.compressed_bytes += other.compressed_bytes;
        self.compression_count += other.compression_count;
        self.decompression_count += other.decompression_count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_none() {
        let data = b"Hello, World!";
        let compressed = Compression::None.compress(data).unwrap();
        assert_eq!(compressed.as_ref(), data);

        let decompressed = Compression::None.decompress(&compressed).unwrap();
        assert_eq!(decompressed.as_ref(), data);
    }

    #[test]
    fn test_compression_gzip() {
        // Use larger data to ensure compression is effective (small data may expand due to headers)
        let data: Vec<u8> = b"Hello, World! This is a test message for compression. ".repeat(20);
        let compressed = Compression::Gzip.compress(&data).unwrap();
        // Gzip should compress repetitive data effectively
        assert!(compressed.len() < data.len());

        let decompressed = Compression::Gzip.decompress(&compressed).unwrap();
        assert_eq!(decompressed.as_ref(), data.as_slice());
    }

    #[test]
    fn test_compression_lz4() {
        let data = b"Hello, World! This is a test message for compression.";
        let compressed = Compression::Lz4.compress(data).unwrap();

        let decompressed = Compression::Lz4.decompress(&compressed).unwrap();
        assert_eq!(decompressed.as_ref(), data);
    }

    #[test]
    fn test_compression_snappy() {
        let data = b"Hello, World! This is a test message for compression.";
        let compressed = Compression::Snappy.compress(data).unwrap();

        let decompressed = Compression::Snappy.decompress(&compressed).unwrap();
        assert_eq!(decompressed.as_ref(), data);
    }

    #[test]
    fn test_compression_large_data() {
        let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();

        for compression in [
            Compression::None,
            Compression::Gzip,
            Compression::Lz4,
            Compression::Snappy,
        ] {
            let compressed = compression.compress(&data).unwrap();
            let decompressed = compression.decompress(&compressed).unwrap();
            assert_eq!(decompressed.as_ref(), data.as_slice());
        }
    }

    #[test]
    fn test_compression_from_str() {
        assert_eq!("none".parse::<Compression>().unwrap(), Compression::None);
        assert_eq!("gzip".parse::<Compression>().unwrap(), Compression::Gzip);
        assert_eq!("lz4".parse::<Compression>().unwrap(), Compression::Lz4);
        assert_eq!(
            "snappy".parse::<Compression>().unwrap(),
            Compression::Snappy
        );
        assert_eq!("GZIP".parse::<Compression>().unwrap(), Compression::Gzip);
    }

    #[test]
    fn test_compression_from_u8() {
        assert_eq!(Compression::from_u8(0), Some(Compression::None));
        assert_eq!(Compression::from_u8(1), Some(Compression::Gzip));
        assert_eq!(Compression::from_u8(2), Some(Compression::Lz4));
        assert_eq!(Compression::from_u8(3), Some(Compression::Snappy));
        assert_eq!(Compression::from_u8(4), None);
    }

    #[test]
    fn test_compression_stats() {
        let mut stats = CompressionStats::new();
        stats.record_compression(1000, 500);
        stats.record_compression(1000, 600);

        assert_eq!(stats.uncompressed_bytes, 2000);
        assert_eq!(stats.compressed_bytes, 1100);
        assert_eq!(stats.compression_count, 2);
        assert!((stats.compression_ratio() - 0.55).abs() < 0.01);
        assert!((stats.space_savings() - 45.0).abs() < 1.0);
    }

    #[test]
    fn test_empty_data_compression() {
        let data = b"";
        for compression in [
            Compression::None,
            Compression::Gzip,
            Compression::Lz4,
            Compression::Snappy,
        ] {
            let compressed = compression.compress(data).unwrap();
            let decompressed = compression.decompress(&compressed).unwrap();
            assert_eq!(decompressed.as_ref(), data);
        }
    }

    #[test]
    fn test_compression_name() {
        assert_eq!(Compression::None.name(), "none");
        assert_eq!(Compression::Gzip.name(), "gzip");
        assert_eq!(Compression::Lz4.name(), "lz4");
        assert_eq!(Compression::Snappy.name(), "snappy");
    }

    #[test]
    fn test_compression_estimate() {
        let size = 1000;
        assert_eq!(Compression::None.estimate_compressed_size(size), size);
        assert!(Compression::Gzip.estimate_compressed_size(size) < size);
        assert!(Compression::Lz4.estimate_compressed_size(size) < size);
        assert!(Compression::Snappy.estimate_compressed_size(size) < size);
    }
}
