//! Block-based compression for time-series data
//!
//! This module provides block-level compression that combines multiple
//! compression techniques for optimal space efficiency.

use crate::error::{Result, TsdbError};
use crate::types::DataPoint;
use super::{CompressionType, compress_points, decompress_points};
use super::varint::{encode_varint, decode_varint};
use crc32fast::Hasher;

/// Default block size (number of points)
pub const DEFAULT_BLOCK_SIZE: usize = 1000;

/// Compressed block of time-series data
#[derive(Debug, Clone)]
pub struct CompressedBlock {
    /// Compression type used
    pub compression_type: CompressionType,
    /// First timestamp in the block
    pub min_timestamp: i64,
    /// Last timestamp in the block
    pub max_timestamp: i64,
    /// Minimum value in the block
    pub min_value: f64,
    /// Maximum value in the block
    pub max_value: f64,
    /// Number of points in the block
    pub count: u32,
    /// Compressed data
    pub data: Vec<u8>,
    /// CRC32 checksum
    pub checksum: u32,
}

impl CompressedBlock {
    /// Create a new compressed block from data points
    pub fn from_points(points: &[DataPoint]) -> Result<Self> {
        if points.is_empty() {
            return Err(TsdbError::InvalidFormat("Cannot create block from empty points".into()));
        }

        // Calculate statistics
        let min_timestamp = points.iter().map(|p| p.timestamp).min().unwrap();
        let max_timestamp = points.iter().map(|p| p.timestamp).max().unwrap();
        let min_value = points.iter().map(|p| p.value).fold(f64::INFINITY, f64::min);
        let max_value = points.iter().map(|p| p.value).fold(f64::NEG_INFINITY, f64::max);

        // Compress the data
        let data = compress_points(points)?;

        // Calculate checksum
        let mut hasher = Hasher::new();
        hasher.update(&data);
        let checksum = hasher.finalize();

        Ok(Self {
            compression_type: CompressionType::DeltaGorilla,
            min_timestamp,
            max_timestamp,
            min_value,
            max_value,
            count: points.len() as u32,
            data,
            checksum,
        })
    }

    /// Decompress the block to data points
    pub fn decompress(&self) -> Result<Vec<DataPoint>> {
        // Verify checksum
        let mut hasher = Hasher::new();
        hasher.update(&self.data);
        let actual_checksum = hasher.finalize();

        if actual_checksum != self.checksum {
            return Err(TsdbError::ChecksumMismatch {
                expected: self.checksum,
                actual: actual_checksum,
            });
        }

        decompress_points(&self.data)
    }

    /// Serialize the block to bytes
    pub fn serialize(&self) -> Vec<u8> {
        let mut result = Vec::new();

        // Write header
        result.push(self.compression_type as u8);
        result.extend_from_slice(&self.min_timestamp.to_le_bytes());
        result.extend_from_slice(&self.max_timestamp.to_le_bytes());
        result.extend_from_slice(&self.min_value.to_le_bytes());
        result.extend_from_slice(&self.max_value.to_le_bytes());
        result.extend_from_slice(&self.count.to_le_bytes());
        result.extend_from_slice(&self.checksum.to_le_bytes());

        // Write data length and data
        result.extend_from_slice(&encode_varint(self.data.len() as u64));
        result.extend_from_slice(&self.data);

        result
    }

    /// Deserialize a block from bytes
    pub fn deserialize(data: &[u8]) -> Result<(Self, usize)> {
        if data.len() < 45 {
            return Err(TsdbError::InvalidFormat("Block header too small".into()));
        }

        let mut offset = 0;

        // Read header
        let compression_type = CompressionType::try_from(data[offset])?;
        offset += 1;

        let min_timestamp = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += 8;

        let max_timestamp = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += 8;

        let min_value = f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += 8;

        let max_value = f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += 8;

        let count = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        offset += 4;

        let checksum = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        offset += 4;

        // Read data length and data
        let (data_len, bytes_read) = decode_varint(&data[offset..])?;
        offset += bytes_read;

        if offset + data_len as usize > data.len() {
            return Err(TsdbError::InvalidFormat("Block data truncated".into()));
        }

        let block_data = data[offset..offset + data_len as usize].to_vec();
        offset += data_len as usize;

        Ok((
            Self {
                compression_type,
                min_timestamp,
                max_timestamp,
                min_value,
                max_value,
                count,
                data: block_data,
                checksum,
            },
            offset,
        ))
    }

    /// Check if this block overlaps with a time range
    pub fn overlaps(&self, start: i64, end: i64) -> bool {
        self.min_timestamp <= end && self.max_timestamp >= start
    }

    /// Get the uncompressed size estimate
    pub fn uncompressed_size(&self) -> usize {
        self.count as usize * std::mem::size_of::<DataPoint>()
    }

    /// Get the compression ratio
    pub fn compression_ratio(&self) -> f64 {
        self.uncompressed_size() as f64 / self.data.len() as f64
    }
}

/// Block compressor for batching and compressing data points
#[derive(Debug)]
pub struct BlockCompressor {
    /// Maximum points per block
    block_size: usize,
    /// Current buffer
    buffer: Vec<DataPoint>,
    /// Completed blocks
    blocks: Vec<CompressedBlock>,
}

impl BlockCompressor {
    /// Create a new block compressor with default block size
    pub fn new() -> Self {
        Self::with_block_size(DEFAULT_BLOCK_SIZE)
    }

    /// Create a block compressor with custom block size
    pub fn with_block_size(block_size: usize) -> Self {
        Self {
            block_size,
            buffer: Vec::with_capacity(block_size),
            blocks: Vec::new(),
        }
    }

    /// Add a data point
    pub fn push(&mut self, point: DataPoint) -> Result<()> {
        self.buffer.push(point);

        if self.buffer.len() >= self.block_size {
            self.flush_buffer()?;
        }

        Ok(())
    }

    /// Add multiple data points
    pub fn push_all(&mut self, points: &[DataPoint]) -> Result<()> {
        for &point in points {
            self.push(point)?;
        }
        Ok(())
    }

    /// Flush the current buffer to a block
    fn flush_buffer(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        // Sort by timestamp before compressing
        self.buffer.sort_by_key(|p| p.timestamp);

        let block = CompressedBlock::from_points(&self.buffer)?;
        self.blocks.push(block);
        self.buffer.clear();

        Ok(())
    }

    /// Finish and return all blocks
    pub fn finish(mut self) -> Result<Vec<CompressedBlock>> {
        self.flush_buffer()?;
        Ok(self.blocks)
    }

    /// Get number of completed blocks
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Get number of points in current buffer
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }
}

impl Default for BlockCompressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Block decompressor for reading compressed blocks
#[derive(Debug, Default)]
pub struct BlockDecompressor {
    /// Blocks being read
    blocks: Vec<CompressedBlock>,
}

impl BlockDecompressor {
    /// Create a new decompressor
    pub fn new() -> Self {
        Self::default()
    }

    /// Load blocks from serialized data
    pub fn load(&mut self, data: &[u8]) -> Result<()> {
        let mut offset = 0;

        while offset < data.len() {
            let (block, bytes_read) = CompressedBlock::deserialize(&data[offset..])?;
            self.blocks.push(block);
            offset += bytes_read;
        }

        Ok(())
    }

    /// Get all points in a time range
    pub fn query_range(&self, start: i64, end: i64) -> Result<Vec<DataPoint>> {
        let mut result = Vec::new();

        for block in &self.blocks {
            if block.overlaps(start, end) {
                let points = block.decompress()?;
                for point in points {
                    if point.timestamp >= start && point.timestamp <= end {
                        result.push(point);
                    }
                }
            }
        }

        result.sort_by_key(|p| p.timestamp);
        Ok(result)
    }

    /// Get all points
    pub fn all_points(&self) -> Result<Vec<DataPoint>> {
        let mut result = Vec::new();

        for block in &self.blocks {
            result.extend(block.decompress()?);
        }

        result.sort_by_key(|p| p.timestamp);
        Ok(result)
    }

    /// Get block count
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_points(count: usize, start_ts: i64) -> Vec<DataPoint> {
        (0..count)
            .map(|i| DataPoint::new(start_ts + i as i64 * 60, (i as f64 * 0.1).sin() * 100.0))
            .collect()
    }

    #[test]
    fn test_compressed_block_roundtrip() {
        let points = generate_points(100, 1000);
        let block = CompressedBlock::from_points(&points).unwrap();

        assert_eq!(block.count, 100);
        assert_eq!(block.min_timestamp, 1000);
        assert_eq!(block.max_timestamp, 1000 + 99 * 60);

        let decompressed = block.decompress().unwrap();
        assert_eq!(points.len(), decompressed.len());

        for (orig, dec) in points.iter().zip(decompressed.iter()) {
            assert_eq!(orig.timestamp, dec.timestamp);
            assert!((orig.value - dec.value).abs() < 1e-10);
        }
    }

    #[test]
    fn test_block_serialize_deserialize() {
        let points = generate_points(50, 5000);
        let block = CompressedBlock::from_points(&points).unwrap();

        let serialized = block.serialize();
        let (deserialized, bytes_read) = CompressedBlock::deserialize(&serialized).unwrap();

        assert_eq!(bytes_read, serialized.len());
        assert_eq!(block.count, deserialized.count);
        assert_eq!(block.min_timestamp, deserialized.min_timestamp);
        assert_eq!(block.max_timestamp, deserialized.max_timestamp);
        assert_eq!(block.checksum, deserialized.checksum);
    }

    #[test]
    fn test_block_overlaps() {
        let points = generate_points(100, 1000);
        let block = CompressedBlock::from_points(&points).unwrap();

        // Block: 1000 to 6940 (1000 + 99*60)
        assert!(block.overlaps(500, 1500)); // Overlaps start
        assert!(block.overlaps(6000, 7000)); // Overlaps end
        assert!(block.overlaps(2000, 5000)); // Inside
        assert!(block.overlaps(0, 10000)); // Contains
        assert!(!block.overlaps(0, 500)); // Before
        assert!(!block.overlaps(10000, 20000)); // After
    }

    #[test]
    fn test_block_compressor() {
        let mut compressor = BlockCompressor::with_block_size(100);

        for i in 0..250 {
            compressor.push(DataPoint::new(i * 60, i as f64)).unwrap();
        }

        let blocks = compressor.finish().unwrap();
        assert_eq!(blocks.len(), 3); // 100 + 100 + 50

        // Verify all points are present
        let mut total_points = 0;
        for block in &blocks {
            total_points += block.count;
        }
        assert_eq!(total_points, 250);
    }

    #[test]
    fn test_block_decompressor() {
        let points = generate_points(200, 0);

        let mut compressor = BlockCompressor::with_block_size(50);
        compressor.push_all(&points).unwrap();
        let blocks = compressor.finish().unwrap();

        // Serialize all blocks
        let mut serialized = Vec::new();
        for block in &blocks {
            serialized.extend(block.serialize());
        }

        // Load and query
        let mut decompressor = BlockDecompressor::new();
        decompressor.load(&serialized).unwrap();

        assert_eq!(decompressor.block_count(), 4); // 200 / 50 = 4

        let all = decompressor.all_points().unwrap();
        assert_eq!(all.len(), 200);

        // Query specific range
        let range = decompressor.query_range(1000, 2000).unwrap();
        assert!(!range.is_empty());
        assert!(range.iter().all(|p| p.timestamp >= 1000 && p.timestamp <= 2000));
    }

    #[test]
    fn test_checksum_verification() {
        let points = generate_points(50, 1000);
        let mut block = CompressedBlock::from_points(&points).unwrap();

        // Corrupt the data
        if !block.data.is_empty() {
            block.data[0] ^= 0xFF;
        }

        let result = block.decompress();
        assert!(matches!(result, Err(TsdbError::ChecksumMismatch { .. })));
    }

    #[test]
    fn test_compression_ratio() {
        let points = generate_points(1000, 0);
        let block = CompressedBlock::from_points(&points).unwrap();

        let ratio = block.compression_ratio();
        assert!(ratio > 1.0, "Compression ratio was only {:.2}x", ratio);
    }

    #[test]
    fn test_empty_compressor() {
        let compressor = BlockCompressor::new();
        let blocks = compressor.finish().unwrap();
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_single_point_block() {
        let points = vec![DataPoint::new(1000, 42.0)];
        let block = CompressedBlock::from_points(&points).unwrap();

        assert_eq!(block.count, 1);
        assert_eq!(block.min_timestamp, 1000);
        assert_eq!(block.max_timestamp, 1000);

        let decompressed = block.decompress().unwrap();
        assert_eq!(decompressed.len(), 1);
        assert_eq!(decompressed[0].timestamp, 1000);
        assert_eq!(decompressed[0].value, 42.0);
    }

    #[test]
    fn test_block_statistics() {
        let points = vec![
            DataPoint::new(100, 10.0),
            DataPoint::new(200, 50.0),
            DataPoint::new(300, 30.0),
        ];

        let block = CompressedBlock::from_points(&points).unwrap();

        assert_eq!(block.min_timestamp, 100);
        assert_eq!(block.max_timestamp, 300);
        assert_eq!(block.min_value, 10.0);
        assert_eq!(block.max_value, 50.0);
    }
}
