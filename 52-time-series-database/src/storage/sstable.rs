//! SSTable (Sorted String Table) implementation for persistent storage
//!
//! SSTables provide immutable, sorted storage for time-series data.
//! They are organized as:
//! - Index block: series keys and their offsets
//! - Data blocks: compressed time-series data
//! - Footer: metadata and index location

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crc32fast::Hasher;

use crate::error::{Result, TsdbError};
use crate::types::{DataPoint, SeriesKey, Series, Metric};
use crate::compression::{compress_points, decompress_points};
use crate::compression::varint::{encode_varint, decode_varint};

/// Magic number for SSTable files
const SSTABLE_MAGIC: u32 = 0x54535354; // "TSST"

/// SSTable version
const SSTABLE_VERSION: u32 = 1;

/// Footer size in bytes (u64 + u64 + i64 + i64 + u32 + u32 + u32 = 44)
const FOOTER_SIZE: usize = 44;

/// SSTable file format:
/// [Data Blocks] [Index Block] [Footer]
///
/// Data Block:
///   - Series key (u64)
///   - Metric name length + metric name
///   - Tags count + tags (key-value pairs)
///   - Compressed data length + compressed data
///
/// Index Block:
///   - Number of entries (varint)
///   - For each entry: series key (u64) + offset (u64) + length (u32)
///
/// Footer:
///   - Index offset (u64)
///   - Index length (u64)
///   - Min timestamp (i64)
///   - Max timestamp (i64)
///   - Checksum (u32)
///   - Version (u32)
///   - Magic (u32)

/// SSTable metadata
#[derive(Debug, Clone)]
pub struct SSTableMeta {
    /// Path to the SSTable file
    pub path: PathBuf,
    /// Minimum timestamp in the table
    pub min_timestamp: i64,
    /// Maximum timestamp in the table
    pub max_timestamp: i64,
    /// Number of series in the table
    pub series_count: usize,
    /// Total number of data points
    pub point_count: usize,
    /// File size in bytes
    pub file_size: u64,
}

/// Index entry for a series
#[derive(Debug, Clone)]
struct IndexEntry {
    series_key: SeriesKey,
    offset: u64,
    length: u32,
    min_timestamp: i64,
    max_timestamp: i64,
}

/// SSTable builder for creating new SSTables
#[derive(Debug)]
pub struct SSTableBuilder {
    /// Output path
    path: PathBuf,
    /// Writer
    writer: BufWriter<File>,
    /// Index entries
    index: Vec<IndexEntry>,
    /// Current offset
    offset: u64,
    /// Min timestamp
    min_timestamp: i64,
    /// Max timestamp
    max_timestamp: i64,
    /// Point count
    point_count: usize,
}

impl SSTableBuilder {
    /// Create a new SSTable builder
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        Ok(Self {
            path,
            writer: BufWriter::new(file),
            index: Vec::new(),
            offset: 0,
            min_timestamp: i64::MAX,
            max_timestamp: i64::MIN,
            point_count: 0,
        })
    }

    /// Add a series to the SSTable
    pub fn add_series(&mut self, series: &Series) -> Result<()> {
        if series.is_empty() {
            return Ok(());
        }

        let start_offset = self.offset;

        // Write series key
        self.writer.write_u64::<LittleEndian>(series.key)?;
        self.offset += 8;

        // Write metric name
        let name_bytes = series.metric.name.as_bytes();
        let name_len_bytes = encode_varint(name_bytes.len() as u64);
        self.writer.write_all(&name_len_bytes)?;
        self.writer.write_all(name_bytes)?;
        self.offset += name_len_bytes.len() as u64 + name_bytes.len() as u64;

        // Write tags
        let tags_len_bytes = encode_varint(series.metric.tags.len() as u64);
        self.writer.write_all(&tags_len_bytes)?;
        self.offset += tags_len_bytes.len() as u64;

        for (key, value) in &series.metric.tags {
            let key_bytes = key.as_bytes();
            let key_len_bytes = encode_varint(key_bytes.len() as u64);
            self.writer.write_all(&key_len_bytes)?;
            self.writer.write_all(key_bytes)?;
            self.offset += key_len_bytes.len() as u64 + key_bytes.len() as u64;

            let value_bytes = value.as_bytes();
            let value_len_bytes = encode_varint(value_bytes.len() as u64);
            self.writer.write_all(&value_len_bytes)?;
            self.writer.write_all(value_bytes)?;
            self.offset += value_len_bytes.len() as u64 + value_bytes.len() as u64;
        }

        // Compress and write data
        let compressed = compress_points(&series.points)?;
        let data_len_bytes = encode_varint(compressed.len() as u64);
        self.writer.write_all(&data_len_bytes)?;
        self.writer.write_all(&compressed)?;
        self.offset += data_len_bytes.len() as u64 + compressed.len() as u64;

        // Update statistics
        let series_min = series.first_timestamp().unwrap();
        let series_max = series.last_timestamp().unwrap();
        self.min_timestamp = self.min_timestamp.min(series_min);
        self.max_timestamp = self.max_timestamp.max(series_max);
        self.point_count += series.len();

        // Add index entry
        self.index.push(IndexEntry {
            series_key: series.key,
            offset: start_offset,
            length: (self.offset - start_offset) as u32,
            min_timestamp: series_min,
            max_timestamp: series_max,
        });

        Ok(())
    }

    /// Finish building the SSTable
    pub fn finish(mut self) -> Result<SSTableMeta> {
        if self.index.is_empty() {
            // Remove empty file
            drop(self.writer);
            let _ = std::fs::remove_file(&self.path);
            return Err(TsdbError::storage("Cannot create empty SSTable"));
        }

        let index_offset = self.offset;

        // Write index block
        let index_count_bytes = encode_varint(self.index.len() as u64);
        self.writer.write_all(&index_count_bytes)?;

        for entry in &self.index {
            self.writer.write_u64::<LittleEndian>(entry.series_key)?;
            self.writer.write_u64::<LittleEndian>(entry.offset)?;
            self.writer.write_u32::<LittleEndian>(entry.length)?;
            self.writer.write_i64::<LittleEndian>(entry.min_timestamp)?;
            self.writer.write_i64::<LittleEndian>(entry.max_timestamp)?;
        }

        self.writer.flush()?;
        let index_length = self.writer.stream_position()? - index_offset;

        // Calculate checksum of the file so far
        self.writer.flush()?;
        let mut hasher = Hasher::new();
        let file_content = std::fs::read(&self.path)?;
        hasher.update(&file_content);
        let checksum = hasher.finalize();

        // Write footer
        self.writer.write_u64::<LittleEndian>(index_offset)?;
        self.writer.write_u64::<LittleEndian>(index_length)?;
        self.writer.write_i64::<LittleEndian>(self.min_timestamp)?;
        self.writer.write_i64::<LittleEndian>(self.max_timestamp)?;
        self.writer.write_u32::<LittleEndian>(checksum)?;
        self.writer.write_u32::<LittleEndian>(SSTABLE_VERSION)?;
        self.writer.write_u32::<LittleEndian>(SSTABLE_MAGIC)?;
        self.writer.flush()?;

        let file_size = self.writer.stream_position()?;

        Ok(SSTableMeta {
            path: self.path,
            min_timestamp: self.min_timestamp,
            max_timestamp: self.max_timestamp,
            series_count: self.index.len(),
            point_count: self.point_count,
            file_size,
        })
    }
}

/// SSTable reader for querying SSTables
#[derive(Debug)]
pub struct SSTableReader {
    /// Path to the SSTable file
    path: PathBuf,
    /// Index entries (loaded lazily)
    index: Vec<IndexEntry>,
    /// Metadata
    pub meta: SSTableMeta,
}

impl SSTableReader {
    /// Open an SSTable for reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = BufReader::new(File::open(&path)?);

        // Read footer
        let file_size = file.seek(SeekFrom::End(0))?;
        if file_size < FOOTER_SIZE as u64 {
            return Err(TsdbError::corruption("SSTable file too small"));
        }

        file.seek(SeekFrom::End(-(FOOTER_SIZE as i64)))?;

        let index_offset = file.read_u64::<LittleEndian>()?;
        let index_length = file.read_u64::<LittleEndian>()?;
        let min_timestamp = file.read_i64::<LittleEndian>()?;
        let max_timestamp = file.read_i64::<LittleEndian>()?;
        let _checksum = file.read_u32::<LittleEndian>()?;
        let version = file.read_u32::<LittleEndian>()?;
        let magic = file.read_u32::<LittleEndian>()?;

        if magic != SSTABLE_MAGIC {
            return Err(TsdbError::corruption("Invalid SSTable magic number"));
        }

        if version != SSTABLE_VERSION {
            return Err(TsdbError::corruption(format!(
                "Unsupported SSTable version: {}",
                version
            )));
        }

        // Read index
        file.seek(SeekFrom::Start(index_offset))?;
        let mut index_data = vec![0u8; index_length as usize];
        file.read_exact(&mut index_data)?;

        let (count, mut offset) = decode_varint(&index_data)?;
        let mut index = Vec::with_capacity(count as usize);

        for _ in 0..count {
            let series_key = u64::from_le_bytes(index_data[offset..offset + 8].try_into().unwrap());
            offset += 8;
            let entry_offset = u64::from_le_bytes(index_data[offset..offset + 8].try_into().unwrap());
            offset += 8;
            let length = u32::from_le_bytes(index_data[offset..offset + 4].try_into().unwrap());
            offset += 4;
            let entry_min_ts = i64::from_le_bytes(index_data[offset..offset + 8].try_into().unwrap());
            offset += 8;
            let entry_max_ts = i64::from_le_bytes(index_data[offset..offset + 8].try_into().unwrap());
            offset += 8;

            index.push(IndexEntry {
                series_key,
                offset: entry_offset,
                length,
                min_timestamp: entry_min_ts,
                max_timestamp: entry_max_ts,
            });
        }

        let meta = SSTableMeta {
            path: path.clone(),
            min_timestamp,
            max_timestamp,
            series_count: index.len(),
            point_count: 0, // Not stored in footer
            file_size,
        };

        Ok(Self { path, index, meta })
    }

    /// Read a specific series from the SSTable
    pub fn read_series(&self, series_key: SeriesKey) -> Result<Option<Series>> {
        // Find the index entry
        let entry = match self.index.iter().find(|e| e.series_key == series_key) {
            Some(e) => e,
            None => return Ok(None),
        };

        let mut file = BufReader::new(File::open(&self.path)?);
        file.seek(SeekFrom::Start(entry.offset))?;

        let mut data = vec![0u8; entry.length as usize];
        file.read_exact(&mut data)?;

        let mut offset = 0;

        // Read series key
        let _key = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += 8;

        // Read metric name
        let (name_len, bytes_read) = decode_varint(&data[offset..])?;
        offset += bytes_read;
        let name = String::from_utf8(data[offset..offset + name_len as usize].to_vec())
            .map_err(|e| TsdbError::corruption(format!("Invalid metric name: {}", e)))?;
        offset += name_len as usize;

        // Read tags
        let (tags_count, bytes_read) = decode_varint(&data[offset..])?;
        offset += bytes_read;

        let mut tags = std::collections::BTreeMap::new();
        for _ in 0..tags_count {
            let (key_len, bytes_read) = decode_varint(&data[offset..])?;
            offset += bytes_read;
            let key = String::from_utf8(data[offset..offset + key_len as usize].to_vec())
                .map_err(|e| TsdbError::corruption(format!("Invalid tag key: {}", e)))?;
            offset += key_len as usize;

            let (value_len, bytes_read) = decode_varint(&data[offset..])?;
            offset += bytes_read;
            let value = String::from_utf8(data[offset..offset + value_len as usize].to_vec())
                .map_err(|e| TsdbError::corruption(format!("Invalid tag value: {}", e)))?;
            offset += value_len as usize;

            tags.insert(key, value);
        }

        // Read compressed data
        let (data_len, bytes_read) = decode_varint(&data[offset..])?;
        offset += bytes_read;

        let points = decompress_points(&data[offset..offset + data_len as usize])?;

        let metric = Metric::with_tags(name, tags);
        let mut series = Series::new(metric);
        series.points = points;

        Ok(Some(series))
    }

    /// Query points in a time range for a specific series
    pub fn query_range(
        &self,
        series_key: SeriesKey,
        start: i64,
        end: i64,
    ) -> Result<Vec<DataPoint>> {
        // Check if the series might overlap with the time range
        let entry = match self.index.iter().find(|e| e.series_key == series_key) {
            Some(e) => e,
            None => return Ok(Vec::new()),
        };

        // Quick check using index metadata
        if entry.max_timestamp < start || entry.min_timestamp > end {
            return Ok(Vec::new());
        }

        // Read and filter
        if let Some(series) = self.read_series(series_key)? {
            Ok(series.range(start, end).to_vec())
        } else {
            Ok(Vec::new())
        }
    }

    /// Get all series keys in the SSTable
    pub fn series_keys(&self) -> Vec<SeriesKey> {
        self.index.iter().map(|e| e.series_key).collect()
    }

    /// Check if the SSTable contains a specific series
    pub fn contains_series(&self, series_key: SeriesKey) -> bool {
        self.index.iter().any(|e| e.series_key == series_key)
    }

    /// Check if the SSTable overlaps with a time range
    pub fn overlaps(&self, start: i64, end: i64) -> bool {
        self.meta.min_timestamp <= end && self.meta.max_timestamp >= start
    }

    /// Get series that might overlap with a time range
    pub fn series_in_range(&self, start: i64, end: i64) -> Vec<SeriesKey> {
        self.index
            .iter()
            .filter(|e| e.min_timestamp <= end && e.max_timestamp >= start)
            .map(|e| e.series_key)
            .collect()
    }
}

/// SSTable represents an immutable sorted string table
#[derive(Debug)]
pub struct SSTable {
    reader: SSTableReader,
}

impl SSTable {
    /// Open an existing SSTable
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(Self {
            reader: SSTableReader::open(path)?,
        })
    }

    /// Create a new SSTable from series data
    pub fn create<P: AsRef<Path>>(path: P, series: &[Series]) -> Result<SSTableMeta> {
        let mut builder = SSTableBuilder::new(path)?;

        for s in series {
            builder.add_series(s)?;
        }

        builder.finish()
    }

    /// Create from a BTreeMap of series
    pub fn create_from_map<P: AsRef<Path>>(
        path: P,
        data: &BTreeMap<SeriesKey, Series>,
    ) -> Result<SSTableMeta> {
        let mut builder = SSTableBuilder::new(path)?;

        for series in data.values() {
            builder.add_series(series)?;
        }

        builder.finish()
    }

    /// Read a series
    pub fn read_series(&self, series_key: SeriesKey) -> Result<Option<Series>> {
        self.reader.read_series(series_key)
    }

    /// Query range
    pub fn query_range(&self, series_key: SeriesKey, start: i64, end: i64) -> Result<Vec<DataPoint>> {
        self.reader.query_range(series_key, start, end)
    }

    /// Get metadata
    pub fn meta(&self) -> &SSTableMeta {
        &self.reader.meta
    }

    /// Get all series keys
    pub fn series_keys(&self) -> Vec<SeriesKey> {
        self.reader.series_keys()
    }

    /// Check if contains a series
    pub fn contains_series(&self, series_key: SeriesKey) -> bool {
        self.reader.contains_series(series_key)
    }

    /// Check time range overlap
    pub fn overlaps(&self, start: i64, end: i64) -> bool {
        self.reader.overlaps(start, end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_series(name: &str, point_count: usize) -> Series {
        let metric = Metric::new(name).tag("host", "server1").tag("region", "us-east");
        let mut series = Series::new(metric);
        for i in 0..point_count {
            series.push(DataPoint::new(1000 + i as i64 * 60, i as f64 * 1.5));
        }
        series
    }

    #[test]
    fn test_sstable_create_and_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");

        let series1 = create_test_series("cpu.usage", 100);
        let series2 = create_test_series("memory.used", 50);

        let meta = SSTable::create(&path, &[series1.clone(), series2.clone()]).unwrap();

        assert_eq!(meta.series_count, 2);
        assert!(meta.file_size > 0);

        let sstable = SSTable::open(&path).unwrap();

        let read1 = sstable.read_series(series1.key).unwrap().unwrap();
        assert_eq!(read1.len(), 100);
        assert_eq!(read1.metric.name, "cpu.usage");

        let read2 = sstable.read_series(series2.key).unwrap().unwrap();
        assert_eq!(read2.len(), 50);
    }

    #[test]
    fn test_sstable_query_range() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");

        let series = create_test_series("test", 100);
        SSTable::create(&path, &[series.clone()]).unwrap();

        let sstable = SSTable::open(&path).unwrap();
        let points = sstable.query_range(series.key, 2000, 4000).unwrap();

        assert!(!points.is_empty());
        assert!(points.iter().all(|p| p.timestamp >= 2000 && p.timestamp <= 4000));
    }

    #[test]
    fn test_sstable_series_keys() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");

        let series1 = create_test_series("metric1", 10);
        let series2 = create_test_series("metric2", 10);

        SSTable::create(&path, &[series1.clone(), series2.clone()]).unwrap();

        let sstable = SSTable::open(&path).unwrap();
        let keys = sstable.series_keys();

        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&series1.key));
        assert!(keys.contains(&series2.key));
    }

    #[test]
    fn test_sstable_overlaps() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");

        let series = create_test_series("test", 100);
        // Series timestamps: 1000 to 1000 + 99*60 = 6940

        SSTable::create(&path, &[series]).unwrap();

        let sstable = SSTable::open(&path).unwrap();

        assert!(sstable.overlaps(0, 2000));
        assert!(sstable.overlaps(5000, 10000));
        assert!(sstable.overlaps(2000, 5000));
        assert!(!sstable.overlaps(10000, 20000));
        assert!(!sstable.overlaps(0, 500));
    }

    #[test]
    fn test_sstable_from_map() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");

        let series1 = create_test_series("metric1", 10);
        let series2 = create_test_series("metric2", 10);

        let mut map = BTreeMap::new();
        map.insert(series1.key, series1);
        map.insert(series2.key, series2);

        let meta = SSTable::create_from_map(&path, &map).unwrap();
        assert_eq!(meta.series_count, 2);
    }

    #[test]
    fn test_sstable_nonexistent_series() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");

        let series = create_test_series("test", 10);
        SSTable::create(&path, &[series]).unwrap();

        let sstable = SSTable::open(&path).unwrap();
        let result = sstable.read_series(12345).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_sstable_contains_series() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");

        let series = create_test_series("test", 10);
        SSTable::create(&path, &[series.clone()]).unwrap();

        let sstable = SSTable::open(&path).unwrap();
        assert!(sstable.contains_series(series.key));
        assert!(!sstable.contains_series(12345));
    }

    #[test]
    fn test_sstable_empty_series() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");

        let metric = Metric::new("empty");
        let series = Series::new(metric);

        let mut builder = SSTableBuilder::new(&path).unwrap();
        builder.add_series(&series).unwrap(); // Empty series is skipped

        let result = builder.finish();
        assert!(result.is_err()); // Should fail with empty SSTable
    }

    #[test]
    fn test_sstable_metadata() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");

        let series = create_test_series("test", 100);
        let meta = SSTable::create(&path, &[series]).unwrap();

        assert_eq!(meta.series_count, 1);
        assert_eq!(meta.point_count, 100);
        assert_eq!(meta.min_timestamp, 1000);
        assert_eq!(meta.max_timestamp, 1000 + 99 * 60);
    }

    #[test]
    fn test_sstable_series_in_range() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");

        let mut series1 = Series::new(Metric::new("early"));
        for i in 0..10 {
            series1.push(DataPoint::new(1000 + i * 60, i as f64));
        }

        let mut series2 = Series::new(Metric::new("late"));
        for i in 0..10 {
            series2.push(DataPoint::new(10000 + i * 60, i as f64));
        }

        SSTable::create(&path, &[series1.clone(), series2.clone()]).unwrap();
        let sstable = SSTable::open(&path).unwrap();

        let reader = SSTableReader::open(&path).unwrap();
        let early_series = reader.series_in_range(0, 2000);
        let late_series = reader.series_in_range(9000, 20000);
        let all_series = reader.series_in_range(0, 20000);

        assert_eq!(early_series.len(), 1);
        assert_eq!(late_series.len(), 1);
        assert_eq!(all_series.len(), 2);
    }
}
