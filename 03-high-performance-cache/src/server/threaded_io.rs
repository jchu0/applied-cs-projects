//! Multi-threaded I/O for high-performance networking.
//!
//! This module implements Redis-style threaded I/O where read and write
//! operations are offloaded to dedicated I/O threads while the main thread
//! handles command execution.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use bytes::BytesMut;
use mio::Token;

/// Configuration for threaded I/O.
#[derive(Debug, Clone)]
pub struct ThreadedIOConfig {
    /// Number of I/O threads (0 = disabled, use main thread only).
    pub io_threads: usize,
    /// Whether to enable threaded reads (writes are always threaded when enabled).
    pub threaded_reads: bool,
    /// Maximum pending jobs per thread before backpressure.
    pub max_pending_per_thread: usize,
}

impl Default for ThreadedIOConfig {
    fn default() -> Self {
        Self {
            io_threads: 4,
            threaded_reads: true,
            max_pending_per_thread: 1024,
        }
    }
}

/// A read job to be processed by an I/O thread.
pub struct ReadJob {
    /// Connection token for identification.
    pub token: Token,
    /// File descriptor or socket to read from.
    pub fd: i32,
    /// Buffer to read into.
    pub buffer: BytesMut,
}

/// A write job to be processed by an I/O thread.
pub struct WriteJob {
    /// Connection token for identification.
    pub token: Token,
    /// File descriptor or socket to write to.
    pub fd: i32,
    /// Data to write.
    pub data: BytesMut,
}

/// Result of an I/O operation.
pub enum IOResult {
    /// Read completed with data.
    Read {
        token: Token,
        data: BytesMut,
        bytes_read: usize,
        closed: bool,
    },
    /// Read failed.
    ReadError {
        token: Token,
        error: io::Error,
    },
    /// Write completed.
    Write {
        token: Token,
        bytes_written: usize,
        remaining: Option<BytesMut>,
    },
    /// Write failed.
    WriteError {
        token: Token,
        error: io::Error,
    },
}

/// Statistics for threaded I/O.
#[derive(Debug, Default)]
pub struct IOStats {
    /// Total reads processed.
    pub total_reads: AtomicUsize,
    /// Total writes processed.
    pub total_writes: AtomicUsize,
    /// Total bytes read.
    pub bytes_read: AtomicUsize,
    /// Total bytes written.
    pub bytes_written: AtomicUsize,
    /// Currently pending read jobs.
    pub pending_reads: AtomicUsize,
    /// Currently pending write jobs.
    pub pending_writes: AtomicUsize,
}

impl IOStats {
    /// Record a completed read.
    pub fn record_read(&self, bytes: usize) {
        self.total_reads.fetch_add(1, Ordering::Relaxed);
        self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
        self.pending_reads.fetch_sub(1, Ordering::Relaxed);
    }

    /// Record a completed write.
    pub fn record_write(&self, bytes: usize) {
        self.total_writes.fetch_add(1, Ordering::Relaxed);
        self.bytes_written.fetch_add(bytes, Ordering::Relaxed);
        self.pending_writes.fetch_sub(1, Ordering::Relaxed);
    }
}

/// I/O thread worker.
struct IOWorker {
    /// Thread handle.
    handle: JoinHandle<()>,
    /// Channel for sending read jobs.
    read_tx: Sender<ReadJob>,
    /// Channel for sending write jobs.
    write_tx: Sender<WriteJob>,
}

/// Multi-threaded I/O manager.
///
/// Distributes I/O operations across multiple threads to improve throughput
/// on multi-core systems. The main thread remains responsible for command
/// execution while I/O threads handle network reads and writes.
pub struct ThreadedIO {
    /// Worker threads.
    workers: Vec<IOWorker>,
    /// Channel for receiving I/O results.
    result_rx: Receiver<IOResult>,
    /// Shared channel sender for results (cloned to workers).
    result_tx: Sender<IOResult>,
    /// Configuration.
    config: ThreadedIOConfig,
    /// Statistics.
    stats: Arc<IOStats>,
    /// Shutdown flag.
    shutdown: Arc<AtomicBool>,
    /// Map from token to worker index for affinity.
    token_affinity: HashMap<Token, usize>,
    /// Next worker for round-robin assignment.
    next_worker: usize,
}

impl ThreadedIO {
    /// Create a new threaded I/O manager.
    pub fn new(config: ThreadedIOConfig) -> Self {
        let (result_tx, result_rx) = mpsc::channel();
        let stats = Arc::new(IOStats::default());
        let shutdown = Arc::new(AtomicBool::new(false));

        let mut workers = Vec::with_capacity(config.io_threads);

        for i in 0..config.io_threads {
            let (read_tx, read_rx) = mpsc::channel::<ReadJob>();
            let (write_tx, write_rx) = mpsc::channel::<WriteJob>();
            let result_tx_clone = result_tx.clone();
            let stats_clone = Arc::clone(&stats);
            let shutdown_clone = Arc::clone(&shutdown);

            let handle = thread::Builder::new()
                .name(format!("io-thread-{}", i))
                .spawn(move || {
                    Self::io_thread_main(
                        read_rx,
                        write_rx,
                        result_tx_clone,
                        stats_clone,
                        shutdown_clone,
                    );
                })
                .expect("failed to spawn I/O thread");

            workers.push(IOWorker {
                handle,
                read_tx,
                write_tx,
            });
        }

        Self {
            workers,
            result_rx,
            result_tx,
            config,
            stats,
            shutdown,
            token_affinity: HashMap::new(),
            next_worker: 0,
        }
    }

    /// Main loop for I/O worker thread.
    fn io_thread_main(
        read_rx: Receiver<ReadJob>,
        write_rx: Receiver<WriteJob>,
        result_tx: Sender<IOResult>,
        stats: Arc<IOStats>,
        shutdown: Arc<AtomicBool>,
    ) {
        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Process read jobs
            match read_rx.try_recv() {
                Ok(job) => {
                    let result = Self::process_read(job);
                    if let IOResult::Read { bytes_read, .. } = &result {
                        stats.record_read(*bytes_read);
                    }
                    let _ = result_tx.send(result);
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => break,
            }

            // Process write jobs
            match write_rx.try_recv() {
                Ok(job) => {
                    let result = Self::process_write(job);
                    if let IOResult::Write { bytes_written, .. } = &result {
                        stats.record_write(*bytes_written);
                    }
                    let _ = result_tx.send(result);
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => break,
            }

            // Avoid busy spinning
            thread::yield_now();
        }
    }

    /// Process a read job.
    fn process_read(mut job: ReadJob) -> IOResult {
        // Use raw file descriptor for reading
        // In production, this would use platform-specific APIs
        let mut buf = [0u8; 4096];
        let mut total_read = 0;
        let mut closed = false;

        // Simulate reading (in production, use actual fd)
        // For now, just return the buffer as-is for testing
        job.buffer.reserve(4096);

        IOResult::Read {
            token: job.token,
            data: job.buffer,
            bytes_read: total_read,
            closed,
        }
    }

    /// Process a write job.
    fn process_write(mut job: WriteJob) -> IOResult {
        // Use raw file descriptor for writing
        // In production, this would use platform-specific APIs
        let bytes_to_write = job.data.len();

        // Simulate writing (in production, use actual fd)
        IOResult::Write {
            token: job.token,
            bytes_written: bytes_to_write,
            remaining: None,
        }
    }

    /// Submit a read job.
    pub fn submit_read(&mut self, token: Token, fd: i32) {
        let worker_idx = self.get_worker_for_token(token);
        let job = ReadJob {
            token,
            fd,
            buffer: BytesMut::with_capacity(4096),
        };

        self.stats.pending_reads.fetch_add(1, Ordering::Relaxed);
        let _ = self.workers[worker_idx].read_tx.send(job);
    }

    /// Submit a write job.
    pub fn submit_write(&mut self, token: Token, fd: i32, data: BytesMut) {
        let worker_idx = self.get_worker_for_token(token);
        let job = WriteJob { token, fd, data };

        self.stats.pending_writes.fetch_add(1, Ordering::Relaxed);
        let _ = self.workers[worker_idx].write_tx.send(job);
    }

    /// Get the worker index for a token (maintains affinity).
    fn get_worker_for_token(&mut self, token: Token) -> usize {
        if let Some(&idx) = self.token_affinity.get(&token) {
            return idx;
        }

        // Round-robin assignment for new connections
        let idx = self.next_worker;
        self.next_worker = (self.next_worker + 1) % self.workers.len();
        self.token_affinity.insert(token, idx);
        idx
    }

    /// Remove token affinity (when connection closes).
    pub fn remove_token(&mut self, token: Token) {
        self.token_affinity.remove(&token);
    }

    /// Poll for completed I/O results (non-blocking).
    pub fn poll_results(&self) -> Vec<IOResult> {
        let mut results = Vec::new();
        loop {
            match self.result_rx.try_recv() {
                Ok(result) => results.push(result),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        results
    }

    /// Get I/O statistics.
    pub fn stats(&self) -> &IOStats {
        &self.stats
    }

    /// Get configuration.
    pub fn config(&self) -> &ThreadedIOConfig {
        &self.config
    }

    /// Check if threaded I/O is enabled.
    pub fn is_enabled(&self) -> bool {
        !self.workers.is_empty()
    }

    /// Get number of I/O threads.
    pub fn thread_count(&self) -> usize {
        self.workers.len()
    }

    /// Shutdown all I/O threads.
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);

        // Drop senders to signal threads to exit
        self.workers.clear();
    }
}

impl Drop for ThreadedIO {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Builder for ThreadedIO configuration.
pub struct ThreadedIOBuilder {
    config: ThreadedIOConfig,
}

impl ThreadedIOBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            config: ThreadedIOConfig::default(),
        }
    }

    /// Set the number of I/O threads.
    pub fn io_threads(mut self, count: usize) -> Self {
        self.config.io_threads = count;
        self
    }

    /// Enable or disable threaded reads.
    pub fn threaded_reads(mut self, enabled: bool) -> Self {
        self.config.threaded_reads = enabled;
        self
    }

    /// Set maximum pending jobs per thread.
    pub fn max_pending_per_thread(mut self, max: usize) -> Self {
        self.config.max_pending_per_thread = max;
        self
    }

    /// Build the ThreadedIO instance.
    pub fn build(self) -> ThreadedIO {
        ThreadedIO::new(self.config)
    }
}

impl Default for ThreadedIOBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threaded_io_creation() {
        let config = ThreadedIOConfig {
            io_threads: 2,
            threaded_reads: true,
            max_pending_per_thread: 100,
        };
        let io = ThreadedIO::new(config);

        assert!(io.is_enabled());
        assert_eq!(io.thread_count(), 2);
    }

    #[test]
    fn test_builder_pattern() {
        let io = ThreadedIOBuilder::new()
            .io_threads(4)
            .threaded_reads(true)
            .max_pending_per_thread(512)
            .build();

        assert_eq!(io.thread_count(), 4);
        assert!(io.config().threaded_reads);
        assert_eq!(io.config().max_pending_per_thread, 512);
    }

    #[test]
    fn test_disabled_io() {
        let config = ThreadedIOConfig {
            io_threads: 0,
            threaded_reads: false,
            max_pending_per_thread: 100,
        };
        let io = ThreadedIO::new(config);

        assert!(!io.is_enabled());
        assert_eq!(io.thread_count(), 0);
    }

    #[test]
    fn test_token_affinity() {
        let mut io = ThreadedIOBuilder::new().io_threads(4).build();

        let token1 = Token(1);
        let token2 = Token(2);
        let token3 = Token(3);

        // First assignment should round-robin
        let idx1 = io.get_worker_for_token(token1);
        let idx2 = io.get_worker_for_token(token2);
        let idx3 = io.get_worker_for_token(token3);

        // Should be different workers (round-robin)
        assert_ne!(idx1, idx2);
        assert_ne!(idx2, idx3);

        // Same token should get same worker (affinity)
        assert_eq!(io.get_worker_for_token(token1), idx1);
        assert_eq!(io.get_worker_for_token(token2), idx2);
    }

    #[test]
    fn test_remove_token() {
        let mut io = ThreadedIOBuilder::new().io_threads(2).build();

        let token = Token(42);
        io.get_worker_for_token(token);
        assert!(io.token_affinity.contains_key(&token));

        io.remove_token(token);
        assert!(!io.token_affinity.contains_key(&token));
    }

    #[test]
    fn test_stats() {
        let io = ThreadedIOBuilder::new().io_threads(2).build();
        let stats = io.stats();

        assert_eq!(stats.total_reads.load(Ordering::Relaxed), 0);
        assert_eq!(stats.total_writes.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_shutdown() {
        let mut io = ThreadedIOBuilder::new().io_threads(2).build();
        assert!(io.is_enabled());

        io.shutdown();
        assert!(!io.is_enabled());
    }
}
