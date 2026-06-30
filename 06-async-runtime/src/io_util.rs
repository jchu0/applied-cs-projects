//! I/O utilities
//!
//! Provides buffered I/O and other utilities.

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Async reader trait
pub trait AsyncRead {
    /// Read data into the buffer
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>>;
}

/// Async writer trait
pub trait AsyncWrite {
    /// Write data from the buffer
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>>;

    /// Flush any buffered data
    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>>;

    /// Shutdown the writer
    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>>;
}

/// Buffered reader
pub struct BufReader<R> {
    inner: R,
    buf: Vec<u8>,
    pos: usize,
    cap: usize,
}

impl<R> BufReader<R> {
    /// Create a new buffered reader with default capacity
    pub fn new(inner: R) -> Self {
        Self::with_capacity(8192, inner)
    }

    /// Create a new buffered reader with specified capacity
    pub fn with_capacity(capacity: usize, inner: R) -> Self {
        Self {
            inner,
            buf: vec![0; capacity],
            pos: 0,
            cap: 0,
        }
    }

    /// Get a reference to the inner reader
    pub fn get_ref(&self) -> &R {
        &self.inner
    }

    /// Get a mutable reference to the inner reader
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    /// Unwrap the inner reader
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Get the number of bytes available in the buffer
    pub fn buffer(&self) -> &[u8] {
        &self.buf[self.pos..self.cap]
    }
}

/// Buffered writer
pub struct BufWriter<W> {
    inner: W,
    buf: Vec<u8>,
}

impl<W> BufWriter<W> {
    /// Create a new buffered writer with default capacity
    pub fn new(inner: W) -> Self {
        Self::with_capacity(8192, inner)
    }

    /// Create a new buffered writer with specified capacity
    pub fn with_capacity(capacity: usize, inner: W) -> Self {
        Self {
            inner,
            buf: Vec::with_capacity(capacity),
        }
    }

    /// Get a reference to the inner writer
    pub fn get_ref(&self) -> &W {
        &self.inner
    }

    /// Get a mutable reference to the inner writer
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.inner
    }

    /// Unwrap the inner writer
    pub fn into_inner(self) -> W {
        self.inner
    }

    /// Get the buffered data
    pub fn buffer(&self) -> &[u8] {
        &self.buf
    }
}

/// Split a stream into read and write halves
pub fn split<T>(stream: T) -> (ReadHalf<T>, WriteHalf<T>)
where
    T: AsyncRead + AsyncWrite,
{
    let stream = std::sync::Arc::new(parking_lot::Mutex::new(stream));
    (
        ReadHalf {
            inner: stream.clone(),
        },
        WriteHalf { inner: stream },
    )
}

/// Read half of a split stream
pub struct ReadHalf<T> {
    inner: std::sync::Arc<parking_lot::Mutex<T>>,
}

/// Write half of a split stream
pub struct WriteHalf<T> {
    inner: std::sync::Arc<parking_lot::Mutex<T>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buf_reader_creation() {
        let data = std::io::Cursor::new(vec![1, 2, 3]);
        let reader = BufReader::new(data);
        assert_eq!(reader.buffer().len(), 0);
    }

    #[test]
    fn test_buf_writer_creation() {
        let data: Vec<u8> = Vec::new();
        let writer = BufWriter::new(data);
        assert_eq!(writer.buffer().len(), 0);
    }
}
