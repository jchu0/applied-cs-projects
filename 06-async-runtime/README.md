# Async Runtime

A custom async runtime implementation in Rust, similar to Tokio. This project demonstrates deep understanding of async/await internals, I/O multiplexing, and concurrent systems.

## Features

### Core Runtime
- **Epoll-based reactor** - Event-driven I/O on Linux
- **Work-stealing scheduler** - Efficient multi-threaded task execution
- **Hierarchical timer wheel** - O(1) timer operations
- **Cooperative multitasking** - Non-blocking async execution

### Synchronization Primitives
- **Oneshot channel** - Single-value communication between tasks
- **MPSC channel** - Bounded multi-producer, single-consumer queue
- **JoinHandle** - Task completion awaiting

### Networking
- **TCP** - Async TcpListener and TcpStream
- **UDP** - Async UdpSocket with send/recv operations
- **Unix sockets** - UnixListener and UnixStream

### Time Utilities
- **sleep/sleep_until** - Async delay functions
- **timeout/timeout_at** - Deadline enforcement
- **interval** - Periodic timer

### Future Combinators
- **select** - Race multiple futures
- **join/join3** - Wait for multiple futures
- **yield_now** - Cooperative yielding
- **ready/pending** - Utility futures

## Usage

### Basic Example

```rust
use async_runtime::{block_on, spawn, sleep};
use std::time::Duration;

fn main() {
    block_on(async {
        // Spawn a background task
        spawn(async {
            sleep(Duration::from_millis(100)).await;
            println!("Background task complete!");
        });

        // Main task
        println!("Hello from async runtime!");
    });
}
```

### TCP Echo Server

```rust
use async_runtime::net::TcpListener;
use async_runtime::{spawn, Runtime};

fn main() -> std::io::Result<()> {
    let rt = Runtime::new()?;

    rt.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:8080".parse().unwrap())?;

        loop {
            let (mut stream, _) = listener.accept().await?;

            spawn(async move {
                let mut buf = [0u8; 1024];
                loop {
                    let n = stream.read(&mut buf).await?;
                    if n == 0 { break; }
                    stream.write_all(&buf[..n]).await?;
                }
                Ok::<_, std::io::Error>(())
            });
        }
    })
}
```

### Using Channels

```rust
use async_runtime::sync::{oneshot, mpsc};
use async_runtime::{block_on, spawn};

fn main() {
    block_on(async {
        // Oneshot channel
        let (tx, rx) = oneshot::channel();
        spawn(async move { tx.send(42).unwrap() });
        let value = rx.await.unwrap();

        // MPSC channel
        let (tx, mut rx) = mpsc::channel(16);
        tx.try_send(1).unwrap();
        tx.try_send(2).unwrap();

        while let Ok(v) = rx.try_recv() {
            println!("Received: {}", v);
        }
    });
}
```

### Timeout and Select

```rust
use async_runtime::{block_on, timeout, select, sleep, Either};
use std::time::Duration;

fn main() {
    block_on(async {
        // Timeout
        let result = timeout(
            Duration::from_millis(100),
            async { "completed" }
        ).await;

        // Select between futures
        let fast = sleep(Duration::from_millis(10));
        let slow = sleep(Duration::from_millis(100));

        match select(fast, slow).await {
            Either::Left(_) => println!("Fast won!"),
            Either::Right(_) => println!("Slow won!"),
        }
    });
}
```

## Architecture

### Reactor
The reactor uses Linux's epoll for I/O event notification. It maintains a mapping of file descriptors to wakers and efficiently polls for ready events.

### Scheduler
The work-stealing scheduler distributes tasks across worker threads. Each worker has a local FIFO queue and can steal from others when idle.

### Timer Wheel
A hierarchical timer wheel provides O(1) insertion and cancellation. Timers are organized into 4 levels of granularity for efficient processing.

### Task System
Tasks are represented as pinned futures with associated wakers. The runtime manages task state transitions and polling.

## Running Examples

```bash
# Echo server
cargo run --example echo_server

# Timeout demo
cargo run --example timeout_demo

# Channel demo
cargo run --example channels
```

## Running Benchmarks

```bash
cargo bench
```

## Running Tests

```bash
cargo test
```

## Project Structure

```
src/
├── lib.rs          # Main library exports
├── reactor.rs      # Epoll-based event reactor
├── task.rs         # Task and waker implementation
├── executor.rs     # Single-threaded executor
├── scheduler.rs    # Work-stealing scheduler
├── runtime.rs      # Multi-threaded runtime
├── timer.rs        # Hierarchical timer wheel
├── time.rs         # Time utilities (sleep, timeout)
├── future.rs       # Future combinators
├── io.rs           # Legacy I/O (being replaced)
├── io_util.rs      # Buffered I/O utilities
├── sync/           # Synchronization primitives
│   ├── mod.rs
│   ├── oneshot.rs
│   └── mpsc.rs
└── net/            # Networking
    ├── mod.rs
    ├── tcp.rs
    ├── udp.rs
    └── unix.rs
```

## Design Decisions

1. **Epoll over io_uring** - Chose epoll for broader compatibility and simpler implementation
2. **Work-stealing** - Provides good load balancing with minimal overhead
3. **Timer wheel** - O(1) operations are critical for high-throughput systems
4. **Thread-local executor** - Simplifies implementation while maintaining efficiency

## Future Improvements

- Add macOS kqueue support
- Implement async file I/O
- Add tracing/debugging support
- Optimize hot paths
- Add more comprehensive error handling
