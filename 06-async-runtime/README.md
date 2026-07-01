# Async Runtime

A Tokio-style asynchronous runtime built from scratch in Rust: an `epoll`-based
I/O reactor, a work-stealing multi-threaded scheduler, a hierarchical timer
wheel, and a full set of async primitives (channels, timers, locks, structured
concurrency). It implements the `Future`/`Waker` machinery directly on top of the
Linux syscall layer with no `tokio`/`mio`/`async-std` dependencies.

## Features

- **Epoll reactor** — edge-triggered I/O readiness via `epoll_create1`/`epoll_ctl`/`epoll_wait`, mapping `Token`s to `Waker`s (`reactor::Reactor`).
- **Work-stealing scheduler** — a global `crossbeam_deque::Injector` plus per-worker FIFO queues with cross-worker stealing (`scheduler::Scheduler`, `runtime::Runtime`).
- **Single-threaded executor** — a `SegQueue`-backed executor driving the `block_on`/`spawn` free functions (`executor::Executor`).
- **Manual waker vtable** — `Arc<Task>` is turned into a `RawWaker` with a hand-written `RawWakerVTable` (`task::Task`).
- **Hierarchical timer wheel** — four cascading wheels (256, 64, 64, 64 slots) at 1 ms tick granularity (`timer::TimerWheel`).
- **Time utilities** — `sleep`, `sleep_until`, `timeout`, `timeout_at`, `interval`, with an `Elapsed` error type (`time`).
- **Future combinators** — `select` (returning `Either`), `join`, `join3`, `yield_now`, `ready`, `pending` (`future`).
- **Channels** — single-use `oneshot` and bounded `mpsc` with backpressure and waker wake-ups (`sync::oneshot`, `sync::mpsc`).
- **Async sync primitives** — `Mutex`, `RwLock`, `Notify`, `Semaphore`, `Barrier` (`sync::mutex`, `sync::notify`).
- **Cancellation** — hierarchical `CancellationToken` with child tokens and `run_until_cancelled` (`sync::cancellation`).
- **Structured concurrency** — `scope`/`scope_detached`/`TaskSet` that join all spawned work before returning (`scope`).
- **Networking** — async `TcpListener`/`TcpStream`, `UdpSocket`, and `UnixListener`/`UnixStream` over the reactor (`net`).

## Architecture

```mermaid
flowchart TD
    APP[Async application code] --> RT[Runtime / block_on]
    RT --> SCHED[Work-stealing scheduler]
    RT --> EXEC[Single-threaded executor]
    SCHED --> TASK[Task (Pin Box dyn Future)]
    EXEC --> TASK
    TASK --> WAKER[Manual RawWaker vtable]
    WAKER --> SCHED
    TASK --> REACTOR[Epoll reactor]
    TASK --> TIMER[Timer wheel]
    REACTOR --> EPOLL[(Linux epoll fd)]
    REACTOR --> WAKER
    TIMER --> WAKER
```

| Component | Module | Responsibility |
|-----------|--------|----------------|
| Reactor | `reactor` | Register fds with `epoll`, poll for readiness, wake the right `Waker` |
| Scheduler | `scheduler` | Global injector + per-worker deques with work stealing |
| Runtime | `runtime` | Multi-threaded runtime, `Builder`, worker threads, `JoinHandle` |
| Executor | `executor` | Single-threaded `block_on`/`spawn` driver |
| Task | `task` | Future state, `Arc<Task>` to `RawWaker` conversion |
| Timer wheel | `timer` | O(1) insert/cancel, cascading hierarchical wheels |
| Time | `time` | `sleep`/`timeout`/`interval` futures over the timer wheel |
| Combinators | `future` | `select`/`join`/`yield_now`/`ready`/`pending` |
| Channels | `sync::oneshot`, `sync::mpsc` | Task-to-task value passing |
| Sync | `sync::mutex`, `sync::notify`, `sync::cancellation` | Async locks, notifications, cancellation |
| Scope | `scope` | Structured concurrency over spawned tasks |
| Net | `net::tcp`, `net::udp`, `net::unix` | Async sockets driven by the reactor |

## Quick Start

### Prerequisites

- Rust 1.70+ (`edition = "2021"`) and Cargo.
- **Linux only.** The reactor calls the `epoll` syscall family unconditionally, so
  the crate has a `compile_error!` guard and will not build on macOS/Windows. Use a
  Linux host, VM, or container.

### Installation

```bash
cd 06-async-runtime
cargo build
```

### Running

```bash
cargo run --example echo_server    # TCP echo server on 127.0.0.1:8080
cargo run --example timeout_demo   # timeout combinator demo
cargo run --example channels       # oneshot + mpsc channel demo
```

## Usage

Spawn background work on the single-threaded executor with the free functions:

```rust
use async_runtime::{block_on, spawn, sleep};
use std::time::Duration;

fn main() {
    block_on(async {
        spawn(async {
            sleep(Duration::from_millis(10)).await;
            println!("background task done");
        });
        println!("hello from the runtime");
    });
}
```

Use the multi-threaded `Runtime` when you want worker threads and `JoinHandle`s.
`Runtime::spawn` returns a `JoinHandle<T>` that resolves to `Result<T, JoinError>`:

```rust
use async_runtime::Runtime;

fn main() -> std::io::Result<()> {
    let rt = Runtime::new()?;

    let handle = rt.spawn(async { 21 * 2 });
    let result = rt.block_on(handle).unwrap();
    println!("{result}");

    rt.shutdown();
    Ok(())
}
```

Channels, timeout, and select against the real public API:

```rust
use async_runtime::{block_on, spawn, timeout, select, sleep, Either};
use async_runtime::sync::{oneshot, mpsc};
use std::time::Duration;

fn main() {
    block_on(async {
        // oneshot: single value across tasks
        let (tx, rx) = oneshot::channel();
        spawn(async move { tx.send(42).unwrap(); });
        let _value = rx.await.unwrap();

        // bounded mpsc with try_send / try_recv
        let (tx, mut rx) = mpsc::channel(16);
        tx.try_send(1).unwrap();
        while let Ok(v) = rx.try_recv() {
            println!("received {v}");
        }

        // deadline enforcement
        let _ = timeout(Duration::from_millis(100), async { "done" }).await;

        // race two futures
        match select(sleep(Duration::from_millis(10)),
                     sleep(Duration::from_millis(100))).await {
            Either::Left(_)  => println!("fast won"),
            Either::Right(_) => println!("slow won"),
        }
    });
}
```

## What's Real vs Simulated

- **Real:** The `epoll` reactor (`epoll_create1`/`epoll_ctl`/`epoll_wait`, edge-triggered) is genuine and exercised by reactor unit tests against `UnixStream` pairs. The hierarchical timer wheel, the manual `RawWaker` vtable, the `oneshot`/`mpsc` channels, the future combinators, the work-stealing scheduler primitives (`Injector` + per-worker deques with stealing), and the async sync primitives are all fully implemented with passing unit tests. TCP/UDP/Unix sockets set `O_NONBLOCK` and register real fds with the reactor.
- **Linux only:** Builds and runs on Linux exclusively. There is no `kqueue`/IOCP backend; the crate emits a `compile_error!` on non-Linux targets.
- **Simulated / aspirational:** I/O futures register their `Waker` through the **single-threaded** `EXECUTOR` thread-local, so socket readiness wiring is wired for `block_on`/`spawn`, not for the multi-threaded `Runtime` worker loop. `JoinHandle::poll` in `task.rs` busy-wakes rather than registering a completion waker. The integration suites `tests/executor_tests.rs` and `tests/io_tests.rs` describe a larger target API (`ExecutorConfig`, `async_runtime::fs`, async `bind`/`accept`, `stream.split`, task-locals) that is **not** implemented and does not compile; the trustworthy tests are the in-module `#[cfg(test)]` units.

## Testing

```bash
cargo test --lib          # in-module unit tests (reactor, timer, scheduler, channels, ...)
cargo bench               # Criterion benchmarks
```

The in-module unit tests cover the reactor (register/poll/deregister), the timer
wheel (insert/cancel/cascade), the scheduler (push/pop/steal), the channels, the
combinators, and the TCP listener. They need no external services. The standalone
`tests/` integration files target an unimplemented API and are not part of the
passing suite (see What's Real vs Simulated).

## Project Structure

```
06-async-runtime/
  README.md                 # this file
  Cargo.toml                # crate + criterion bench config
  src/
    lib.rs                  # public exports, Interest/Token/Event/Events
    reactor.rs              # epoll-based I/O reactor
    task.rs                 # Task + manual RawWaker vtable
    executor.rs             # single-threaded block_on/spawn executor
    scheduler.rs            # work-stealing scheduler
    runtime.rs              # multi-threaded Runtime + Builder
    timer.rs                # hierarchical timer wheel
    time.rs                 # sleep/timeout/interval
    future.rs               # select/join/yield_now/ready/pending
    scope.rs                # structured concurrency (scope/TaskSet)
    io.rs, io_util.rs       # async I/O traits + buffered helpers
    sync/                   # oneshot, mpsc, mutex, notify, cancellation
    net/                    # tcp, udp, unix sockets
  examples/                 # echo_server, timeout_demo, channels
  benches/benchmarks.rs     # Criterion benchmarks
  docs/BLUEPRINT.md         # full architecture and design
```

## License

MIT — see ../LICENSE
