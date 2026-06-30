# Project 5: Async Runtime (Tokio/libuv clone)

## Staff-Level Design Document

**Complexity:** ⭐⭐⭐⭐⭐ (Expert+)
**Timeline:** 12-14 weeks
**Languages:** Rust (primary) or C++ (alternative)

> **Concepts covered:** [§01 Rust async](../../01-software-engineering/rust/05-async-rust/rust-async.md) (futures, executors, wakers — this project *implements* the runtime that the tutorial uses). See also [Project 51](../51-message-queue/) and [Project 52](../52-time-series-database/), both built on Tokio. Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

---

## What This Project Teaches

### Core Concepts
- **Event loops and reactor pattern** - Non-blocking I/O multiplexing with epoll/kqueue
- **Futures and promises** - Lazy evaluation, state machines, zero-cost abstractions
- **Cooperative scheduling** - Task yielding, fairness, starvation prevention
- **Thread pool executors** - Work distribution, load balancing, thread safety
- **Timer management** - Timer wheels, hierarchical timing, efficiency at scale
- **Memory safety in async** - Pinning, self-referential structs, cancellation
- **Waker mechanism** - Cross-thread wake-ups, efficient notifications

### Industry Relevance
This is how Tokio, async-std, libuv, and libevent work internally. Understanding async runtimes is essential for building high-performance servers, databases, and network applications.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Async Runtime                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                         Executor                              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │   │
│  │  │   Task      │  │   Task      │  │   Task      │           │   │
│  │  │   Queue     │  │   Queue     │  │   Queue     │           │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘           │   │
│  │         │                │                │                   │   │
│  │  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐           │   │
│  │  │   Worker    │  │   Worker    │  │   Worker    │           │   │
│  │  │   Thread    │  │   Thread    │  │   Thread    │           │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘           │   │
│  │         │                │                │                   │   │
│  │         └────────────────┼────────────────┘                   │   │
│  │                          │                                    │   │
│  │                   ┌──────▼──────┐                             │   │
│  │                   │Work Stealing│                             │   │
│  │                   │    Deque    │                             │   │
│  │                   └─────────────┘                             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                               │                                      │
│  ┌────────────────────────────┼────────────────────────────────┐    │
│  │                            │                                 │    │
│  │  ┌────────────┐     ┌──────▼─────┐     ┌────────────┐       │    │
│  │  │   Timer    │     │   Reactor  │     │   Parker   │       │    │
│  │  │   Wheel    │     │(epoll/kqueue)    │   /Waker   │       │    │
│  │  └────────────┘     └────────────┘     └────────────┘       │    │
│  │         I/O Driver Layer                                     │    │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. Reactor (I/O Driver)
**Responsibilities:**
- Monitor file descriptors for readiness
- Translate OS events to runtime notifications
- Manage registration/deregistration
- Handle edge-triggered vs level-triggered events

**Implementation:**
```rust
pub struct Reactor {
    /// Platform-specific selector (epoll on Linux, kqueue on BSD/macOS)
    selector: Selector,
    /// Token to waker mapping
    wakers: Mutex<HashMap<Token, Waker>>,
    /// Current token counter
    next_token: AtomicUsize,
}

impl Reactor {
    pub fn new() -> io::Result<Self> {
        Ok(Self {
            selector: Selector::new()?,
            wakers: Mutex::new(HashMap::new()),
            next_token: AtomicUsize::new(0),
        })
    }

    /// Register interest in events for a file descriptor
    pub fn register(
        &self,
        fd: RawFd,
        interest: Interest,
        waker: Waker,
    ) -> io::Result<Token> {
        let token = Token(self.next_token.fetch_add(1, Ordering::Relaxed));

        self.selector.register(fd, token, interest)?;
        self.wakers.lock().unwrap().insert(token, waker);

        Ok(token)
    }

    /// Deregister a file descriptor
    pub fn deregister(&self, fd: RawFd, token: Token) -> io::Result<()> {
        self.selector.deregister(fd)?;
        self.wakers.lock().unwrap().remove(&token);
        Ok(())
    }

    /// Poll for events with timeout
    pub fn poll(&self, timeout: Option<Duration>) -> io::Result<()> {
        let mut events = Events::with_capacity(1024);
        self.selector.select(&mut events, timeout)?;

        let wakers = self.wakers.lock().unwrap();
        for event in &events {
            if let Some(waker) = wakers.get(&event.token()) {
                waker.wake_by_ref();
            }
        }

        Ok(())
    }
}

// Platform-specific selector
#[cfg(target_os = "linux")]
pub struct Selector {
    epoll_fd: RawFd,
}

#[cfg(target_os = "linux")]
impl Selector {
    pub fn new() -> io::Result<Self> {
        let fd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { epoll_fd: fd })
    }

    pub fn register(&self, fd: RawFd, token: Token, interest: Interest) -> io::Result<()> {
        let mut event = libc::epoll_event {
            events: interest_to_epoll(interest) as u32,
            u64: token.0 as u64,
        };

        let op = libc::EPOLL_CTL_ADD;
        let result = unsafe {
            libc::epoll_ctl(self.epoll_fd, op, fd, &mut event)
        };

        if result < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    pub fn select(&self, events: &mut Events, timeout: Option<Duration>) -> io::Result<usize> {
        let timeout_ms = timeout
            .map(|d| d.as_millis() as i32)
            .unwrap_or(-1);

        let result = unsafe {
            libc::epoll_wait(
                self.epoll_fd,
                events.as_mut_ptr(),
                events.capacity() as i32,
                timeout_ms,
            )
        };

        if result < 0 {
            Err(io::Error::last_os_error())
        } else {
            unsafe { events.set_len(result as usize) };
            Ok(result as usize)
        }
    }
}

fn interest_to_epoll(interest: Interest) -> i32 {
    let mut events = libc::EPOLLET; // Edge-triggered

    if interest.is_readable() {
        events |= libc::EPOLLIN;
    }
    if interest.is_writable() {
        events |= libc::EPOLLOUT;
    }

    events
}
```

#### 2. Task System
**Responsibilities:**
- Represent units of async work
- Store future state
- Handle wake-up notifications
- Support cancellation

```rust
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

/// A spawned task
pub struct Task {
    /// The future being executed
    future: Mutex<Pin<Box<dyn Future<Output = ()> + Send>>>,
    /// Executor handle for re-scheduling
    executor: Arc<Executor>,
    /// Task state
    state: AtomicU8,
}

const TASK_IDLE: u8 = 0;
const TASK_SCHEDULED: u8 = 1;
const TASK_RUNNING: u8 = 2;
const TASK_COMPLETED: u8 = 3;

impl Task {
    pub fn new<F>(future: F, executor: Arc<Executor>) -> Arc<Self>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        Arc::new(Self {
            future: Mutex::new(Box::pin(future)),
            executor,
            state: AtomicU8::new(TASK_SCHEDULED),
        })
    }

    /// Poll the task's future
    pub fn poll(self: Arc<Self>) -> bool {
        // Mark as running
        self.state.store(TASK_RUNNING, Ordering::SeqCst);

        // Create waker from task
        let waker = self.clone().into_waker();
        let mut cx = Context::from_waker(&waker);

        // Poll future
        let mut future = self.future.lock().unwrap();
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(()) => {
                self.state.store(TASK_COMPLETED, Ordering::SeqCst);
                true
            }
            Poll::Pending => {
                self.state.store(TASK_IDLE, Ordering::SeqCst);
                false
            }
        }
    }

    /// Convert task to a waker
    fn into_waker(self: Arc<Self>) -> Waker {
        let ptr = Arc::into_raw(self) as *const ();
        let vtable = &TASK_WAKER_VTABLE;
        unsafe { Waker::from_raw(RawWaker::new(ptr, vtable)) }
    }

    /// Wake up the task (schedule for execution)
    fn wake(self: Arc<Self>) {
        let prev = self.state.swap(TASK_SCHEDULED, Ordering::SeqCst);
        if prev == TASK_IDLE {
            self.executor.schedule(self);
        }
    }
}

// Waker vtable for Task
static TASK_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    // clone
    |ptr| {
        let arc = unsafe { Arc::from_raw(ptr as *const Task) };
        let cloned = arc.clone();
        std::mem::forget(arc);
        RawWaker::new(Arc::into_raw(cloned) as *const (), &TASK_WAKER_VTABLE)
    },
    // wake
    |ptr| {
        let arc = unsafe { Arc::from_raw(ptr as *const Task) };
        arc.wake();
    },
    // wake_by_ref
    |ptr| {
        let arc = unsafe { Arc::from_raw(ptr as *const Task) };
        arc.clone().wake();
        std::mem::forget(arc);
    },
    // drop
    |ptr| {
        unsafe { Arc::from_raw(ptr as *const Task) };
    },
);
```

#### 3. Executor
**Responsibilities:**
- Manage worker threads
- Distribute tasks across workers
- Handle work stealing
- Coordinate shutdown

```rust
pub struct Executor {
    /// Worker threads
    workers: Vec<Worker>,
    /// Global task queue (for spawning)
    global_queue: SegQueue<Arc<Task>>,
    /// Shutdown signal
    shutdown: AtomicBool,
    /// Reactor for I/O
    reactor: Arc<Reactor>,
}

pub struct Worker {
    /// Thread handle
    handle: Option<JoinHandle<()>>,
    /// Local task queue (LIFO for cache locality)
    local_queue: Injector<Arc<Task>>,
    /// Stealer handles for other workers
    stealers: Vec<Stealer<Arc<Task>>>,
}

impl Executor {
    pub fn new(num_workers: usize) -> Arc<Self> {
        let reactor = Arc::new(Reactor::new().unwrap());

        // Create work-stealing deques
        let mut workers = Vec::with_capacity(num_workers);
        let mut stealers = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            let (worker, stealer) = deque::new();
            workers.push(worker);
            stealers.push(stealer);
        }

        let executor = Arc::new(Self {
            workers: Vec::new(),
            global_queue: SegQueue::new(),
            shutdown: AtomicBool::new(false),
            reactor,
        });

        // Spawn worker threads
        // (implementation continues)

        executor
    }

    /// Spawn a new task
    pub fn spawn<F>(&self, future: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let task = Task::new(future, Arc::clone(&self));
        self.global_queue.push(task);
        self.notify_workers();
    }

    /// Schedule a task for execution
    pub fn schedule(&self, task: Arc<Task>) {
        // Push to global queue
        self.global_queue.push(task);
        self.notify_workers();
    }

    /// Worker loop
    fn worker_loop(
        &self,
        worker_id: usize,
        local: Worker<Arc<Task>>,
        stealers: Vec<Stealer<Arc<Task>>>,
    ) {
        loop {
            if self.shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Try to get a task
            let task = local.pop()
                .or_else(|| self.global_queue.steal().success())
                .or_else(|| {
                    // Work stealing from other workers
                    stealers.iter()
                        .map(|s| s.steal())
                        .find_map(|s| s.success())
                });

            match task {
                Some(task) => {
                    task.poll();
                }
                None => {
                    // No work, poll reactor or park
                    self.reactor.poll(Some(Duration::from_millis(10))).ok();
                }
            }
        }
    }
}
```

#### 4. Timer Wheel
**Responsibilities:**
- Efficient timer management
- O(1) insert and cancel
- Handle many concurrent timers
- Support various timeout durations

```rust
/// Hierarchical timer wheel
pub struct TimerWheel {
    /// Wheels at different granularities
    wheels: [Wheel; 4],
    /// Current time in ticks
    current_tick: u64,
    /// Tick duration
    tick_duration: Duration,
}

struct Wheel {
    slots: Vec<Vec<TimerEntry>>,
    mask: usize,
}

struct TimerEntry {
    deadline: u64,
    waker: Waker,
}

impl TimerWheel {
    pub fn new(tick_duration: Duration) -> Self {
        Self {
            wheels: [
                Wheel::new(256),   // 256 ticks = 256ms at 1ms granularity
                Wheel::new(64),    // 64 * 256 = 16 seconds
                Wheel::new(64),    // 64 * 16s = ~17 minutes
                Wheel::new(64),    // 64 * 17m = ~18 hours
            ],
            current_tick: 0,
            tick_duration,
        }
    }

    /// Insert a timer
    pub fn insert(&mut self, deadline: Instant, waker: Waker) -> TimerHandle {
        let ticks = self.instant_to_ticks(deadline);
        let delta = ticks.saturating_sub(self.current_tick);

        // Determine which wheel and slot
        let (wheel_idx, slot_idx) = self.calculate_slot(delta);

        let entry = TimerEntry { deadline: ticks, waker };
        self.wheels[wheel_idx].slots[slot_idx].push(entry);

        TimerHandle { ticks }
    }

    /// Advance time and fire expired timers
    pub fn advance(&mut self, now: Instant) -> Vec<Waker> {
        let target_tick = self.instant_to_ticks(now);
        let mut wakers = Vec::new();

        while self.current_tick < target_tick {
            self.current_tick += 1;

            // Check wheel 0 slot
            let slot_idx = (self.current_tick as usize) & self.wheels[0].mask;
            let slot = std::mem::take(&mut self.wheels[0].slots[slot_idx]);

            for entry in slot {
                if entry.deadline <= self.current_tick {
                    wakers.push(entry.waker);
                } else {
                    // Reinsert into appropriate slot
                    let delta = entry.deadline - self.current_tick;
                    let (wheel_idx, new_slot) = self.calculate_slot(delta);
                    self.wheels[wheel_idx].slots[new_slot].push(entry);
                }
            }

            // Cascade from higher wheels if needed
            if slot_idx == 0 {
                self.cascade(1);
            }
        }

        wakers
    }

    fn calculate_slot(&self, delta: u64) -> (usize, usize) {
        if delta < 256 {
            (0, delta as usize)
        } else if delta < 256 * 64 {
            (1, (delta / 256) as usize)
        } else if delta < 256 * 64 * 64 {
            (2, (delta / (256 * 64)) as usize)
        } else {
            (3, (delta / (256 * 64 * 64)).min(63) as usize)
        }
    }

    fn cascade(&mut self, wheel_idx: usize) {
        if wheel_idx >= self.wheels.len() {
            return;
        }

        // Move entries from this wheel's current slot to lower wheels
        let slot_idx = ((self.current_tick >> (8 + 6 * (wheel_idx - 1))) as usize)
            & self.wheels[wheel_idx].mask;

        let entries = std::mem::take(&mut self.wheels[wheel_idx].slots[slot_idx]);

        for entry in entries {
            let delta = entry.deadline.saturating_sub(self.current_tick);
            let (new_wheel, new_slot) = self.calculate_slot(delta);
            self.wheels[new_wheel].slots[new_slot].push(entry);
        }

        // Continue cascade if needed
        if slot_idx == 0 {
            self.cascade(wheel_idx + 1);
        }
    }
}

impl Wheel {
    fn new(num_slots: usize) -> Self {
        Self {
            slots: (0..num_slots).map(|_| Vec::new()).collect(),
            mask: num_slots - 1,
        }
    }
}
```

---

## Core Internals

### Future State Machine

```rust
// What the compiler generates for:
// async fn example() {
//     let a = foo().await;
//     let b = bar(a).await;
//     b
// }

enum ExampleFuture {
    State0 { foo_future: FooFuture },
    State1 { a: FooOutput, bar_future: BarFuture },
    Completed,
}

impl Future for ExampleFuture {
    type Output = BarOutput;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };

        loop {
            match this {
                ExampleFuture::State0 { foo_future } => {
                    let foo_future = unsafe { Pin::new_unchecked(foo_future) };
                    match foo_future.poll(cx) {
                        Poll::Ready(a) => {
                            let bar_future = bar(a);
                            *this = ExampleFuture::State1 {
                                a: a,
                                bar_future
                            };
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }
                ExampleFuture::State1 { bar_future, .. } => {
                    let bar_future = unsafe { Pin::new_unchecked(bar_future) };
                    match bar_future.poll(cx) {
                        Poll::Ready(b) => {
                            *this = ExampleFuture::Completed;
                            return Poll::Ready(b);
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }
                ExampleFuture::Completed => {
                    panic!("polled after completion");
                }
            }
        }
    }
}
```

### Pinning and Self-Referential Structs

```rust
/// Async read operation that borrows from buffer
pub struct AsyncRead<'a> {
    fd: RawFd,
    buffer: &'a mut [u8],
    state: ReadState,
}

enum ReadState {
    Initial,
    Reading { token: Token },
    Complete(usize),
}

impl<'a> Future for AsyncRead<'a> {
    type Output = io::Result<usize>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            match self.state {
                ReadState::Initial => {
                    // Register for read events
                    let token = REACTOR.with(|r| {
                        r.register(self.fd, Interest::READABLE, cx.waker().clone())
                    })?;
                    self.state = ReadState::Reading { token };
                }
                ReadState::Reading { token } => {
                    // Try non-blocking read
                    match read_nonblocking(self.fd, self.buffer) {
                        Ok(n) => {
                            REACTOR.with(|r| r.deregister(self.fd, token)).ok();
                            return Poll::Ready(Ok(n));
                        }
                        Err(e) if e.kind() == WouldBlock => {
                            return Poll::Pending;
                        }
                        Err(e) => {
                            REACTOR.with(|r| r.deregister(self.fd, token)).ok();
                            return Poll::Ready(Err(e));
                        }
                    }
                }
                ReadState::Complete(n) => {
                    return Poll::Ready(Ok(n));
                }
            }
        }
    }
}
```

### Work-Stealing Deque

```rust
/// Lock-free work-stealing deque (Chase-Lev)
pub struct Deque<T> {
    /// Array of tasks
    buffer: AtomicPtr<Buffer<T>>,
    /// Bottom index (modified by owner)
    bottom: AtomicIsize,
    /// Top index (modified by stealers)
    top: AtomicIsize,
}

struct Buffer<T> {
    data: Box<[MaybeUninit<T>]>,
    capacity: usize,
}

impl<T> Deque<T> {
    /// Push to bottom (owner only)
    pub fn push(&self, item: T) {
        let bottom = self.bottom.load(Ordering::Relaxed);
        let top = self.top.load(Ordering::Acquire);
        let buffer = unsafe { &*self.buffer.load(Ordering::Relaxed) };

        let size = bottom.wrapping_sub(top);
        if size >= buffer.capacity as isize - 1 {
            // Grow buffer
            self.grow(bottom, top);
        }

        let buffer = unsafe { &*self.buffer.load(Ordering::Relaxed) };
        unsafe {
            buffer.write(bottom, item);
        }

        atomic::fence(Ordering::Release);
        self.bottom.store(bottom.wrapping_add(1), Ordering::Relaxed);
    }

    /// Pop from bottom (owner only)
    pub fn pop(&self) -> Option<T> {
        let bottom = self.bottom.load(Ordering::Relaxed).wrapping_sub(1);
        let buffer = unsafe { &*self.buffer.load(Ordering::Relaxed) };

        self.bottom.store(bottom, Ordering::Relaxed);
        atomic::fence(Ordering::SeqCst);

        let top = self.top.load(Ordering::Relaxed);
        let size = bottom.wrapping_sub(top);

        if size < 0 {
            self.bottom.store(top, Ordering::Relaxed);
            return None;
        }

        let item = unsafe { buffer.read(bottom) };

        if size > 0 {
            return Some(item);
        }

        // Last item, race with stealers
        if self.top.compare_exchange(
            top,
            top.wrapping_add(1),
            Ordering::SeqCst,
            Ordering::Relaxed,
        ).is_err() {
            // Stolen
            self.bottom.store(top.wrapping_add(1), Ordering::Relaxed);
            return None;
        }

        self.bottom.store(top.wrapping_add(1), Ordering::Relaxed);
        Some(item)
    }

    /// Steal from top (other workers)
    pub fn steal(&self) -> Option<T> {
        let top = self.top.load(Ordering::Acquire);
        atomic::fence(Ordering::SeqCst);
        let bottom = self.bottom.load(Ordering::Acquire);

        let size = bottom.wrapping_sub(top);
        if size <= 0 {
            return None;
        }

        let buffer = unsafe { &*self.buffer.load(Ordering::Acquire) };
        let item = unsafe { buffer.read(top) };

        if self.top.compare_exchange(
            top,
            top.wrapping_add(1),
            Ordering::SeqCst,
            Ordering::Relaxed,
        ).is_ok() {
            Some(item)
        } else {
            None
        }
    }
}
```

---

## Enterprise Features

### 1. Backpressure Handling

```rust
/// Bounded channel with backpressure
pub struct BoundedChannel<T> {
    buffer: Mutex<VecDeque<T>>,
    capacity: usize,
    send_wakers: Mutex<Vec<Waker>>,
    recv_wakers: Mutex<Vec<Waker>>,
}

impl<T> BoundedChannel<T> {
    pub fn send(&self, item: T) -> Send<'_, T> {
        Send {
            channel: self,
            item: Some(item),
        }
    }

    pub fn recv(&self) -> Recv<'_, T> {
        Recv { channel: self }
    }
}

pub struct Send<'a, T> {
    channel: &'a BoundedChannel<T>,
    item: Option<T>,
}

impl<T> Future for Send<'_, T> {
    type Output = Result<(), SendError<T>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut buffer = self.channel.buffer.lock().unwrap();

        if buffer.len() < self.channel.capacity {
            buffer.push_back(self.item.take().unwrap());

            // Wake receivers
            let wakers = std::mem::take(&mut *self.channel.recv_wakers.lock().unwrap());
            for waker in wakers {
                waker.wake();
            }

            Poll::Ready(Ok(()))
        } else {
            // Channel full, register waker
            self.channel.send_wakers.lock().unwrap().push(cx.waker().clone());
            Poll::Pending
        }
    }
}
```

### 2. Prioritized Tasks

```rust
/// Priority-based task scheduler
pub struct PriorityScheduler {
    queues: [Mutex<VecDeque<Arc<Task>>>; 4],
}

impl PriorityScheduler {
    pub fn spawn_with_priority<F>(&self, priority: Priority, future: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let task = Task::new(future);
        let queue_idx = priority as usize;
        self.queues[queue_idx].lock().unwrap().push_back(task);
    }

    pub fn get_next_task(&self) -> Option<Arc<Task>> {
        // Check queues from highest to lowest priority
        for queue in &self.queues {
            if let Some(task) = queue.lock().unwrap().pop_front() {
                return Some(task);
            }
        }
        None
    }
}

#[derive(Clone, Copy)]
pub enum Priority {
    Critical = 0,
    High = 1,
    Normal = 2,
    Low = 3,
}
```

### 3. Cancellation Propagation

```rust
/// Cancellation token for cooperative cancellation
pub struct CancellationToken {
    inner: Arc<CancellationInner>,
}

struct CancellationInner {
    cancelled: AtomicBool,
    wakers: Mutex<Vec<Waker>>,
    children: Mutex<Vec<CancellationToken>>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(CancellationInner {
                cancelled: AtomicBool::new(false),
                wakers: Mutex::new(Vec::new()),
                children: Mutex::new(Vec::new()),
            }),
        }
    }

    pub fn child_token(&self) -> CancellationToken {
        let child = CancellationToken::new();
        self.inner.children.lock().unwrap().push(child.clone());

        // If already cancelled, cancel child immediately
        if self.is_cancelled() {
            child.cancel();
        }

        child
    }

    pub fn cancel(&self) {
        if self.inner.cancelled.swap(true, Ordering::SeqCst) {
            return; // Already cancelled
        }

        // Wake all waiting tasks
        let wakers = std::mem::take(&mut *self.inner.wakers.lock().unwrap());
        for waker in wakers {
            waker.wake();
        }

        // Cancel all children
        let children = self.inner.children.lock().unwrap();
        for child in children.iter() {
            child.cancel();
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::SeqCst)
    }

    /// Future that completes when cancelled
    pub fn cancelled(&self) -> Cancelled {
        Cancelled { token: self.clone() }
    }
}

pub struct Cancelled {
    token: CancellationToken,
}

impl Future for Cancelled {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.token.is_cancelled() {
            Poll::Ready(())
        } else {
            self.token.inner.wakers.lock().unwrap().push(cx.waker().clone());
            Poll::Pending
        }
    }
}

// Usage with select
pub async fn cancellable_operation(token: CancellationToken) -> Result<(), Cancelled> {
    tokio::select! {
        result = do_work() => Ok(result),
        _ = token.cancelled() => Err(Cancelled),
    }
}
```

### 4. Structured Concurrency

```rust
/// Scope for structured concurrency (all tasks complete before scope ends)
pub struct Scope<'env> {
    tasks: Mutex<Vec<JoinHandle<()>>>,
    _marker: PhantomData<&'env ()>,
}

impl<'env> Scope<'env> {
    pub fn spawn<F>(&self, future: F)
    where
        F: Future<Output = ()> + Send + 'env,
    {
        // Extend lifetime (safe because we join all before returning)
        let future: Pin<Box<dyn Future<Output = ()> + Send>> =
            unsafe { std::mem::transmute(Box::pin(future)) };

        let handle = spawn(future);
        self.tasks.lock().unwrap().push(handle);
    }
}

/// Run scoped tasks
pub async fn scope<'env, F, R>(f: F) -> R
where
    F: FnOnce(&Scope<'env>) -> R,
{
    let scope = Scope {
        tasks: Mutex::new(Vec::new()),
        _marker: PhantomData,
    };

    let result = f(&scope);

    // Wait for all spawned tasks
    let tasks = std::mem::take(&mut *scope.tasks.lock().unwrap());
    for task in tasks {
        task.await;
    }

    result
}

// Usage
async fn example() {
    let data = vec![1, 2, 3, 4, 5];

    scope(|s| {
        for item in &data {
            s.spawn(async move {
                process(item).await;
            });
        }
    }).await;
    // All tasks complete here, safe to drop data
}
```

---

## Performance Considerations

### CPU Efficiency
- **Work stealing:** Minimize contention, maximize CPU utilization
- **Cache locality:** Process tasks on same thread when possible
- **Batch waking:** Reduce syscall overhead by batching notifications
- **Inline small futures:** Avoid heap allocations for small tasks

### Memory Efficiency
- **Task size:** Minimize Future enum size
- **Avoid boxing:** Use generics over trait objects where possible
- **Pool allocations:** Reuse task memory

### I/O Efficiency
- **Edge-triggered events:** Reduce number of syscalls
- **Vectored I/O:** Use readv/writev for scatter-gather
- **Zero-copy:** Use splice/sendfile where available

### Benchmarks Target
| Operation | Target | Notes |
|-----------|--------|-------|
| Task spawn | <100ns | Without contention |
| Task switch | <50ns | Hot path |
| Timer insert | O(1) | Timer wheel |
| I/O registration | <1us | |
| Epoll wait | <1us | Per event |

---

## Stretch Goals

### 1. Multi-Threaded Runtime

```rust
/// Multi-threaded executor with runtime-controlled thread count
pub struct MultiThreadRuntime {
    /// Worker threads
    workers: Vec<WorkerThread>,
    /// Shared scheduler
    scheduler: Arc<Scheduler>,
    /// I/O driver (shared)
    driver: Arc<IoDriver>,
}

impl MultiThreadRuntime {
    pub fn new() -> Builder {
        Builder::new()
    }

    pub fn block_on<F: Future>(self, future: F) -> F::Output {
        let main_task = Task::new(future);
        self.scheduler.push(main_task.clone());

        // Start workers
        for worker in &mut self.workers {
            worker.start();
        }

        // Run main task on current thread
        loop {
            if let Some(result) = main_task.try_take_result() {
                return result;
            }

            // Drive I/O
            self.driver.turn(Some(Duration::from_millis(1)));

            // Run some tasks
            for _ in 0..61 {
                if let Some(task) = self.scheduler.pop() {
                    task.poll();
                } else {
                    break;
                }
            }
        }
    }
}
```

### 2. IO_uring Backend

```rust
#[cfg(target_os = "linux")]
pub struct IoUringDriver {
    ring: IoUring,
    pending: HashMap<u64, Waker>,
    next_user_data: u64,
}

#[cfg(target_os = "linux")]
impl IoUringDriver {
    pub fn new(entries: u32) -> io::Result<Self> {
        let ring = IoUring::builder()
            .setup_sqpoll(1000)
            .build(entries)?;

        Ok(Self {
            ring,
            pending: HashMap::new(),
            next_user_data: 0,
        })
    }

    pub fn submit_read(
        &mut self,
        fd: RawFd,
        buf: &mut [u8],
        waker: Waker,
    ) -> u64 {
        let user_data = self.next_user_data;
        self.next_user_data += 1;

        let entry = opcode::Read::new(types::Fd(fd), buf.as_mut_ptr(), buf.len() as u32)
            .build()
            .user_data(user_data);

        unsafe {
            self.ring.submission().push(&entry).unwrap();
        }

        self.pending.insert(user_data, waker);
        user_data
    }

    pub fn poll(&mut self) -> io::Result<()> {
        self.ring.submit_and_wait(1)?;

        for cqe in self.ring.completion() {
            let user_data = cqe.user_data();
            if let Some(waker) = self.pending.remove(&user_data) {
                waker.wake();
            }
        }

        Ok(())
    }
}
```

### 3. Async-Aware Mutex

```rust
/// Async mutex that yields to scheduler while waiting
pub struct Mutex<T> {
    state: AtomicU32,
    waiters: SegQueue<Waker>,
    data: UnsafeCell<T>,
}

const UNLOCKED: u32 = 0;
const LOCKED: u32 = 1;

unsafe impl<T: Send> Send for Mutex<T> {}
unsafe impl<T: Send> Sync for Mutex<T> {}

impl<T> Mutex<T> {
    pub fn new(data: T) -> Self {
        Self {
            state: AtomicU32::new(UNLOCKED),
            waiters: SegQueue::new(),
            data: UnsafeCell::new(data),
        }
    }

    pub async fn lock(&self) -> MutexGuard<'_, T> {
        loop {
            // Try to acquire
            if self.state.compare_exchange(
                UNLOCKED,
                LOCKED,
                Ordering::Acquire,
                Ordering::Relaxed,
            ).is_ok() {
                return MutexGuard { mutex: self };
            }

            // Wait
            let wait = Wait { mutex: self };
            wait.await;
        }
    }
}

struct Wait<'a, T> {
    mutex: &'a Mutex<T>,
}

impl<T> Future for Wait<'_, T> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        // Try once more
        if self.mutex.state.compare_exchange(
            UNLOCKED,
            LOCKED,
            Ordering::Acquire,
            Ordering::Relaxed,
        ).is_ok() {
            return Poll::Ready(());
        }

        // Register waker and yield
        self.mutex.waiters.push(cx.waker().clone());
        Poll::Pending
    }
}

pub struct MutexGuard<'a, T> {
    mutex: &'a Mutex<T>,
}

impl<T> Drop for MutexGuard<'_, T> {
    fn drop(&mut self) {
        self.mutex.state.store(UNLOCKED, Ordering::Release);

        // Wake next waiter
        if let Some(waker) = self.mutex.waiters.pop() {
            waker.wake();
        }
    }
}
```

---

## Testing Strategy

### Unit Tests
- Future state machine transitions
- Timer wheel correctness
- Work-stealing deque
- Waker implementation

### Integration Tests
- End-to-end async operations
- Multi-task coordination
- I/O operations
- Timer accuracy

### Stress Tests
- Many concurrent tasks (10K+)
- Heavy work stealing
- Timer precision under load
- Memory under sustained load

### Correctness Tests
- Loom for concurrency bugs
- Miri for undefined behavior
- Thread sanitizer
- Address sanitizer

---

## Implementation Phases

### Phase 1: Core Reactor (Week 1-3)
- [ ] Platform selector (epoll/kqueue)
- [ ] Event registration/deregistration
- [ ] Basic polling loop
- [ ] Interest flags (read/write)

### Phase 2: Task System (Week 4-5)
- [ ] Task structure
- [ ] Waker implementation
- [ ] Basic future polling
- [ ] Task lifecycle

### Phase 3: Executor (Week 6-7)
- [ ] Single-threaded executor
- [ ] Task queue
- [ ] spawn() function
- [ ] block_on()

### Phase 4: Async I/O (Week 8-9)
- [ ] TcpStream
- [ ] TcpListener
- [ ] Async read/write
- [ ] Non-blocking operations

### Phase 5: Timers (Week 10)
- [ ] Timer wheel
- [ ] sleep() function
- [ ] timeout()
- [ ] interval()

### Phase 6: Advanced (Week 11-14)
- [ ] Work-stealing
- [ ] Multi-threaded executor
- [ ] Channels (mpsc)
- [ ] Cancellation
- [ ] Structured concurrency

---

## References

- [Tokio Internals](https://tokio.rs/tokio/tutorial)
- [Rust Async Book](https://rust-lang.github.io/async-book/)
- [libuv Design Overview](https://docs.libuv.org/en/v1.x/design.html)
- [Work-Stealing Paper](https://dl.acm.org/doi/10.1145/324133.324234)
- [Timer Wheel Paper](http://www.cs.columbia.edu/~nahum/w6998/papers/sosp87-timing-wheels.pdf)
- [io_uring Documentation](https://kernel.dk/io_uring.pdf)
- [Chase-Lev Deque](https://www.dre.vanderbilt.edu/~schmidt/PDF/work-stealing-dequeue.pdf)
