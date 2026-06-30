//! Process management.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

/// Process ID type.
pub type Pid = u64;

/// User ID type.
pub type Uid = u32;

/// Group ID type.
pub type Gid = u32;

/// File descriptor type.
pub type Fd = i32;

/// Global PID counter.
static NEXT_PID: AtomicU64 = AtomicU64::new(1);

/// Generate a new PID.
fn allocate_pid() -> Pid {
    NEXT_PID.fetch_add(1, Ordering::SeqCst)
}

/// Process state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessState {
    /// Ready to run.
    Ready,
    /// Currently running.
    Running,
    /// Blocked waiting for something.
    Blocked(BlockReason),
    /// Terminated but not yet reaped.
    Zombie,
    /// Stopped by signal.
    Stopped,
}

/// Reason for blocking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockReason {
    /// Waiting for I/O.
    Io,
    /// Waiting for child process.
    Child,
    /// Sleeping.
    Sleep,
    /// Waiting for signal.
    Signal,
}

/// CPU context for context switching.
#[derive(Debug, Clone, Default)]
#[repr(C)]
pub struct CpuContext {
    // General purpose registers
    pub rax: u64,
    pub rbx: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub rbp: u64,
    pub rsp: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,

    // Instruction pointer and flags
    pub rip: u64,
    pub rflags: u64,

    // Segment selectors
    pub cs: u64,
    pub ss: u64,
}

/// File descriptor entry.
#[derive(Debug, Clone)]
pub struct FileDescriptor {
    /// File path.
    pub path: String,
    /// Current offset.
    pub offset: u64,
    /// Flags.
    pub flags: u32,
}

/// Signal handler action.
#[derive(Debug, Clone, Copy)]
pub enum SignalAction {
    /// Default action.
    Default,
    /// Ignore signal.
    Ignore,
    /// Custom handler address.
    Handler(u64),
}

/// Process Control Block.
pub struct Process {
    /// Process ID.
    pub pid: Pid,
    /// Parent process ID.
    pub ppid: Pid,
    /// Process state.
    pub state: ProcessState,

    /// CPU context.
    pub context: CpuContext,

    /// Memory info.
    pub heap_start: u64,
    pub heap_end: u64,
    pub stack_top: u64,
    pub stack_bottom: u64,

    /// Open file descriptors.
    pub file_descriptors: BTreeMap<Fd, FileDescriptor>,
    /// Next file descriptor number.
    pub next_fd: Fd,

    /// Credentials.
    pub uid: Uid,
    pub gid: Gid,
    pub euid: Uid,
    pub egid: Gid,

    /// Signal handlers.
    pub signal_handlers: [SignalAction; 32],
    /// Pending signals (bitmask).
    pub pending_signals: u32,
    /// Signal mask (blocked signals).
    pub signal_mask: u32,

    /// Scheduling info.
    pub priority: i32,
    pub nice: i32,
    pub time_slice: u64,

    /// Statistics.
    pub user_time: u64,
    pub system_time: u64,
    pub start_time: u64,

    /// Exit code if zombie.
    pub exit_code: Option<i32>,

    /// Current working directory.
    pub cwd: String,

    /// Process name.
    pub name: String,
}

impl Process {
    /// Create a new process.
    pub fn new(name: String) -> Self {
        let pid = allocate_pid();

        Self {
            pid,
            ppid: 0,
            state: ProcessState::Ready,
            context: CpuContext::default(),
            heap_start: 0,
            heap_end: 0,
            stack_top: 0,
            stack_bottom: 0,
            file_descriptors: BTreeMap::new(),
            next_fd: 3, // 0, 1, 2 reserved for stdin/stdout/stderr
            uid: 0,
            gid: 0,
            euid: 0,
            egid: 0,
            signal_handlers: [SignalAction::Default; 32],
            pending_signals: 0,
            signal_mask: 0,
            priority: 0,
            nice: 0,
            time_slice: 10,
            user_time: 0,
            system_time: 0,
            start_time: 0,
            exit_code: None,
            cwd: String::from("/"),
            name,
        }
    }

    /// Fork this process.
    pub fn fork(&self) -> Box<Process> {
        let mut child = Process::new(self.name.clone());
        child.ppid = self.pid;
        child.context = self.context.clone();
        child.heap_start = self.heap_start;
        child.heap_end = self.heap_end;
        child.stack_top = self.stack_top;
        child.stack_bottom = self.stack_bottom;
        child.uid = self.uid;
        child.gid = self.gid;
        child.euid = self.euid;
        child.egid = self.egid;
        child.signal_handlers = self.signal_handlers;
        child.signal_mask = self.signal_mask;
        child.priority = self.priority;
        child.nice = self.nice;
        child.cwd = self.cwd.clone();

        // Clone file descriptors
        for (fd, desc) in &self.file_descriptors {
            child.file_descriptors.insert(*fd, desc.clone());
        }
        child.next_fd = self.next_fd;

        Box::new(child)
    }

    /// Exit the process.
    pub fn exit(&mut self, code: i32) {
        self.state = ProcessState::Zombie;
        self.exit_code = Some(code);

        // Close all file descriptors
        self.file_descriptors.clear();
    }

    /// Allocate a file descriptor.
    pub fn allocate_fd(&mut self) -> Fd {
        let fd = self.next_fd;
        self.next_fd += 1;
        fd
    }

    /// Get a file descriptor.
    pub fn get_fd(&self, fd: Fd) -> Option<&FileDescriptor> {
        self.file_descriptors.get(&fd)
    }

    /// Get a mutable file descriptor.
    pub fn get_fd_mut(&mut self, fd: Fd) -> Option<&mut FileDescriptor> {
        self.file_descriptors.get_mut(&fd)
    }

    /// Close a file descriptor.
    pub fn close_fd(&mut self, fd: Fd) -> bool {
        self.file_descriptors.remove(&fd).is_some()
    }

    /// Send a signal to this process.
    pub fn send_signal(&mut self, signal: u32) {
        if signal < 32 {
            self.pending_signals |= 1 << signal;
        }
    }

    /// Check if process has pending signals.
    pub fn has_pending_signals(&self) -> bool {
        (self.pending_signals & !self.signal_mask) != 0
    }

    /// Get next pending signal.
    pub fn next_signal(&mut self) -> Option<u32> {
        let deliverable = self.pending_signals & !self.signal_mask;
        if deliverable == 0 {
            return None;
        }

        let signal = deliverable.trailing_zeros();
        self.pending_signals &= !(1 << signal);
        Some(signal)
    }
}

/// Initialize process management.
pub fn init() {
    // Create idle process
    let _idle = Process::new(String::from("idle"));
    // In a real kernel, we would add this to the scheduler
}
