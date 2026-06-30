//! Process scheduler.

use crate::process::{Process, ProcessState};
use crate::serial_println;
use crate::syscall::SyscallError;
use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use spin::Mutex;

/// System tick counter.
static TICKS: AtomicU64 = AtomicU64::new(0);

/// Global scheduler.
static SCHEDULER: Mutex<Option<Scheduler>> = Mutex::new(None);

/// Round-robin scheduler.
pub struct Scheduler {
    /// Ready queue.
    ready_queue: VecDeque<Box<Process>>,
    /// Currently running process.
    current: Option<Box<Process>>,
    /// Blocked processes.
    blocked: VecDeque<Box<Process>>,
    /// Zombie processes waiting to be reaped.
    zombies: VecDeque<Box<Process>>,
}

impl Scheduler {
    /// Create a new scheduler.
    pub fn new() -> Self {
        Self {
            ready_queue: VecDeque::new(),
            current: None,
            blocked: VecDeque::new(),
            zombies: VecDeque::new(),
        }
    }

    /// Add a process to the ready queue.
    pub fn add(&mut self, mut process: Box<Process>) {
        process.state = ProcessState::Ready;
        self.ready_queue.push_back(process);
    }

    /// Schedule next process.
    pub fn schedule(&mut self) {
        // Save current process if running
        if let Some(mut current) = self.current.take() {
            match current.state {
                ProcessState::Running => {
                    current.state = ProcessState::Ready;
                    self.ready_queue.push_back(current);
                }
                ProcessState::Zombie => {
                    self.zombies.push_back(current);
                }
                ProcessState::Blocked(_) => {
                    self.blocked.push_back(current);
                }
                _ => {
                    self.ready_queue.push_back(current);
                }
            }
        }

        // Get next ready process
        if let Some(mut next) = self.ready_queue.pop_front() {
            next.state = ProcessState::Running;
            serial_println!("Scheduling process: {} (PID {})", next.name, next.pid);
            self.current = Some(next);
        }
    }

    /// Get current process.
    pub fn current(&self) -> Option<&Process> {
        self.current.as_ref().map(|p| p.as_ref())
    }

    /// Get current process mutably.
    pub fn current_mut(&mut self) -> Option<&mut Process> {
        self.current.as_mut().map(|p| p.as_mut())
    }

    /// Block current process.
    pub fn block_current(&mut self, reason: crate::process::BlockReason) {
        if let Some(current) = &mut self.current {
            current.state = ProcessState::Blocked(reason);
        }
        self.schedule();
    }

    /// Wake up a blocked process.
    pub fn wake(&mut self, pid: crate::process::Pid) {
        let mut i = 0;
        while i < self.blocked.len() {
            if self.blocked[i].pid == pid {
                let mut process = self.blocked.remove(i).unwrap();
                process.state = ProcessState::Ready;
                self.ready_queue.push_back(process);
                return;
            }
            i += 1;
        }
    }

    /// Reap zombie process.
    pub fn reap(&mut self, pid: crate::process::Pid) -> Option<i32> {
        let mut i = 0;
        while i < self.zombies.len() {
            if self.zombies[i].pid == pid {
                let process = self.zombies.remove(i).unwrap();
                return process.exit_code;
            }
            i += 1;
        }
        None
    }

    /// Get number of ready processes.
    pub fn ready_count(&self) -> usize {
        self.ready_queue.len()
    }

    /// Check sleeping processes and wake if needed.
    pub fn check_sleepers(&mut self, current_ticks: u64) {
        let mut to_wake = Vec::new();

        for process in &self.blocked {
            if let ProcessState::Blocked(crate::process::BlockReason::Sleep) = process.state {
                // In a real kernel, we would check wake time
                // For now, wake after some ticks
                if current_ticks % 100 == 0 {
                    to_wake.push(process.pid);
                }
            }
        }

        for pid in to_wake {
            self.wake(pid);
        }
    }
}

/// Initialize the scheduler.
pub fn init() {
    let scheduler = Scheduler::new();
    *SCHEDULER.lock() = Some(scheduler);
}

/// Handle timer tick.
pub fn tick() {
    let ticks = TICKS.fetch_add(1, Ordering::SeqCst);

    if let Some(scheduler) = SCHEDULER.lock().as_mut() {
        // Check sleepers
        scheduler.check_sleepers(ticks);

        // Decrement time slice and reschedule if needed
        if let Some(current) = scheduler.current_mut() {
            if current.time_slice > 0 {
                current.time_slice -= 1;
            } else {
                current.time_slice = 10; // Reset time slice
                scheduler.schedule();
            }
        }
    }
}

/// Get current tick count.
pub fn get_ticks() -> u64 {
    TICKS.load(Ordering::SeqCst)
}

/// Add a process to the scheduler.
pub fn add_process(process: Box<Process>) {
    if let Some(scheduler) = SCHEDULER.lock().as_mut() {
        scheduler.add(process);
    }
}

/// Get current process PID.
pub fn current_pid() -> Option<crate::process::Pid> {
    SCHEDULER.lock().as_ref()?.current().map(|p| p.pid)
}

/// Block current process.
pub fn block(reason: crate::process::BlockReason) {
    if let Some(scheduler) = SCHEDULER.lock().as_mut() {
        scheduler.block_current(reason);
    }
}

/// Wake a process.
pub fn wake(pid: crate::process::Pid) {
    if let Some(scheduler) = SCHEDULER.lock().as_mut() {
        scheduler.wake(pid);
    }
}

/// Yield current time slice.
pub fn yield_now() {
    if let Some(scheduler) = SCHEDULER.lock().as_mut() {
        scheduler.schedule();
    }
}

/// Exit current process.
pub fn exit(code: i32) {
    if let Some(scheduler) = SCHEDULER.lock().as_mut() {
        if let Some(current) = scheduler.current_mut() {
            current.exit(code);
        }
        scheduler.schedule();
    }
}

/// Wait for child process.
pub fn wait(pid: crate::process::Pid) -> Option<i32> {
    SCHEDULER.lock().as_mut()?.reap(pid)
}

/// Fork the current process.
///
/// Creates a child process that is a copy of the current process.
/// The child's context is set up so that fork returns 0 to it.
pub fn fork_current() -> Result<Box<Process>, SyscallError> {
    let mut guard = SCHEDULER.lock();
    let scheduler = guard.as_mut().ok_or(SyscallError::ESRCH)?;

    let current = scheduler.current.as_ref().ok_or(SyscallError::ESRCH)?;

    // Fork the process
    let mut child = current.fork();

    // Set child's rax to 0 (fork returns 0 to child)
    child.context.rax = 0;

    serial_println!(
        "fork: parent {} -> child {}",
        current.pid,
        child.pid
    );

    Ok(child)
}

/// Execute a new program in the current process.
///
/// Replaces the current process image with a new program.
pub fn exec_current(
    entry_point: u64,
    stack_top: u64,
    argv: &[String],
    _envp: &[String],
) -> Result<(), SyscallError> {
    let mut guard = SCHEDULER.lock();
    let scheduler = guard.as_mut().ok_or(SyscallError::ESRCH)?;

    let current = scheduler.current_mut().ok_or(SyscallError::ESRCH)?;

    // Set up user stack with argv
    // Stack layout (grows down):
    //   - null (end of envp)
    //   - null (end of argv)
    //   - argv[n] pointers
    //   - argv[0] pointer
    //   - argc
    //   - (alignment padding)
    //   - argv strings
    //   - envp strings

    // For simplicity, we set up a basic stack
    // In a real kernel, we would copy argv/envp strings to user stack
    let mut sp = stack_top;

    // Reserve space for strings (simplified)
    sp -= 4096;  // 4KB for argv/envp strings

    // Align stack to 16 bytes
    sp &= !0xF;

    // Push argc
    sp -= 8;

    // Update process context for new program
    current.context.rip = entry_point;
    current.context.rsp = sp;
    current.context.rbp = 0;
    current.context.rax = 0;
    current.context.rbx = 0;
    current.context.rcx = 0;
    current.context.rdx = 0;
    current.context.rsi = 0;
    current.context.rdi = argv.len() as u64;  // argc in rdi (System V ABI)

    // Update stack bounds
    current.stack_top = stack_top;
    current.stack_bottom = stack_top - 0x10000;  // 64KB stack

    // Reset signal handlers to default (exec clears handlers)
    current.signal_handlers = [crate::process::SignalAction::Default; 32];

    // Clear pending signals
    current.pending_signals = 0;

    serial_println!(
        "exec: pid {} entry=0x{:x} stack=0x{:x}",
        current.pid,
        entry_point,
        sp
    );

    Ok(())
}

/// Get the current process reference (for syscalls).
pub fn with_current<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&Process) -> R,
{
    let guard = SCHEDULER.lock();
    guard.as_ref()?.current().map(f)
}

/// Get the current process mutable reference (for syscalls).
pub fn with_current_mut<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&mut Process) -> R,
{
    let mut guard = SCHEDULER.lock();
    guard.as_mut()?.current_mut().map(f)
}

// ============================================================================
// Shell-compatible API wrappers
// ============================================================================

/// Fork the current process (shell-compatible API).
///
/// Returns child PID to parent, 0 to child.
pub fn fork() -> Result<crate::process::Pid, SyscallError> {
    let child = fork_current()?;
    let child_pid = child.pid;

    // Add child to scheduler
    if let Some(scheduler) = SCHEDULER.lock().as_mut() {
        scheduler.add(child);
    }

    Ok(child_pid)
}

/// Wait for a specific child process (shell-compatible API).
///
/// Returns (pid, exit_status) tuple.
pub fn waitpid(pid: crate::process::Pid, _flags: i32) -> Result<(crate::process::Pid, i32), SyscallError> {
    // Block until child exits
    loop {
        if let Some(exit_code) = wait(pid) {
            return Ok((pid, exit_code));
        }

        // Check if child exists
        let exists = SCHEDULER.lock().as_ref()
            .map(|s| {
                s.ready_queue.iter().any(|p| p.pid == pid) ||
                s.blocked.iter().any(|p| p.pid == pid) ||
                s.current.as_ref().map(|p| p.pid == pid).unwrap_or(false)
            })
            .unwrap_or(false);

        if !exists {
            return Err(SyscallError::ECHILD);
        }

        // Block current process waiting for child
        block(crate::process::BlockReason::Child);
    }
}

/// Execute a new program (shell-compatible API).
pub fn execve(path: &str, argv: &[&str], envp: &[&str]) -> Result<(), SyscallError> {
    // Load the ELF binary
    let data = crate::vfs::read_file(path)?;

    // Parse and load ELF
    let loaded = crate::elf::load_elf(&data)
        .map_err(|_| SyscallError::ENOEXEC)?;

    // Convert args to owned strings
    let argv_owned: Vec<String> = argv.iter().map(|s| String::from(*s)).collect();
    let envp_owned: Vec<String> = envp.iter().map(|s| String::from(*s)).collect();

    exec_current(loaded.entry, loaded.stack_top, &argv_owned, &envp_owned)
}

/// Yield CPU to next process (shell-compatible alias).
pub fn yield_cpu() {
    yield_now()
}

/// Send a signal to a process.
pub fn kill(pid: crate::process::Pid, signal: u32) -> Result<(), SyscallError> {
    let mut guard = SCHEDULER.lock();
    let scheduler = guard.as_mut().ok_or(SyscallError::ESRCH)?;

    // Check current process
    if let Some(current) = scheduler.current_mut() {
        if current.pid == pid {
            current.send_signal(signal);
            return Ok(());
        }
    }

    // Check ready queue
    for process in scheduler.ready_queue.iter_mut() {
        if process.pid == pid {
            process.send_signal(signal);
            return Ok(());
        }
    }

    // Check blocked queue
    for process in scheduler.blocked.iter_mut() {
        if process.pid == pid {
            process.send_signal(signal);
            return Ok(());
        }
    }

    Err(SyscallError::ESRCH)
}

/// Process info for listing.
#[derive(Debug, Clone)]
pub struct ProcessInfo {
    pub pid: crate::process::Pid,
    pub ppid: crate::process::Pid,
    pub state: ProcessState,
    pub name: String,
}

/// Wait for any child process (init-compatible API).
///
/// Used by init to reap zombies without knowing their PIDs.
pub fn waitpid_any(flags: i32) -> Result<(crate::process::Pid, i32), SyscallError> {
    let mut guard = SCHEDULER.lock();
    let scheduler = guard.as_mut().ok_or(SyscallError::ECHILD)?;

    // Check for any zombie child of current process
    let current_pid = scheduler.current.as_ref().map(|p| p.pid).unwrap_or(0);

    // Find first zombie whose parent is current
    for i in 0..scheduler.zombies.len() {
        if scheduler.zombies[i].ppid == current_pid {
            let zombie = scheduler.zombies.remove(i).unwrap();
            let pid = zombie.pid;
            let exit_code = zombie.exit_code.unwrap_or(0);
            return Ok((pid, exit_code));
        }
    }

    // No zombie found
    if flags & 1 != 0 {  // WNOHANG
        return Ok((0, 0));  // Return 0 meaning no child changed state
    }

    // Would block - for init this shouldn't happen with WNOHANG
    Err(SyscallError::ECHILD)
}

/// Re-parent children of a process to a new parent (for orphan handling).
pub fn reparent_children(old_parent: crate::process::Pid, new_parent: crate::process::Pid) {
    let mut guard = SCHEDULER.lock();
    let scheduler = match guard.as_mut() {
        Some(s) => s,
        None => return,
    };

    // Update ppid in all queues
    for process in scheduler.ready_queue.iter_mut() {
        if process.ppid == old_parent {
            process.ppid = new_parent;
        }
    }

    for process in scheduler.blocked.iter_mut() {
        if process.ppid == old_parent {
            process.ppid = new_parent;
        }
    }

    for process in scheduler.zombies.iter_mut() {
        if process.ppid == old_parent {
            process.ppid = new_parent;
        }
    }
}

/// List all processes (for ps command).
pub fn list_processes() -> Vec<ProcessInfo> {
    let guard = SCHEDULER.lock();
    let scheduler = match guard.as_ref() {
        Some(s) => s,
        None => return Vec::new(),
    };

    let mut processes = Vec::new();

    // Current process
    if let Some(current) = &scheduler.current {
        processes.push(ProcessInfo {
            pid: current.pid,
            ppid: current.ppid,
            state: current.state,
            name: current.name.clone(),
        });
    }

    // Ready queue
    for process in &scheduler.ready_queue {
        processes.push(ProcessInfo {
            pid: process.pid,
            ppid: process.ppid,
            state: process.state,
            name: process.name.clone(),
        });
    }

    // Blocked queue
    for process in &scheduler.blocked {
        processes.push(ProcessInfo {
            pid: process.pid,
            ppid: process.ppid,
            state: process.state,
            name: process.name.clone(),
        });
    }

    // Zombie queue
    for process in &scheduler.zombies {
        processes.push(ProcessInfo {
            pid: process.pid,
            ppid: process.ppid,
            state: process.state,
            name: process.name.clone(),
        });
    }

    processes
}
