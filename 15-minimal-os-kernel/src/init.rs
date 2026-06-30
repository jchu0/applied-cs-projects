//! Init process - PID 1, the first user-space process.
//!
//! The init process is responsible for:
//! - Being the first user-space process (PID 1)
//! - Spawning the shell
//! - Reaping zombie processes
//! - Handling orphaned processes

use alloc::string::String;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::process::{Pid, Process, ProcessState};
use crate::scheduler;
use crate::vfs;
use crate::serial_println;
use crate::shell;

/// Init process state.
static INIT_RUNNING: AtomicBool = AtomicBool::new(false);

/// Init process PID.
pub const INIT_PID: Pid = 1;

/// Start the init process.
///
/// This creates PID 1 and begins the user-space initialization.
pub fn start() {
    if INIT_RUNNING.swap(true, Ordering::SeqCst) {
        serial_println!("Init already running!");
        return;
    }

    serial_println!("Starting init process...");

    // Create init process
    let init_process = create_init_process();

    // Add to scheduler
    scheduler::add_process(init_process);

    serial_println!("Init process created (PID 1)");
}

/// Create the init process.
fn create_init_process() -> alloc::boxed::Box<Process> {
    let mut process = Process::new(String::from("init"));
    process.state = ProcessState::Ready;

    // Set up init's working directory
    process.cwd = String::from("/");

    // Init runs as root
    process.uid = 0;
    process.gid = 0;
    process.euid = 0;
    process.egid = 0;

    alloc::boxed::Box::new(process)
}

/// Init main loop - runs in kernel mode to spawn shell.
pub fn init_main() {
    serial_println!("[init] Starting...");

    // Mount root filesystem if not already mounted
    init_filesystems();

    // Create /dev entries
    init_devices();

    // Spawn shell
    spawn_shell();

    // Main loop: reap zombies
    loop {
        reap_zombies();

        // Yield to other processes
        scheduler::yield_cpu();
    }
}

/// Initialize filesystems.
fn init_filesystems() {
    serial_println!("[init] Initializing filesystems...");

    // Create essential directories
    let dirs = ["/bin", "/dev", "/etc", "/home", "/proc", "/tmp", "/var"];

    for dir in dirs.iter() {
        if let Err(e) = vfs::mkdir(dir, 0o755) {
            // Ignore "already exists" errors
            serial_println!("[init] mkdir {}: {:?}", dir, e);
        }
    }

    // Create /etc/passwd
    create_passwd_file();

    serial_println!("[init] Filesystems initialized");
}

/// Create /etc/passwd file.
fn create_passwd_file() {
    let passwd = "root:x:0:0:root:/root:/bin/sh\n";

    if let Ok(fd) = vfs::open("/etc/passwd", vfs::O_CREAT | vfs::O_WRONLY, 0o644) {
        let _ = vfs::write_fd(fd, passwd.as_bytes());
        let _ = vfs::close(fd);
    }
}

/// Initialize device nodes.
fn init_devices() {
    serial_println!("[init] Creating device nodes...");

    // Create /dev/null
    if let Err(e) = vfs::mknod("/dev/null", vfs::S_IFCHR | 0o666, 1, 3) {
        serial_println!("[init] mknod /dev/null: {:?}", e);
    }

    // Create /dev/zero
    if let Err(e) = vfs::mknod("/dev/zero", vfs::S_IFCHR | 0o666, 1, 5) {
        serial_println!("[init] mknod /dev/zero: {:?}", e);
    }

    // Create /dev/console
    if let Err(e) = vfs::mknod("/dev/console", vfs::S_IFCHR | 0o620, 5, 1) {
        serial_println!("[init] mknod /dev/console: {:?}", e);
    }

    // Create /dev/tty
    if let Err(e) = vfs::mknod("/dev/tty", vfs::S_IFCHR | 0o666, 5, 0) {
        serial_println!("[init] mknod /dev/tty: {:?}", e);
    }

    serial_println!("[init] Device nodes created");
}

/// Spawn the shell process.
fn spawn_shell() {
    serial_println!("[init] Spawning shell...");

    // Fork a new process
    let shell_pid = match scheduler::fork() {
        Ok(pid) => pid,
        Err(e) => {
            serial_println!("[init] Failed to fork: {:?}", e);
            return;
        }
    };

    if shell_pid == 0 {
        // Child process - exec shell
        shell::shell_main();
        // Should not return, but if it does, exit
        scheduler::exit(0);
    } else {
        serial_println!("[init] Shell spawned with PID {}", shell_pid);
    }
}

/// Reap zombie child processes.
fn reap_zombies() {
    // Wait for any child with WNOHANG
    loop {
        match scheduler::waitpid_any(WaitFlags::WNOHANG) {
            Ok((pid, _status)) if pid > 0 => {
                serial_println!("[init] Reaped zombie process {}", pid);
            }
            _ => break,
        }
    }
}

/// Wait flags for waitpid.
pub struct WaitFlags;

impl WaitFlags {
    pub const WNOHANG: i32 = 1;
    pub const WUNTRACED: i32 = 2;
    pub const WCONTINUED: i32 = 8;
}

/// Signal handler for init.
///
/// Init (PID 1) has special signal handling:
/// - SIGCHLD: Reap zombie children
/// - Most signals are ignored (init can't be killed)
pub fn handle_signal(signum: u32) {
    match signum {
        17 => { // SIGCHLD
            reap_zombies();
        }
        _ => {
            serial_println!("[init] Ignoring signal {}", signum);
        }
    }
}

/// Adopt orphaned processes.
///
/// When a parent process exits, its children become orphans
/// and are re-parented to init (PID 1).
pub fn adopt_orphans(parent_pid: Pid) {
    scheduler::reparent_children(parent_pid, INIT_PID);
}

/// Check if init is running.
pub fn is_running() -> bool {
    INIT_RUNNING.load(Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_pid() {
        assert_eq!(INIT_PID, 1);
    }

    #[test]
    fn test_wait_flags() {
        assert_eq!(WaitFlags::WNOHANG, 1);
        assert_eq!(WaitFlags::WUNTRACED, 2);
    }
}
