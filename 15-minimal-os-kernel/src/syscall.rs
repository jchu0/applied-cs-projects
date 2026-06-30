//! System call interface.

use crate::elf;
use crate::process::{Fd, Pid};
use crate::scheduler;
use crate::vfs;
use alloc::string::String;
use alloc::vec::Vec;

/// System call numbers.
#[repr(u64)]
#[derive(Debug, Clone, Copy)]
pub enum SyscallNumber {
    Read = 0,
    Write = 1,
    Open = 2,
    Close = 3,
    Stat = 4,
    Fstat = 5,
    Lseek = 6,
    Mmap = 9,
    Munmap = 11,
    Brk = 12,
    Ioctl = 16,
    Dup = 32,
    Dup2 = 33,
    Getpid = 39,
    Fork = 57,
    Execve = 59,
    Exit = 60,
    Wait4 = 61,
    Kill = 62,
    Getcwd = 79,
    Chdir = 80,
    Getuid = 102,
    Setuid = 105,
    Getppid = 110,
}

/// System call error codes.
#[repr(i64)]
#[derive(Debug, Clone, Copy)]
pub enum SyscallError {
    /// Operation not permitted.
    EPERM = 1,
    /// No such file or directory.
    ENOENT = 2,
    /// No such process.
    ESRCH = 3,
    /// Interrupted system call.
    EINTR = 4,
    /// I/O error.
    EIO = 5,
    /// Exec format error.
    ENOEXEC = 8,
    /// Bad file descriptor.
    EBADF = 9,
    /// No child processes.
    ECHILD = 10,
    /// Try again.
    EAGAIN = 11,
    /// Out of memory.
    ENOMEM = 12,
    /// Permission denied.
    EACCES = 13,
    /// Bad address.
    EFAULT = 14,
    /// Device or resource busy.
    EBUSY = 16,
    /// File exists.
    EEXIST = 17,
    /// Is a directory.
    EISDIR = 21,
    /// Invalid argument.
    EINVAL = 22,
    /// Too many open files in system.
    ENFILE = 23,
    /// Too many open files.
    EMFILE = 24,
    /// Text file busy.
    ETXTBSY = 26,
    /// No space left on device.
    ENOSPC = 28,
    /// Illegal seek.
    ESPIPE = 29,
    /// Read-only file system.
    EROFS = 30,
    /// Function not implemented.
    ENOSYS = 38,
    /// Directory not empty.
    ENOTEMPTY = 39,
}

/// System call result.
pub type SyscallResult = Result<i64, SyscallError>;

/// Handle a system call.
pub fn syscall_handler(
    num: u64,
    arg1: u64,
    arg2: u64,
    arg3: u64,
    _arg4: u64,
    _arg5: u64,
    _arg6: u64,
) -> i64 {
    let result = match num {
        0 => sys_read(arg1 as Fd, arg2, arg3 as usize),
        1 => sys_write(arg1 as Fd, arg2, arg3 as usize),
        2 => sys_open(arg1, arg2 as u32, arg3 as u32),
        3 => sys_close(arg1 as Fd),
        39 => sys_getpid(),
        57 => sys_fork(),
        59 => sys_execve(arg1, arg2, arg3),
        60 => sys_exit(arg1 as i32),
        61 => sys_wait4(arg1 as i32),
        62 => sys_kill(arg1 as Pid, arg2 as i32),
        110 => sys_getppid(),
        _ => Err(SyscallError::ENOSYS),
    };

    match result {
        Ok(val) => val,
        Err(err) => -(err as i64),
    }
}

/// Read from file descriptor.
fn sys_read(fd: Fd, buf_ptr: u64, count: usize) -> SyscallResult {
    // Validate pointer
    if buf_ptr == 0 {
        return Err(SyscallError::EFAULT);
    }

    // Get file descriptor info
    // In a real kernel, we would read from the file
    match fd {
        0 => {
            // stdin - not implemented
            Ok(0)
        }
        _ => {
            // Try to read from VFS
            let _path = get_fd_path(fd)?;
            // Would read from VFS here
            Ok(count as i64)
        }
    }
}

/// Write to file descriptor.
fn sys_write(fd: Fd, buf_ptr: u64, count: usize) -> SyscallResult {
    // Validate pointer
    if buf_ptr == 0 {
        return Err(SyscallError::EFAULT);
    }

    // Get file descriptor info
    match fd {
        1 | 2 => {
            // stdout/stderr - write to serial
            let slice = unsafe {
                core::slice::from_raw_parts(buf_ptr as *const u8, count)
            };
            for &byte in slice {
                crate::serial_print!("{}", byte as char);
            }
            Ok(count as i64)
        }
        _ => {
            // Try to write to VFS
            let _path = get_fd_path(fd)?;
            // Would write to VFS here
            Ok(count as i64)
        }
    }
}

/// Open a file.
fn sys_open(path_ptr: u64, flags: u32, _mode: u32) -> SyscallResult {
    if path_ptr == 0 {
        return Err(SyscallError::EFAULT);
    }

    // Read path string
    let path = unsafe { read_user_string(path_ptr) }?;

    // Check if file exists
    if !vfs::exists(&path) {
        // Check for O_CREAT flag
        if flags & 0x40 != 0 {
            // Create file
            vfs::create_file(&path)?;
        } else {
            return Err(SyscallError::ENOENT);
        }
    }

    // Allocate file descriptor
    // In a real kernel, we would track this in the process
    Ok(3) // Return a dummy fd
}

/// Close a file descriptor.
fn sys_close(fd: Fd) -> SyscallResult {
    if fd < 0 {
        return Err(SyscallError::EBADF);
    }

    // In a real kernel, we would close the fd in the process
    Ok(0)
}

/// Get current process ID.
fn sys_getpid() -> SyscallResult {
    scheduler::current_pid()
        .map(|pid| pid as i64)
        .ok_or(SyscallError::ESRCH)
}

/// Get parent process ID.
fn sys_getppid() -> SyscallResult {
    // In a real kernel, we would get this from the process
    Ok(1) // Return init's PID
}

/// Fork current process.
fn sys_fork() -> SyscallResult {
    // Get the current process and fork it
    let child = scheduler::fork_current()?;
    let child_pid = child.pid;

    // The child process context has rax = 0 (fork returns 0 to child)
    // Add child to scheduler
    scheduler::add_process(child);

    // Return child PID to parent
    Ok(child_pid as i64)
}

/// Execute a new program.
fn sys_execve(path_ptr: u64, argv_ptr: u64, envp_ptr: u64) -> SyscallResult {
    if path_ptr == 0 {
        return Err(SyscallError::EFAULT);
    }

    // Read path string
    let path = unsafe { read_user_string(path_ptr) }?;

    // Read argv array (null-terminated array of null-terminated strings)
    let argv = unsafe { read_user_string_array(argv_ptr) }.unwrap_or_else(|_| Vec::new());

    // Read envp array (null-terminated array of null-terminated strings)
    let envp = unsafe { read_user_string_array(envp_ptr) }.unwrap_or_else(|_| Vec::new());

    // Read the executable from VFS
    let elf_data = vfs::read_file(&path).map_err(|_| SyscallError::ENOENT)?;

    // Load the ELF
    let loaded = elf::load_elf(&elf_data).map_err(|_| SyscallError::ENOENT)?;

    // Set up the process for execution
    scheduler::exec_current(loaded.entry, loaded.stack_top, &argv, &envp)?;

    // execve doesn't return on success (it "returns" to the new program)
    // but we return 0 here as the context switch will happen via scheduler
    Ok(0)
}

/// Read a null-terminated array of null-terminated strings from user space.
unsafe fn read_user_string_array(ptr: u64) -> Result<Vec<String>, SyscallError> {
    if ptr == 0 {
        return Ok(Vec::new());
    }

    let mut result = Vec::new();
    let mut current = ptr as *const u64;

    loop {
        let string_ptr = *current;
        if string_ptr == 0 {
            break;
        }

        let s = read_user_string(string_ptr)?;
        result.push(s);
        current = current.add(1);

        // Limit number of arguments
        if result.len() > 1024 {
            return Err(SyscallError::EINVAL);
        }
    }

    Ok(result)
}

/// Exit current process.
fn sys_exit(code: i32) -> SyscallResult {
    scheduler::exit(code);
    // Never returns
    Ok(0)
}

/// Wait for child process.
fn sys_wait4(pid: i32) -> SyscallResult {
    if pid <= 0 {
        // Wait for any child
        // Not fully implemented
        return Err(SyscallError::ENOSYS);
    }

    match scheduler::wait(pid as Pid) {
        Some(exit_code) => Ok(exit_code as i64),
        None => Err(SyscallError::ESRCH),
    }
}

/// Send signal to process.
fn sys_kill(pid: Pid, signal: i32) -> SyscallResult {
    if signal < 0 || signal > 31 {
        return Err(SyscallError::EINVAL);
    }

    // In a real kernel, we would send the signal to the process
    // For now, just return success
    let _ = pid;
    Ok(0)
}

/// Get file path for a file descriptor.
fn get_fd_path(_fd: Fd) -> Result<alloc::string::String, SyscallError> {
    // In a real kernel, we would look this up in the process
    Err(SyscallError::EBADF)
}

/// Read a null-terminated string from user space.
unsafe fn read_user_string(ptr: u64) -> Result<alloc::string::String, SyscallError> {
    use alloc::string::String;

    let mut string = String::new();
    let mut current = ptr as *const u8;

    loop {
        let byte = *current;
        if byte == 0 {
            break;
        }
        string.push(byte as char);
        current = current.add(1);

        // Limit string length
        if string.len() > 4096 {
            return Err(SyscallError::EINVAL);
        }
    }

    Ok(string)
}
