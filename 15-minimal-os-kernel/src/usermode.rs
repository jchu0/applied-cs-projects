//! User mode support.
//!
//! This module provides functionality for transitioning between kernel mode
//! (Ring 0) and user mode (Ring 3).

use crate::gdt;
use crate::process::CpuContext;
use core::arch::asm;

/// User mode segment selectors.
pub mod selectors {
    /// User mode code segment selector.
    pub const USER_CODE_SELECTOR: u16 = (4 << 3) | 3;  // GDT entry 4, RPL 3
    /// User mode data segment selector.
    pub const USER_DATA_SELECTOR: u16 = (5 << 3) | 3;  // GDT entry 5, RPL 3
    /// Kernel mode code segment selector.
    pub const KERNEL_CODE_SELECTOR: u16 = (1 << 3) | 0;  // GDT entry 1, RPL 0
    /// Kernel mode data segment selector.
    pub const KERNEL_DATA_SELECTOR: u16 = (2 << 3) | 0;  // GDT entry 2, RPL 0
}

/// Jump to user mode.
///
/// This function sets up the stack for an iretq instruction to return to
/// user mode.
///
/// # Safety
/// - `entry_point` must be a valid user-space code address.
/// - `user_stack` must be a valid user-space stack address.
/// - The user-space memory must be properly mapped with user-accessible permissions.
#[inline(never)]
pub unsafe fn jump_to_usermode(entry_point: u64, user_stack: u64) {
    // Prepare stack frame for iretq:
    // SS, RSP, RFLAGS, CS, RIP (pushed in reverse order)
    asm!(
        // Disable interrupts during mode switch
        "cli",
        // Load user data segment into DS, ES, FS, GS
        "mov ax, {user_data:x}",
        "mov ds, ax",
        "mov es, ax",
        "mov fs, ax",
        "mov gs, ax",
        // Push SS (user data segment)
        "push {user_data}",
        // Push user stack pointer
        "push {user_stack}",
        // Push RFLAGS with interrupts enabled (bit 9 = IF)
        "pushfq",
        "pop rax",
        "or rax, 0x200",  // Set IF flag
        "push rax",
        // Push CS (user code segment)
        "push {user_code}",
        // Push entry point (RIP)
        "push {entry}",
        // Return to user mode
        "iretq",
        user_data = in(reg) selectors::USER_DATA_SELECTOR as u64,
        user_code = in(reg) selectors::USER_CODE_SELECTOR as u64,
        user_stack = in(reg) user_stack,
        entry = in(reg) entry_point,
        options(noreturn)
    );
}

/// Return to user mode from kernel after a system call.
///
/// Uses sysretq instruction for fast return to user mode.
///
/// # Safety
/// - The return address and stack must be valid user-space addresses.
/// - R11 will be restored to RFLAGS and RCX to RIP.
#[inline(never)]
pub unsafe fn sysret_to_usermode(rip: u64, rsp: u64, rax: u64) {
    asm!(
        // Load user data segment
        "mov ax, {user_data:x}",
        "mov ds, ax",
        "mov es, ax",
        "mov fs, ax",
        "mov gs, ax",
        // Set up return values
        // RCX = return address (RIP)
        "mov rcx, {rip}",
        // R11 = RFLAGS (with interrupts enabled)
        "mov r11, 0x202",
        // RSP = user stack
        "mov rsp, {rsp}",
        // RAX = system call return value
        "mov rax, {rax}",
        // Return to user mode
        "sysretq",
        user_data = in(reg) selectors::USER_DATA_SELECTOR as u64,
        rip = in(reg) rip,
        rsp = in(reg) rsp,
        rax = in(reg) rax,
        options(noreturn)
    );
}

/// Enter kernel mode from user mode (syscall entry).
///
/// This structure holds the saved user context when entering kernel mode.
#[repr(C)]
pub struct SyscallFrame {
    /// User RIP (saved in RCX by syscall).
    pub rip: u64,
    /// User RFLAGS (saved in R11 by syscall).
    pub rflags: u64,
    /// User RSP (must be saved manually).
    pub rsp: u64,
    /// System call number (in RAX).
    pub syscall_num: u64,
    /// Argument 1 (RDI).
    pub arg1: u64,
    /// Argument 2 (RSI).
    pub arg2: u64,
    /// Argument 3 (RDX).
    pub arg3: u64,
    /// Argument 4 (R10).
    pub arg4: u64,
    /// Argument 5 (R8).
    pub arg5: u64,
    /// Argument 6 (R9).
    pub arg6: u64,
}

impl SyscallFrame {
    /// Create a new syscall frame.
    pub fn new() -> Self {
        Self {
            rip: 0,
            rflags: 0,
            rsp: 0,
            syscall_num: 0,
            arg1: 0,
            arg2: 0,
            arg3: 0,
            arg4: 0,
            arg5: 0,
            arg6: 0,
        }
    }
}

impl Default for SyscallFrame {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate a user-space pointer.
///
/// Returns true if the pointer is in the valid user address range.
pub fn validate_user_ptr(ptr: u64, len: usize) -> bool {
    // User space is typically below 0x0000_7FFF_FFFF_FFFF on x86_64
    const USER_SPACE_END: u64 = 0x0000_7FFF_FFFF_FFFF;

    if ptr == 0 {
        return false;
    }

    let end = ptr.saturating_add(len as u64);
    end <= USER_SPACE_END
}

/// Copy data from user space to kernel space.
///
/// # Safety
/// The source pointer must be valid user-space memory.
pub unsafe fn copy_from_user(dest: *mut u8, src: *const u8, len: usize) -> Result<(), ()> {
    if !validate_user_ptr(src as u64, len) {
        return Err(());
    }

    core::ptr::copy_nonoverlapping(src, dest, len);
    Ok(())
}

/// Copy data from kernel space to user space.
///
/// # Safety
/// The destination pointer must be valid user-space memory.
pub unsafe fn copy_to_user(dest: *mut u8, src: *const u8, len: usize) -> Result<(), ()> {
    if !validate_user_ptr(dest as u64, len) {
        return Err(());
    }

    core::ptr::copy_nonoverlapping(src, dest, len);
    Ok(())
}

/// Read a null-terminated string from user space.
///
/// # Safety
/// The pointer must be valid user-space memory containing a null-terminated string.
pub unsafe fn read_user_string(ptr: u64, max_len: usize) -> Result<alloc::string::String, ()> {
    use alloc::string::String;

    if !validate_user_ptr(ptr, 1) {
        return Err(());
    }

    let mut string = String::new();
    let mut current = ptr as *const u8;

    for _ in 0..max_len {
        if !validate_user_ptr(current as u64, 1) {
            return Err(());
        }

        let byte = *current;
        if byte == 0 {
            return Ok(string);
        }
        string.push(byte as char);
        current = current.add(1);
    }

    Err(())  // String too long
}

/// Initialize user mode support.
///
/// Sets up the GDT entries for user mode segments and configures
/// the syscall/sysret instructions.
pub fn init() {
    // Enable syscall/sysret instructions
    unsafe {
        use x86_64::registers::model_specific::{Efer, EferFlags, Star, LStar, SFMask};
        use x86_64::registers::rflags::RFlags;
        use x86_64::VirtAddr;

        // Enable SCE (System Call Extensions) bit in EFER
        let efer = Efer::read();
        Efer::write(efer | EferFlags::SYSTEM_CALL_EXTENSIONS);

        // Set STAR register with segment selectors
        // SYSRET will use (STAR[63:48] + 16) for CS and (STAR[63:48] + 8) for SS
        // SYSCALL will use STAR[47:32] for CS and (STAR[47:32] + 8) for SS
        Star::write(
            x86_64::structures::gdt::SegmentSelector(selectors::USER_CODE_SELECTOR - 16),
            x86_64::structures::gdt::SegmentSelector(selectors::KERNEL_DATA_SELECTOR),
            x86_64::structures::gdt::SegmentSelector(selectors::KERNEL_CODE_SELECTOR),
            x86_64::structures::gdt::SegmentSelector(selectors::KERNEL_DATA_SELECTOR),
        ).unwrap();

        // Set LSTAR register to syscall entry point
        extern "C" {
            fn syscall_entry();
        }
        // Note: syscall_entry would need to be implemented in assembly
        // LStar::write(VirtAddr::new(syscall_entry as u64));

        // Set SFMASK to clear IF on syscall
        SFMask::write(RFlags::INTERRUPT_FLAG);
    }

    crate::serial_println!("User mode support initialized");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_user_ptr_null() {
        assert!(!validate_user_ptr(0, 10));
    }

    #[test]
    fn test_validate_user_ptr_valid() {
        assert!(validate_user_ptr(0x1000, 0x1000));
    }

    #[test]
    fn test_validate_user_ptr_kernel_space() {
        // Kernel space addresses should be invalid
        assert!(!validate_user_ptr(0xFFFF_8000_0000_0000, 10));
    }

    #[test]
    fn test_syscall_frame_default() {
        let frame = SyscallFrame::default();
        assert_eq!(frame.syscall_num, 0);
        assert_eq!(frame.rip, 0);
    }
}
