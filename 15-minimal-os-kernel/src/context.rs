//! Context switching implementation.
//!
//! This module provides the low-level context switching functionality
//! needed to switch between processes.

use crate::process::CpuContext;
use core::arch::asm;

/// Save current CPU context.
///
/// # Safety
/// This function manipulates CPU registers directly.
#[inline(never)]
#[naked]
pub unsafe extern "C" fn save_context(ctx: *mut CpuContext) {
    asm!(
        // Save general purpose registers
        "mov [rdi + 0x00], rax",
        "mov [rdi + 0x08], rbx",
        "mov [rdi + 0x10], rcx",
        "mov [rdi + 0x18], rdx",
        "mov [rdi + 0x20], rsi",
        "mov [rdi + 0x28], rdi",
        "mov [rdi + 0x30], rbp",
        // Save stack pointer
        "mov rax, rsp",
        "add rax, 8",  // Adjust for return address
        "mov [rdi + 0x38], rax",
        // Save r8-r15
        "mov [rdi + 0x40], r8",
        "mov [rdi + 0x48], r9",
        "mov [rdi + 0x50], r10",
        "mov [rdi + 0x58], r11",
        "mov [rdi + 0x60], r12",
        "mov [rdi + 0x68], r13",
        "mov [rdi + 0x70], r14",
        "mov [rdi + 0x78], r15",
        // Save instruction pointer (return address)
        "mov rax, [rsp]",
        "mov [rdi + 0x80], rax",
        // Save flags
        "pushfq",
        "pop rax",
        "mov [rdi + 0x88], rax",
        // Save segment selectors
        "xor rax, rax",
        "mov ax, cs",
        "mov [rdi + 0x90], rax",
        "mov ax, ss",
        "mov [rdi + 0x98], rax",
        "ret",
        options(noreturn)
    )
}

/// Restore CPU context.
///
/// # Safety
/// This function manipulates CPU registers directly and will not return
/// to the caller but will instead jump to the restored context.
#[inline(never)]
#[naked]
pub unsafe extern "C" fn restore_context(ctx: *const CpuContext) {
    asm!(
        // Restore general purpose registers except rsp
        "mov rax, [rdi + 0x00]",
        "mov rbx, [rdi + 0x08]",
        "mov rcx, [rdi + 0x10]",
        "mov rdx, [rdi + 0x18]",
        "mov rsi, [rdi + 0x20]",
        "mov rbp, [rdi + 0x30]",
        "mov r8, [rdi + 0x40]",
        "mov r9, [rdi + 0x48]",
        "mov r10, [rdi + 0x50]",
        "mov r11, [rdi + 0x58]",
        "mov r12, [rdi + 0x60]",
        "mov r13, [rdi + 0x68]",
        "mov r14, [rdi + 0x70]",
        "mov r15, [rdi + 0x78]",
        // Restore flags
        "push [rdi + 0x88]",
        "popfq",
        // Push return address and new stack pointer
        "mov rsp, [rdi + 0x38]",
        "push [rdi + 0x80]",  // Push RIP
        // Restore rdi last
        "mov rdi, [rdi + 0x28]",
        // Return to new context
        "ret",
        options(noreturn)
    )
}

/// Perform a context switch from one process to another.
///
/// Saves the current context to `from` and restores the context from `to`.
///
/// # Safety
/// This function manipulates CPU registers directly.
pub unsafe fn switch_context(from: *mut CpuContext, to: *const CpuContext) {
    // Save current context
    save_context(from);

    // Restore target context
    restore_context(to);
}

/// Context for kernel entry/exit.
#[repr(C)]
pub struct KernelEntryContext {
    /// Saved user registers.
    pub user_ctx: CpuContext,
    /// Kernel stack pointer.
    pub kernel_rsp: u64,
    /// User stack pointer.
    pub user_rsp: u64,
}

impl KernelEntryContext {
    /// Create a new kernel entry context.
    pub fn new() -> Self {
        Self {
            user_ctx: CpuContext::default(),
            kernel_rsp: 0,
            user_rsp: 0,
        }
    }
}

impl Default for KernelEntryContext {
    fn default() -> Self {
        Self::new()
    }
}

/// TSS (Task State Segment) helper functions.
pub mod tss {
    use x86_64::VirtAddr;

    /// Set the kernel stack pointer in the TSS.
    ///
    /// This is called during context switch to set up the kernel stack
    /// for the next process.
    pub fn set_kernel_stack(stack_ptr: VirtAddr) {
        use crate::gdt::TSS;

        unsafe {
            TSS.lock().privilege_stack_table[0] = stack_ptr;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_context_size() {
        // Ensure CpuContext has expected size
        assert_eq!(core::mem::size_of::<CpuContext>(), 160);
    }

    #[test]
    fn test_kernel_entry_context() {
        let ctx = KernelEntryContext::new();
        assert_eq!(ctx.kernel_rsp, 0);
        assert_eq!(ctx.user_rsp, 0);
    }
}
