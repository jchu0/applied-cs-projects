//! Minimal OS Kernel for x86_64.
//!
//! A minimal but functional operating system kernel demonstrating core OS concepts:
//! boot process, memory management with paging, system calls, process scheduling,
//! and filesystem operations.

#![no_std]
#![feature(abi_x86_interrupt)]
#![feature(alloc_error_handler)]
#![feature(naked_functions)]

extern crate alloc;

pub mod memory;
pub mod interrupts;
pub mod gdt;
pub mod process;
pub mod scheduler;
pub mod syscall;
pub mod vfs;
pub mod serial;
pub mod context;
pub mod usermode;
pub mod elf;
pub mod init;
pub mod shell;

use bootloader::BootInfo;
use core::panic::PanicInfo;
use x86_64::VirtAddr;

/// Kernel entry point.
pub fn kernel_main(boot_info: &'static BootInfo) -> ! {
    // Initialize serial output for debugging
    serial::init();
    serial_println!("Kernel starting...");

    // Initialize GDT and TSS
    gdt::init();
    serial_println!("GDT initialized");

    // Initialize IDT
    interrupts::init_idt();
    serial_println!("IDT initialized");

    // Initialize memory management
    let phys_mem_offset = VirtAddr::new(boot_info.physical_memory_offset);
    let mut mapper = unsafe { memory::init(phys_mem_offset) };
    let mut frame_allocator = unsafe {
        memory::BootInfoFrameAllocator::init(&boot_info.memory_map)
    };
    serial_println!("Memory initialized");

    // Initialize heap
    memory::allocator::init_heap(&mut mapper, &mut frame_allocator)
        .expect("heap initialization failed");
    serial_println!("Heap initialized");

    // Initialize interrupts
    unsafe { interrupts::PICS.lock().initialize() };
    x86_64::instructions::interrupts::enable();
    serial_println!("Interrupts enabled");

    // Initialize scheduler
    scheduler::init();
    serial_println!("Scheduler initialized");

    // Initialize VFS
    vfs::init();
    serial_println!("VFS initialized");

    serial_println!("Kernel initialization complete!");

    // Start init process (PID 1)
    init::start();
    serial_println!("Init process started");

    // Run scheduler - this will switch to init
    scheduler::run();

    // Enter idle loop (should never reach here)
    hlt_loop();
}

/// Halt loop for idle.
pub fn hlt_loop() -> ! {
    loop {
        x86_64::instructions::hlt();
    }
}

/// Panic handler.
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial_println!("KERNEL PANIC: {}", info);
    hlt_loop();
}

/// Allocation error handler.
#[alloc_error_handler]
fn alloc_error_handler(layout: alloc::alloc::Layout) -> ! {
    panic!("allocation error: {:?}", layout);
}
