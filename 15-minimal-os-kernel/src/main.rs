//! Bootable binary entry point for the minimal OS kernel.
//!
//! The kernel logic lives in the library crate (`minimal_os_kernel`); this
//! binary exists only to be linked with the `bootloader` crate into a bootable
//! disk image. `bootloader::entry_point!` generates the real `_start` symbol
//! that the bootloader jumps to after switching the CPU into 64-bit long mode,
//! and hands us a validated `&'static BootInfo`.
//!
//! Build/run (requires the nightly toolchain pinned in `rust-toolchain.toml`,
//! plus the `bootimage` tool and QEMU):
//!
//! ```bash
//! cargo install bootimage
//! cargo bootimage            # produces target/.../bootimage-minimal-os-kernel.bin
//! cargo run                  # builds the image and boots it in QEMU (see .cargo/config.toml)
//! ```
//!
//! The panic handler, global allocator, and alloc-error handler are all defined
//! in the library crate and linked in here — this file deliberately stays tiny.

#![no_std]
#![no_main]

use bootloader::{entry_point, BootInfo};

// Register the kernel's entry point. The macro type-checks the signature
// (`fn(&'static BootInfo) -> !`) and emits the `_start` the bootloader calls.
entry_point!(kernel_entry);

/// Thin wrapper that hands control to the library's `kernel_main`.
fn kernel_entry(boot_info: &'static BootInfo) -> ! {
    minimal_os_kernel::kernel_main(boot_info)
}
