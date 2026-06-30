# Minimal OS Kernel

A minimal x86_64 operating system kernel in Rust demonstrating the full
boot-to-userspace path: GDT/IDT setup, 4-level paging, frame allocation,
round-robin scheduling, system calls, VFS with a RAM filesystem, and a
simple interactive shell.

> **Status:** reference implementation / teaching scaffold built to a strong
> blueprint — not production-grade. See
> [../PROJECTS_STATUS.md](../PROJECTS_STATUS.md) and the
> [2026-06 audit](../../docs/AUDIT_2026-06_public-readiness.md).

> **Concepts covered:** §01 `rust/04-unsafe-rust`, `cpp/01-modern-cpp`

---

## What's real vs simulated

The kernel logic (scheduler, VFS, syscall dispatch, ELF loader, context
switch, signal delivery) is fully implemented. `cargo bootimage` produces a
bootable disk image, but **QEMU boot has not been verified** — the
init→scheduler→shell path has not been confirmed to run end-to-end. There
are no automated integration tests; the `isa-debug-exit` harness is
configured but not yet exercised.

---

## Layout

```
src/
  main.rs        — bootloader entry point (entry_point! macro, kernel_main)
  lib.rs         — crate root
  gdt.rs         — Global Descriptor Table and TSS
  interrupts.rs  — IDT, CPU exceptions, hardware IRQ handlers
  memory/        — frame allocator, 4-level paging, heap
  process.rs     — PCB, fork, exec, exit, signals
  scheduler.rs   — round-robin + priority scheduler, context switch (asm)
  context.rs     — CpuContext save/restore
  syscall.rs     — syscall dispatch table
  vfs.rs         — VFS trait + RAM filesystem
  elf.rs         — ELF64 loader
  shell.rs       — interactive kernel shell
  serial.rs      — UART serial output

BLUEPRINT.md       — full architecture, data structures, boot sequence
rust-toolchain.toml — pins nightly for bare-metal build
```

---

## Build and test

A nightly Rust toolchain and `cargo-bootimage` are required.

```bash
# Install tooling (once)
rustup component add rust-src llvm-tools-preview
cargo install bootimage

# Build
cd 06-real-world-projects/15-minimal-os-kernel
cargo build

# Build bootable disk image
cargo bootimage

# Run library unit tests (host target, no QEMU needed)
cargo test --lib

# Run under QEMU (requires qemu-system-x86_64)
cargo run
```

> **Note:** there is a known build-target mismatch between `bootloader = "0.9"`
> (custom target JSON) and the `.cargo/config.toml` `x86_64-unknown-none`
> target. See the comment in `Cargo.toml` for the two resolution paths.
