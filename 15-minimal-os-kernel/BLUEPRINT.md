# Minimal Operating System Kernel

> **Concepts covered:** §01 software-engineering — `rust/04-unsafe-rust`, `cpp/01-modern-cpp`

## Executive Summary

A minimal but functional operating system kernel demonstrating core OS concepts: boot process, memory management with paging, system calls, process scheduling, and filesystem operations. Targets x86_64 architecture with both BIOS and UEFI boot support. Provides a foundation for understanding how operating systems manage hardware resources and provide abstractions to user programs.

---

## System Architecture

```
                        User Space
    +--------------------------------------------------+
    |  Process A    |  Process B    |  Shell/Init      |
    |  (Ring 3)     |  (Ring 3)     |  (Ring 3)        |
    +-------+---------------+---------------+----------+
            |               |               |
            v               v               v
    +--------------------------------------------------+
    |              System Call Interface               |
    |    (syscall instruction / int 0x80)              |
    +--------------------------------------------------+
                            |
                            v
                      Kernel Space
    +--------------------------------------------------+
    |                                                  |
    |  +-----------+  +-----------+  +-------------+   |
    |  |  Process  |  |  Memory   |  |   File      |   |
    |  |  Manager  |  |  Manager  |  |   System    |   |
    |  +-----------+  +-----------+  +-------------+   |
    |                                                  |
    |  +-----------+  +-----------+  +-------------+   |
    |  | Scheduler |  |    IPC    |  |   Device    |   |
    |  |           |  |           |  |   Drivers   |   |
    |  +-----------+  +-----------+  +-------------+   |
    |                                                  |
    +--------------------------------------------------+
                            |
                            v
    +--------------------------------------------------+
    |           Hardware Abstraction Layer             |
    +--------------------------------------------------+
                            |
                            v
    +--------------------------------------------------+
    |                   Hardware                       |
    |  CPU  |  RAM  |  Disk  |  Timer  |  Keyboard    |
    +--------------------------------------------------+

Memory Layout (x86_64):
+------------------+ 0xFFFFFFFF_FFFFFFFF (256 TB)
|   Kernel Space   |
|   (Higher Half)  |
+------------------+ 0xFFFF8000_00000000
|                  |
|   Unmapped       |
|   (Hole)         |
|                  |
+------------------+ 0x00007FFF_FFFFFFFF
|   User Space     |
|   Stack          |
+------------------+
|                  |
|   User Heap      |
|                  |
+------------------+
|   User Code/Data |
+------------------+ 0x00000000_00400000
|   Reserved       |
+------------------+ 0x00000000_00000000
```

---

## Core Data Structures

### Process Control Block (PCB)

```rust
pub struct Process {
    pub pid: Pid,
    pub ppid: Pid,
    pub state: ProcessState,

    // CPU context
    pub context: CpuContext,

    // Memory
    pub page_table: PageTable,
    pub heap_start: VirtAddr,
    pub heap_end: VirtAddr,
    pub stack_top: VirtAddr,

    // Open files
    pub file_descriptors: BTreeMap<Fd, FileDescriptor>,

    // Credentials
    pub uid: Uid,
    pub gid: Gid,
    pub euid: Uid,
    pub egid: Gid,

    // Signals
    pub pending_signals: SignalSet,
    pub signal_handlers: [SignalHandler; 32],
    pub signal_mask: SignalSet,

    // Scheduling
    pub priority: i32,
    pub nice: i32,
    pub time_slice: u64,

    // Statistics
    pub user_time: u64,
    pub system_time: u64,
    pub start_time: u64,

    // Exit
    pub exit_code: Option<i32>,
}

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

    // Instruction pointer
    pub rip: u64,
    pub rflags: u64,

    // Segment selectors
    pub cs: u64,
    pub ss: u64,

    // Floating point state
    pub fxsave_area: [u8; 512],
}

pub enum ProcessState {
    Ready,
    Running,
    Blocked { reason: BlockReason },
    Zombie,
}

pub enum BlockReason {
    WaitingForIo(Fd),
    WaitingForChild(Option<Pid>),
    WaitingForSignal,
    Sleeping { until: u64 },
}
```

### Page Table Structures

```rust
// x86_64 4-level paging
pub struct PageTable {
    root: PhysAddr,  // CR3 value
    levels: [PageTableLevel; 4],
}

pub struct PageTableLevel {
    entries: &'static mut [PageTableEntry; 512],
}

#[repr(transparent)]
pub struct PageTableEntry(u64);

impl PageTableEntry {
    const PRESENT: u64 = 1 << 0;
    const WRITABLE: u64 = 1 << 1;
    const USER: u64 = 1 << 2;
    const WRITE_THROUGH: u64 = 1 << 3;
    const NO_CACHE: u64 = 1 << 4;
    const ACCESSED: u64 = 1 << 5;
    const DIRTY: u64 = 1 << 6;
    const HUGE_PAGE: u64 = 1 << 7;
    const GLOBAL: u64 = 1 << 8;
    const NO_EXECUTE: u64 = 1 << 63;

    pub fn addr(&self) -> PhysAddr {
        PhysAddr::new(self.0 & 0x000F_FFFF_FFFF_F000)
    }

    pub fn flags(&self) -> PageFlags {
        PageFlags::from_bits_truncate(self.0)
    }

    pub fn set(&mut self, addr: PhysAddr, flags: PageFlags) {
        self.0 = addr.as_u64() | flags.bits();
    }
}

pub struct VirtAddr(u64);

impl VirtAddr {
    pub fn page_table_indices(&self) -> [usize; 4] {
        [
            ((self.0 >> 39) & 0x1FF) as usize,  // PML4
            ((self.0 >> 30) & 0x1FF) as usize,  // PDPT
            ((self.0 >> 21) & 0x1FF) as usize,  // PD
            ((self.0 >> 12) & 0x1FF) as usize,  // PT
        ]
    }

    pub fn page_offset(&self) -> usize {
        (self.0 & 0xFFF) as usize
    }
}
```

### Frame Allocator

```rust
pub struct FrameAllocator {
    // Bitmap of all physical frames
    bitmap: &'static mut [u64],
    total_frames: usize,
    free_frames: usize,

    // Free list for fast allocation
    free_list: Option<PhysAddr>,
}

impl FrameAllocator {
    pub fn allocate(&mut self) -> Option<PhysFrame> {
        // Try free list first
        if let Some(frame) = self.free_list {
            // Read next pointer from frame
            let next = unsafe {
                *(frame.as_u64() as *const Option<PhysAddr>)
            };
            self.free_list = next;
            self.free_frames -= 1;
            return Some(PhysFrame::containing_address(frame));
        }

        // Fall back to bitmap scan
        for (i, word) in self.bitmap.iter_mut().enumerate() {
            if *word != u64::MAX {
                let bit = word.trailing_ones() as usize;
                *word |= 1 << bit;
                self.free_frames -= 1;
                let addr = PhysAddr::new(((i * 64 + bit) * 4096) as u64);
                return Some(PhysFrame::containing_address(addr));
            }
        }

        None
    }

    pub fn deallocate(&mut self, frame: PhysFrame) {
        // Add to free list
        let addr = frame.start_address();
        unsafe {
            *(addr.as_u64() as *mut Option<PhysAddr>) = self.free_list;
        }
        self.free_list = Some(addr);
        self.free_frames += 1;

        // Also clear bitmap
        let frame_num = addr.as_u64() as usize / 4096;
        let word = frame_num / 64;
        let bit = frame_num % 64;
        self.bitmap[word] &= !(1 << bit);
    }
}
```

---

## Boot Process

### BIOS Bootloader

```nasm
; Stage 1 bootloader (512 bytes, loaded at 0x7C00)
[BITS 16]
[ORG 0x7C00]

start:
    ; Set up segments
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00

    ; Save boot drive
    mov [boot_drive], dl

    ; Load stage 2 from disk
    mov ah, 0x02        ; BIOS read sectors
    mov al, 32          ; Number of sectors
    mov ch, 0           ; Cylinder
    mov dh, 0           ; Head
    mov cl, 2           ; Start sector
    mov bx, 0x7E00      ; Destination
    int 0x13

    ; Jump to stage 2
    jmp 0x7E00

boot_drive: db 0

times 510 - ($ - $$) db 0
dw 0xAA55

; Stage 2
[ORG 0x7E00]

stage2:
    ; Enable A20 line
    call enable_a20

    ; Load GDT
    lgdt [gdt_descriptor]

    ; Enter protected mode
    mov eax, cr0
    or eax, 1
    mov cr0, eax

    ; Far jump to protected mode
    jmp 0x08:protected_mode

[BITS 32]
protected_mode:
    ; Set up segments
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    ; Set up paging for long mode
    call setup_paging

    ; Enable long mode
    mov ecx, 0xC0000080  ; EFER MSR
    rdmsr
    or eax, 1 << 8       ; LME bit
    wrmsr

    ; Enable paging
    mov eax, cr0
    or eax, 1 << 31
    mov cr0, eax

    ; Far jump to long mode
    jmp 0x18:long_mode

[BITS 64]
long_mode:
    ; Set up segment registers
    mov ax, 0x20
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    ; Jump to kernel
    mov rax, 0xFFFF800000000000 + kernel_start
    jmp rax
```

### Kernel Initialization

```rust
#[no_mangle]
pub extern "C" fn kernel_main(boot_info: &'static BootInfo) -> ! {
    // Phase 1: Core initialization
    serial::init();
    log::info!("Kernel starting...");

    // Phase 2: Interrupt handling
    gdt::init();
    idt::init();

    // Phase 3: Memory management
    let mut frame_allocator = unsafe {
        FrameAllocator::init(&boot_info.memory_map)
    };
    let mut mapper = unsafe {
        paging::init(&mut frame_allocator)
    };

    // Phase 4: Heap allocation
    heap::init(&mut mapper, &mut frame_allocator)
        .expect("heap initialization failed");

    // Phase 5: ACPI and hardware
    let acpi = acpi::init(boot_info.rsdp_addr);
    apic::init(&acpi);

    // Phase 6: Devices
    pci::init();
    timer::init(1000);  // 1000 Hz tick
    keyboard::init();

    // Phase 7: Process management
    process::init();
    scheduler::init();

    // Phase 8: Filesystem
    vfs::init();
    if let Some(ramdisk) = boot_info.ramdisk {
        ramfs::mount("/", ramdisk);
    }

    // Phase 9: Start init process
    let init = process::spawn("/sbin/init").expect("failed to start init");
    scheduler::add(init);

    // Enable interrupts and start scheduling
    x86_64::instructions::interrupts::enable();
    scheduler::run();
}
```

---

## Interrupt Handling

### IDT Setup

```rust
lazy_static! {
    static ref IDT: InterruptDescriptorTable = {
        let mut idt = InterruptDescriptorTable::new();

        // CPU exceptions
        idt.divide_error.set_handler_fn(divide_error_handler);
        idt.debug.set_handler_fn(debug_handler);
        idt.non_maskable_interrupt.set_handler_fn(nmi_handler);
        idt.breakpoint.set_handler_fn(breakpoint_handler);
        idt.overflow.set_handler_fn(overflow_handler);
        idt.bound_range_exceeded.set_handler_fn(bound_range_handler);
        idt.invalid_opcode.set_handler_fn(invalid_opcode_handler);
        idt.device_not_available.set_handler_fn(device_na_handler);
        idt.double_fault.set_handler_fn(double_fault_handler)
            .set_stack_index(DOUBLE_FAULT_IST_INDEX);
        idt.invalid_tss.set_handler_fn(invalid_tss_handler);
        idt.segment_not_present.set_handler_fn(segment_np_handler);
        idt.stack_segment_fault.set_handler_fn(stack_segment_handler);
        idt.general_protection_fault.set_handler_fn(gpf_handler);
        idt.page_fault.set_handler_fn(page_fault_handler);
        idt.x87_floating_point.set_handler_fn(x87_fp_handler);
        idt.alignment_check.set_handler_fn(alignment_check_handler);
        idt.machine_check.set_handler_fn(machine_check_handler);
        idt.simd_floating_point.set_handler_fn(simd_fp_handler);

        // Hardware interrupts
        idt[InterruptIndex::Timer.as_usize()].set_handler_fn(timer_handler);
        idt[InterruptIndex::Keyboard.as_usize()].set_handler_fn(keyboard_handler);

        // System call
        idt[0x80].set_handler_fn(syscall_handler)
            .set_privilege_level(PrivilegeLevel::Ring3);

        idt
    };
}

extern "x86-interrupt" fn page_fault_handler(
    stack_frame: InterruptStackFrame,
    error_code: PageFaultErrorCode,
) {
    let fault_addr = Cr2::read();

    // Check if this is a valid page fault we can handle
    if let Some(process) = scheduler::current_process() {
        // Check for stack growth
        if fault_addr >= process.stack_bottom - 4096 &&
           fault_addr < process.stack_top
        {
            // Allocate new stack page
            let frame = FRAME_ALLOCATOR.lock().allocate()
                .expect("out of memory");

            let page = Page::containing_address(fault_addr);
            MAPPER.lock().map_to(
                page,
                frame,
                PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::USER,
            ).expect("failed to map stack page");

            return;
        }

        // Check for demand paging
        if process.is_mapped_lazy(fault_addr) {
            // TODO: demand paging
        }
    }

    // Unhandled page fault
    panic!(
        "PAGE FAULT\n\
         Accessed Address: {:?}\n\
         Error Code: {:?}\n\
         {:?}",
        fault_addr, error_code, stack_frame
    );
}

extern "x86-interrupt" fn timer_handler(_stack_frame: InterruptStackFrame) {
    // Acknowledge interrupt
    APIC.lock().eoi();

    // Update system time
    TICKS.fetch_add(1, Ordering::SeqCst);

    // Wake sleeping processes
    scheduler::check_sleepers();

    // Trigger reschedule
    scheduler::schedule();
}
```

---

## System Calls

### Syscall Interface

```rust
#[repr(u64)]
pub enum Syscall {
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
    Fork = 57,
    Execve = 59,
    Exit = 60,
    Wait4 = 61,
    Kill = 62,
    Getpid = 39,
    Getppid = 110,
    Getuid = 102,
    Setuid = 105,
    Getcwd = 79,
    Chdir = 80,
}

pub fn syscall_handler(
    syscall_num: u64,
    arg1: u64,
    arg2: u64,
    arg3: u64,
    arg4: u64,
    arg5: u64,
    arg6: u64,
) -> i64 {
    let result = match Syscall::try_from(syscall_num) {
        Ok(Syscall::Read) => sys_read(arg1 as Fd, arg2 as *mut u8, arg3 as usize),
        Ok(Syscall::Write) => sys_write(arg1 as Fd, arg2 as *const u8, arg3 as usize),
        Ok(Syscall::Open) => sys_open(arg1 as *const u8, arg2 as u32, arg3 as u32),
        Ok(Syscall::Close) => sys_close(arg1 as Fd),
        Ok(Syscall::Fork) => sys_fork(),
        Ok(Syscall::Execve) => sys_execve(
            arg1 as *const u8,
            arg2 as *const *const u8,
            arg3 as *const *const u8,
        ),
        Ok(Syscall::Exit) => sys_exit(arg1 as i32),
        Ok(Syscall::Wait4) => sys_wait4(arg1 as i32, arg2 as *mut i32, arg3 as u32),
        Ok(Syscall::Kill) => sys_kill(arg1 as Pid, arg2 as i32),
        Ok(Syscall::Getpid) => sys_getpid(),
        Ok(Syscall::Mmap) => sys_mmap(arg1, arg2, arg3 as u32, arg4 as u32, arg5 as i32, arg6),
        Ok(Syscall::Brk) => sys_brk(arg1),
        _ => Err(Error::ENOSYS),
    };

    match result {
        Ok(val) => val as i64,
        Err(err) => -(err as i64),
    }
}

// Example syscall implementations
fn sys_write(fd: Fd, buf: *const u8, count: usize) -> Result<usize> {
    let process = scheduler::current_process().ok_or(Error::ESRCH)?;

    // Validate user pointer
    let slice = process.validate_user_slice(buf, count)?;

    // Get file descriptor
    let file = process.file_descriptors.get(&fd).ok_or(Error::EBADF)?;

    // Write to file
    file.write(slice)
}

fn sys_fork() -> Result<Pid> {
    let parent = scheduler::current_process().ok_or(Error::ESRCH)?;

    // Create child process
    let child = Process::fork(parent)?;

    // Copy page tables (CoW)
    child.page_table = parent.page_table.copy_on_write()?;

    // Copy file descriptors
    for (fd, file) in &parent.file_descriptors {
        child.file_descriptors.insert(*fd, file.clone());
    }

    // Add to scheduler
    let child_pid = child.pid;
    scheduler::add(child);

    // Return child PID to parent, 0 to child
    Ok(child_pid)
}

fn sys_execve(
    path: *const u8,
    argv: *const *const u8,
    envp: *const *const u8,
) -> Result<usize> {
    let process = scheduler::current_process().ok_or(Error::ESRCH)?;

    // Validate and read path
    let path_str = process.validate_user_string(path)?;

    // Read argv and envp
    let argv = process.validate_user_string_array(argv)?;
    let envp = process.validate_user_string_array(envp)?;

    // Load executable
    let elf = vfs::read(&path_str)?;
    let entry_point = elf::load(&elf, &mut process.page_table)?;

    // Set up user stack with argv and envp
    let stack_ptr = process.setup_user_stack(&argv, &envp)?;

    // Reset signal handlers
    process.signal_handlers = [SignalHandler::Default; 32];

    // Close CLOEXEC file descriptors
    process.file_descriptors.retain(|_, f| !f.flags.contains(O_CLOEXEC));

    // Set new entry point
    process.context.rip = entry_point;
    process.context.rsp = stack_ptr;

    Ok(0)
}
```

---

## Process Scheduler

### Round-Robin Scheduler

```rust
pub struct Scheduler {
    ready_queue: VecDeque<Box<Process>>,
    current: Option<Box<Process>>,
    idle: Box<Process>,
}

impl Scheduler {
    pub fn schedule(&mut self) {
        // Save current process context
        if let Some(mut current) = self.current.take() {
            if current.state == ProcessState::Running {
                current.state = ProcessState::Ready;
                self.ready_queue.push_back(current);
            }
        }

        // Select next process
        let next = self.ready_queue.pop_front()
            .unwrap_or_else(|| self.idle.clone());

        // Context switch
        let old_ctx = self.current.as_ref().map(|p| &p.context);
        let new_ctx = &next.context;

        // Switch page table
        unsafe {
            Cr3::write(
                PhysFrame::containing_address(next.page_table.root),
                Cr3Flags::empty(),
            );
        }

        next.state = ProcessState::Running;
        self.current = Some(next);

        // Switch context
        unsafe {
            context_switch(old_ctx, new_ctx);
        }
    }
}

#[naked]
unsafe extern "C" fn context_switch(old: *mut CpuContext, new: *const CpuContext) {
    asm!(
        // Save old context
        "mov [rdi + 0x00], rax",
        "mov [rdi + 0x08], rbx",
        "mov [rdi + 0x10], rcx",
        "mov [rdi + 0x18], rdx",
        "mov [rdi + 0x20], rsi",
        "mov [rdi + 0x28], rdi",
        "mov [rdi + 0x30], rbp",
        "mov [rdi + 0x38], rsp",
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

        // Load new context
        "mov rax, [rsi + 0x00]",
        "mov rbx, [rsi + 0x08]",
        "mov rcx, [rsi + 0x10]",
        "mov rdx, [rsi + 0x18]",
        // Skip rsi and rdi for now
        "mov rbp, [rsi + 0x30]",
        "mov rsp, [rsi + 0x38]",
        "mov r8,  [rsi + 0x40]",
        "mov r9,  [rsi + 0x48]",
        "mov r10, [rsi + 0x50]",
        "mov r11, [rsi + 0x58]",
        "mov r12, [rsi + 0x60]",
        "mov r13, [rsi + 0x68]",
        "mov r14, [rsi + 0x70]",
        "mov r15, [rsi + 0x78]",

        // Push new return address
        "push [rsi + 0x80]",

        // Load rsi and rdi last
        "mov rdi, [rsi + 0x28]",
        "mov rsi, [rsi + 0x20]",

        "ret",
        options(noreturn)
    );
}
```

### Priority-Based Scheduler (Enterprise)

```rust
pub struct PriorityScheduler {
    // Priority queues (0 = highest, 139 = lowest)
    queues: [VecDeque<Box<Process>>; 140],
    bitmap: u128,  // Bitmap of non-empty queues
    current: Option<Box<Process>>,
}

impl PriorityScheduler {
    pub fn schedule(&mut self) {
        // Find highest priority non-empty queue
        let highest = self.bitmap.leading_zeros() as usize;
        if highest < 140 {
            let next = self.queues[highest].pop_front().unwrap();
            if self.queues[highest].is_empty() {
                self.bitmap &= !(1 << (139 - highest));
            }
            // Context switch...
        }
    }

    pub fn add(&mut self, process: Box<Process>) {
        let priority = self.calculate_priority(&process);
        self.queues[priority].push_back(process);
        self.bitmap |= 1 << (139 - priority);
    }

    fn calculate_priority(&self, process: &Process) -> usize {
        // Nice value: -20 to 19
        // Static priority: 100 to 139
        let static_prio = (process.nice + 20 + 100) as usize;

        // Dynamic priority based on sleep time
        let bonus = process.sleep_avg / 5;
        static_prio.saturating_sub(bonus).clamp(0, 139)
    }
}
```

---

## Filesystem

### VFS Layer

```rust
pub struct VFS {
    mounts: BTreeMap<PathBuf, Box<dyn FileSystem>>,
    root: Inode,
}

pub trait FileSystem: Send + Sync {
    fn name(&self) -> &str;
    fn read_inode(&self, ino: u64) -> Result<Inode>;
    fn lookup(&self, parent: &Inode, name: &str) -> Result<Inode>;
    fn read(&self, inode: &Inode, offset: usize, buf: &mut [u8]) -> Result<usize>;
    fn write(&self, inode: &Inode, offset: usize, buf: &[u8]) -> Result<usize>;
    fn create(&self, parent: &Inode, name: &str, mode: u32) -> Result<Inode>;
    fn mkdir(&self, parent: &Inode, name: &str, mode: u32) -> Result<Inode>;
    fn unlink(&self, parent: &Inode, name: &str) -> Result<()>;
    fn readdir(&self, inode: &Inode) -> Result<Vec<DirEntry>>;
}

pub struct Inode {
    pub ino: u64,
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub size: u64,
    pub atime: u64,
    pub mtime: u64,
    pub ctime: u64,
    pub nlink: u32,
    pub fs: Weak<dyn FileSystem>,
}

pub struct DirEntry {
    pub ino: u64,
    pub name: String,
    pub file_type: FileType,
}

impl VFS {
    pub fn open(&self, path: &str, flags: u32, mode: u32) -> Result<OpenFile> {
        let (fs, inode) = self.resolve_path(path)?;

        if flags & O_CREAT != 0 && inode.is_none() {
            let parent = self.resolve_parent(path)?;
            let name = path_basename(path);
            let inode = fs.create(&parent, name, mode)?;
            return Ok(OpenFile::new(inode, flags));
        }

        let inode = inode.ok_or(Error::ENOENT)?;
        Ok(OpenFile::new(inode, flags))
    }
}
```

### RAM Filesystem

```rust
pub struct RamFS {
    inodes: RwLock<BTreeMap<u64, RamInode>>,
    next_ino: AtomicU64,
}

struct RamInode {
    metadata: InodeMetadata,
    data: RamInodeData,
}

enum RamInodeData {
    File(Vec<u8>),
    Directory(BTreeMap<String, u64>),
    Symlink(String),
}

impl FileSystem for RamFS {
    fn read(&self, inode: &Inode, offset: usize, buf: &mut [u8]) -> Result<usize> {
        let inodes = self.inodes.read();
        let ram_inode = inodes.get(&inode.ino).ok_or(Error::ENOENT)?;

        match &ram_inode.data {
            RamInodeData::File(data) => {
                let available = data.len().saturating_sub(offset);
                let to_read = std::cmp::min(buf.len(), available);
                buf[..to_read].copy_from_slice(&data[offset..offset + to_read]);
                Ok(to_read)
            }
            _ => Err(Error::EISDIR),
        }
    }

    fn write(&self, inode: &Inode, offset: usize, buf: &[u8]) -> Result<usize> {
        let mut inodes = self.inodes.write();
        let ram_inode = inodes.get_mut(&inode.ino).ok_or(Error::ENOENT)?;

        match &mut ram_inode.data {
            RamInodeData::File(data) => {
                let new_len = offset + buf.len();
                if new_len > data.len() {
                    data.resize(new_len, 0);
                }
                data[offset..offset + buf.len()].copy_from_slice(buf);
                ram_inode.metadata.size = data.len() as u64;
                Ok(buf.len())
            }
            _ => Err(Error::EISDIR),
        }
    }
}
```

---

## Signals

```rust
pub struct SignalHandler {
    handler: SignalAction,
    flags: SignalFlags,
    mask: SignalSet,
}

pub enum SignalAction {
    Default,
    Ignore,
    Handler(VirtAddr),
}

impl Process {
    pub fn deliver_signal(&mut self, signal: Signal) {
        let handler = &self.signal_handlers[signal as usize];

        match handler.handler {
            SignalAction::Default => {
                match signal.default_action() {
                    DefaultAction::Terminate => self.exit(-signal as i32),
                    DefaultAction::Ignore => {},
                    DefaultAction::Stop => self.state = ProcessState::Stopped,
                    DefaultAction::Continue => self.state = ProcessState::Ready,
                    DefaultAction::CoreDump => {
                        // TODO: generate core dump
                        self.exit(-signal as i32);
                    }
                }
            }
            SignalAction::Ignore => {}
            SignalAction::Handler(addr) => {
                // Set up signal trampoline on user stack
                self.setup_signal_frame(signal, addr);
            }
        }
    }

    fn setup_signal_frame(&mut self, signal: Signal, handler: VirtAddr) {
        // Save current context to user stack
        let frame = SignalFrame {
            context: self.context.clone(),
            signal,
        };

        // Push frame to user stack
        self.context.rsp -= std::mem::size_of::<SignalFrame>() as u64;
        unsafe {
            *(self.context.rsp as *mut SignalFrame) = frame;
        }

        // Set up return to sigreturn trampoline
        self.context.rsp -= 8;
        unsafe {
            *(self.context.rsp as *mut u64) = SIGRETURN_TRAMPOLINE;
        }

        // Jump to handler
        self.context.rip = handler.as_u64();
        self.context.rdi = signal as u64;  // First argument
    }
}
```

---

## Implementation Phases

### Phase 1: Bootloader (Week 1)
- [ ] BIOS bootloader (stages 1 & 2)
- [ ] Protected mode setup
- [ ] Long mode transition
- [ ] Initial page tables
- [ ] Kernel loading

### Phase 2: Kernel Initialization (Week 2)
- [ ] GDT and TSS setup
- [ ] IDT and exception handlers
- [ ] Memory map parsing
- [ ] Frame allocator
- [ ] Kernel heap

### Phase 3: Paging (Week 3)
- [ ] 4-level page tables
- [ ] Virtual memory mapping
- [ ] Higher-half kernel
- [ ] Kernel/user space separation

### Phase 4: Processes (Week 4-5)
- [ ] Process structure
- [ ] Context switching
- [ ] Fork implementation
- [ ] Exec implementation
- [ ] Process termination

### Phase 5: Scheduler (Week 6)
- [ ] Round-robin scheduler
- [ ] Timer interrupt
- [ ] Sleep/wake
- [ ] Priority scheduling (optional)

### Phase 6: System Calls (Week 7)
- [ ] Syscall interface
- [ ] File operations (read/write/open/close)
- [ ] Process operations (fork/exec/wait/exit)
- [ ] Memory operations (mmap/brk)

### Phase 7: Filesystem (Week 8)
- [ ] VFS layer
- [ ] RAM filesystem
- [ ] File descriptors
- [ ] Directory operations

### Phase 8: Signals & IPC (Week 9)
- [ ] Signal delivery
- [ ] Signal handlers
- [ ] Pipes
- [ ] Basic IPC

### Phase 9: User Space (Week 10)
- [ ] User/kernel mode
- [ ] Simple shell
- [ ] Init process
- [ ] Basic utilities

---

## Testing Strategy

### Unit Tests
- Page table manipulation
- Frame allocator
- Scheduler logic

### Integration Tests
- Boot sequence
- Process creation/destruction
- System call handling

### User Space Tests
```c
// test_fork.c
#include <unistd.h>
#include <stdio.h>

int main() {
    int pid = fork();
    if (pid == 0) {
        printf("Child process: %d\n", getpid());
        return 42;
    } else {
        int status;
        wait(&status);
        printf("Parent: child exited with %d\n", WEXITSTATUS(status));
    }
    return 0;
}
```

---

## Stretch Goals

### SMP Support
- Per-CPU data structures
- Spinlocks with proper barriers
- IPI for TLB shootdown
- Load balancing

### Demand Paging
- Lazy allocation
- Page fault handling
- Swap support

### Network Stack
- Device driver
- IP/TCP/UDP
- Socket interface

---

## Dependencies

```toml
[dependencies]
bootloader = "0.9"
x86_64 = "0.14"
uart_16550 = "0.2"
pic8259 = "0.10"
pc-keyboard = "0.6"
spin = "0.9"
lazy_static = { version = "1", features = ["spin_no_std"] }
```

---

## References

- [OSDev Wiki](https://wiki.osdev.org/)
- [Writing an OS in Rust](https://os.phil-opp.com/)
- [Intel SDM](https://software.intel.com/content/www/us/en/develop/articles/intel-sdm.html)
- [xv6](https://github.com/mit-pdos/xv6-public)
- [Linux Kernel](https://github.com/torvalds/linux)
