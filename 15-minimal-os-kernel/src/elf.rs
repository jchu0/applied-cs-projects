//! ELF (Executable and Linkable Format) loader.
//!
//! This module provides functionality to load and execute ELF64 binaries.

use alloc::vec::Vec;

/// ELF magic number.
pub const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];

/// ELF class for 64-bit.
pub const ELFCLASS64: u8 = 2;

/// ELF data encoding for little-endian.
pub const ELFDATA2LSB: u8 = 1;

/// ELF machine type for x86_64.
pub const EM_X86_64: u16 = 62;

/// Program header type: loadable segment.
pub const PT_LOAD: u32 = 1;

/// ELF64 file header.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct Elf64Header {
    /// Magic number and other info.
    pub e_ident: [u8; 16],
    /// Object file type.
    pub e_type: u16,
    /// Machine type.
    pub e_machine: u16,
    /// ELF version.
    pub e_version: u32,
    /// Entry point virtual address.
    pub e_entry: u64,
    /// Program header table file offset.
    pub e_phoff: u64,
    /// Section header table file offset.
    pub e_shoff: u64,
    /// Processor-specific flags.
    pub e_flags: u32,
    /// ELF header size.
    pub e_ehsize: u16,
    /// Program header table entry size.
    pub e_phentsize: u16,
    /// Program header table entry count.
    pub e_phnum: u16,
    /// Section header table entry size.
    pub e_shentsize: u16,
    /// Section header table entry count.
    pub e_shnum: u16,
    /// Section name string table index.
    pub e_shstrndx: u16,
}

/// ELF64 program header.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct Elf64ProgramHeader {
    /// Segment type.
    pub p_type: u32,
    /// Segment flags.
    pub p_flags: u32,
    /// Segment file offset.
    pub p_offset: u64,
    /// Segment virtual address.
    pub p_vaddr: u64,
    /// Segment physical address.
    pub p_paddr: u64,
    /// Segment size in file.
    pub p_filesz: u64,
    /// Segment size in memory.
    pub p_memsz: u64,
    /// Segment alignment.
    pub p_align: u64,
}

/// Segment flags.
pub mod flags {
    /// Execute permission.
    pub const PF_X: u32 = 1;
    /// Write permission.
    pub const PF_W: u32 = 2;
    /// Read permission.
    pub const PF_R: u32 = 4;
}

/// ELF loading error.
#[derive(Debug, Clone, Copy)]
pub enum ElfError {
    /// Invalid ELF magic number.
    InvalidMagic,
    /// Unsupported ELF class (not 64-bit).
    UnsupportedClass,
    /// Unsupported data encoding.
    UnsupportedEncoding,
    /// Unsupported machine type.
    UnsupportedMachine,
    /// Invalid program header.
    InvalidProgramHeader,
    /// Memory allocation failed.
    MemoryError,
    /// File too small.
    TooSmall,
}

/// Information about a loaded segment.
#[derive(Debug, Clone)]
pub struct LoadedSegment {
    /// Virtual address.
    pub vaddr: u64,
    /// Memory size.
    pub memsz: u64,
    /// Flags (permissions).
    pub flags: u32,
}

/// Result of loading an ELF file.
#[derive(Debug)]
pub struct LoadedElf {
    /// Entry point address.
    pub entry: u64,
    /// Loaded segments.
    pub segments: Vec<LoadedSegment>,
    /// Stack top address.
    pub stack_top: u64,
}

/// Validate an ELF64 header.
pub fn validate_header(header: &Elf64Header) -> Result<(), ElfError> {
    // Check magic number
    if header.e_ident[0..4] != ELF_MAGIC {
        return Err(ElfError::InvalidMagic);
    }

    // Check 64-bit class
    if header.e_ident[4] != ELFCLASS64 {
        return Err(ElfError::UnsupportedClass);
    }

    // Check little-endian
    if header.e_ident[5] != ELFDATA2LSB {
        return Err(ElfError::UnsupportedEncoding);
    }

    // Check x86_64 machine
    if header.e_machine != EM_X86_64 {
        return Err(ElfError::UnsupportedMachine);
    }

    Ok(())
}

/// Parse the ELF header from raw bytes.
pub fn parse_header(data: &[u8]) -> Result<Elf64Header, ElfError> {
    if data.len() < core::mem::size_of::<Elf64Header>() {
        return Err(ElfError::TooSmall);
    }

    let header = unsafe {
        *(data.as_ptr() as *const Elf64Header)
    };

    validate_header(&header)?;
    Ok(header)
}

/// Parse program headers from ELF data.
pub fn parse_program_headers(data: &[u8], header: &Elf64Header) -> Result<Vec<Elf64ProgramHeader>, ElfError> {
    let phoff = header.e_phoff as usize;
    let phentsize = header.e_phentsize as usize;
    let phnum = header.e_phnum as usize;

    if data.len() < phoff + phentsize * phnum {
        return Err(ElfError::TooSmall);
    }

    let mut headers = Vec::with_capacity(phnum);
    for i in 0..phnum {
        let offset = phoff + i * phentsize;
        let ph = unsafe {
            *(data.as_ptr().add(offset) as *const Elf64ProgramHeader)
        };
        headers.push(ph);
    }

    Ok(headers)
}

/// Load an ELF64 executable.
///
/// This function parses the ELF file and returns information about what
/// needs to be loaded, but does not perform the actual memory mapping.
/// The caller is responsible for setting up page tables and copying data.
pub fn load_elf(data: &[u8]) -> Result<LoadedElf, ElfError> {
    let header = parse_header(data)?;
    let program_headers = parse_program_headers(data, &header)?;

    let mut segments = Vec::new();
    let mut max_vaddr = 0u64;

    for ph in program_headers {
        if ph.p_type == PT_LOAD {
            segments.push(LoadedSegment {
                vaddr: ph.p_vaddr,
                memsz: ph.p_memsz,
                flags: ph.p_flags,
            });

            let end = ph.p_vaddr.saturating_add(ph.p_memsz);
            if end > max_vaddr {
                max_vaddr = end;
            }
        }
    }

    // Allocate stack above loaded segments
    // Stack grows down, so stack_top is the highest address
    let stack_top = (max_vaddr + 0x100000) & !0xFFF;  // Align to page boundary + 1MB

    Ok(LoadedElf {
        entry: header.e_entry,
        segments,
        stack_top,
    })
}

/// Get segment data from ELF file.
pub fn get_segment_data<'a>(data: &'a [u8], ph: &Elf64ProgramHeader) -> &'a [u8] {
    let start = ph.p_offset as usize;
    let end = start + ph.p_filesz as usize;
    &data[start..end]
}

/// Check if a segment is executable.
pub fn is_executable(ph: &Elf64ProgramHeader) -> bool {
    ph.p_flags & flags::PF_X != 0
}

/// Check if a segment is writable.
pub fn is_writable(ph: &Elf64ProgramHeader) -> bool {
    ph.p_flags & flags::PF_W != 0
}

/// Check if a segment is readable.
pub fn is_readable(ph: &Elf64ProgramHeader) -> bool {
    ph.p_flags & flags::PF_R != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_magic() {
        let mut header = Elf64Header {
            e_ident: [0; 16],
            e_type: 0,
            e_machine: EM_X86_64,
            e_version: 0,
            e_entry: 0,
            e_phoff: 0,
            e_shoff: 0,
            e_flags: 0,
            e_ehsize: 0,
            e_phentsize: 0,
            e_phnum: 0,
            e_shentsize: 0,
            e_shnum: 0,
            e_shstrndx: 0,
        };

        // Invalid magic
        assert!(matches!(validate_header(&header), Err(ElfError::InvalidMagic)));

        // Set correct magic
        header.e_ident[0] = 0x7f;
        header.e_ident[1] = b'E';
        header.e_ident[2] = b'L';
        header.e_ident[3] = b'F';
        header.e_ident[4] = ELFCLASS64;
        header.e_ident[5] = ELFDATA2LSB;

        assert!(validate_header(&header).is_ok());
    }

    #[test]
    fn test_segment_flags() {
        let ph = Elf64ProgramHeader {
            p_type: PT_LOAD,
            p_flags: flags::PF_R | flags::PF_X,
            p_offset: 0,
            p_vaddr: 0,
            p_paddr: 0,
            p_filesz: 0,
            p_memsz: 0,
            p_align: 0,
        };

        assert!(is_readable(&ph));
        assert!(is_executable(&ph));
        assert!(!is_writable(&ph));
    }
}
