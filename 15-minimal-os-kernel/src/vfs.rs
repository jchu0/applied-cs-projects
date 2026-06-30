//! Virtual File System.

use crate::syscall::SyscallError;
use crate::process::Fd;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use spin::Mutex;

// ============================================================================
// File open flags (POSIX-compatible)
// ============================================================================

/// Open for reading only.
pub const O_RDONLY: u32 = 0;
/// Open for writing only.
pub const O_WRONLY: u32 = 1;
/// Open for reading and writing.
pub const O_RDWR: u32 = 2;
/// Create file if it doesn't exist.
pub const O_CREAT: u32 = 0o100;
/// Fail if file exists (with O_CREAT).
pub const O_EXCL: u32 = 0o200;
/// Truncate file to zero length.
pub const O_TRUNC: u32 = 0o1000;
/// Append to file.
pub const O_APPEND: u32 = 0o2000;

// ============================================================================
// File mode constants
// ============================================================================

/// Regular file.
pub const S_IFREG: u32 = 0o100000;
/// Directory.
pub const S_IFDIR: u32 = 0o040000;
/// Character device.
pub const S_IFCHR: u32 = 0o020000;
/// Block device.
pub const S_IFBLK: u32 = 0o060000;
/// FIFO.
pub const S_IFIFO: u32 = 0o010000;
/// Symbolic link.
pub const S_IFLNK: u32 = 0o120000;
/// Socket.
pub const S_IFSOCK: u32 = 0o140000;

/// Global VFS.
static VFS: Mutex<Option<Vfs>> = Mutex::new(None);

/// File type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    /// Regular file.
    Regular,
    /// Directory.
    Directory,
    /// Symbolic link.
    Symlink,
    /// Character device.
    CharDevice,
    /// Block device.
    BlockDevice,
    /// Named pipe.
    Fifo,
    /// Socket.
    Socket,
}

/// Inode metadata.
#[derive(Debug, Clone)]
pub struct InodeMetadata {
    /// Inode number.
    pub ino: u64,
    /// File type and permissions.
    pub mode: u32,
    /// Owner UID.
    pub uid: u32,
    /// Owner GID.
    pub gid: u32,
    /// File size.
    pub size: u64,
    /// Access time.
    pub atime: u64,
    /// Modification time.
    pub mtime: u64,
    /// Change time.
    pub ctime: u64,
    /// Link count.
    pub nlink: u32,
}

impl InodeMetadata {
    /// Create new metadata.
    pub fn new(ino: u64, file_type: FileType) -> Self {
        let mode = match file_type {
            FileType::Directory => 0o040755,
            FileType::Regular => 0o100644,
            FileType::Symlink => 0o120777,
            FileType::CharDevice => 0o020666,
            FileType::BlockDevice => 0o060666,
            FileType::Fifo => 0o010644,
            FileType::Socket => 0o140666,
        };

        Self {
            ino,
            mode,
            uid: 0,
            gid: 0,
            size: 0,
            atime: 0,
            mtime: 0,
            ctime: 0,
            nlink: 1,
        }
    }

    /// Get file type.
    pub fn file_type(&self) -> FileType {
        match (self.mode >> 12) & 0xF {
            0o04 => FileType::Directory,
            0o10 => FileType::Regular,
            0o12 => FileType::Symlink,
            0o02 => FileType::CharDevice,
            0o06 => FileType::BlockDevice,
            0o01 => FileType::Fifo,
            0o14 => FileType::Socket,
            _ => FileType::Regular,
        }
    }
}

/// Directory entry.
#[derive(Debug, Clone)]
pub struct DirEntry {
    /// Inode number.
    pub ino: u64,
    /// Entry name.
    pub name: String,
    /// File type.
    pub file_type: FileType,
    /// Is this a directory?
    pub is_dir: bool,
}

/// RAM filesystem inode data.
enum RamInodeData {
    /// Regular file data.
    File(Vec<u8>),
    /// Directory entries.
    Directory(BTreeMap<String, u64>),
    /// Symlink target.
    Symlink(String),
}

/// RAM filesystem inode.
struct RamInode {
    /// Metadata.
    metadata: InodeMetadata,
    /// Data.
    data: RamInodeData,
}

/// RAM filesystem.
pub struct RamFs {
    /// Inodes by number.
    inodes: BTreeMap<u64, RamInode>,
    /// Next inode number.
    next_ino: u64,
}

impl RamFs {
    /// Create a new RAM filesystem.
    pub fn new() -> Self {
        let mut fs = Self {
            inodes: BTreeMap::new(),
            next_ino: 2, // 1 is reserved for root
        };

        // Create root directory
        let root_metadata = InodeMetadata::new(1, FileType::Directory);
        let mut entries = BTreeMap::new();
        entries.insert(String::from("."), 1);
        entries.insert(String::from(".."), 1);

        fs.inodes.insert(1, RamInode {
            metadata: root_metadata,
            data: RamInodeData::Directory(entries),
        });

        fs
    }

    /// Allocate an inode number.
    fn alloc_ino(&mut self) -> u64 {
        let ino = self.next_ino;
        self.next_ino += 1;
        ino
    }

    /// Lookup path.
    pub fn lookup(&self, path: &str) -> Option<u64> {
        let parts: Vec<&str> = path.split('/')
            .filter(|s| !s.is_empty())
            .collect();

        let mut current_ino = 1; // Start at root

        for part in parts {
            let inode = self.inodes.get(&current_ino)?;
            match &inode.data {
                RamInodeData::Directory(entries) => {
                    current_ino = *entries.get(part)?;
                }
                _ => return None,
            }
        }

        Some(current_ino)
    }

    /// Create a file.
    pub fn create(&mut self, path: &str) -> Result<u64, SyscallError> {
        let (parent_path, name) = split_path(path);

        let parent_ino = self.lookup(parent_path)
            .ok_or(SyscallError::ENOENT)?;

        let parent = self.inodes.get_mut(&parent_ino)
            .ok_or(SyscallError::ENOENT)?;

        match &mut parent.data {
            RamInodeData::Directory(entries) => {
                if entries.contains_key(name) {
                    return Err(SyscallError::EEXIST);
                }

                let ino = self.alloc_ino();
                entries.insert(String::from(name), ino);

                let metadata = InodeMetadata::new(ino, FileType::Regular);
                self.inodes.insert(ino, RamInode {
                    metadata,
                    data: RamInodeData::File(Vec::new()),
                });

                Ok(ino)
            }
            _ => Err(SyscallError::ENOTDIR),
        }
    }

    /// Create a directory.
    pub fn mkdir(&mut self, path: &str) -> Result<u64, SyscallError> {
        let (parent_path, name) = split_path(path);

        let parent_ino = self.lookup(parent_path)
            .ok_or(SyscallError::ENOENT)?;

        let parent = self.inodes.get_mut(&parent_ino)
            .ok_or(SyscallError::ENOENT)?;

        match &mut parent.data {
            RamInodeData::Directory(entries) => {
                if entries.contains_key(name) {
                    return Err(SyscallError::EEXIST);
                }

                let ino = self.alloc_ino();
                entries.insert(String::from(name), ino);

                let metadata = InodeMetadata::new(ino, FileType::Directory);
                let mut dir_entries = BTreeMap::new();
                dir_entries.insert(String::from("."), ino);
                dir_entries.insert(String::from(".."), parent_ino);

                self.inodes.insert(ino, RamInode {
                    metadata,
                    data: RamInodeData::Directory(dir_entries),
                });

                Ok(ino)
            }
            _ => Err(SyscallError::ENOTDIR),
        }
    }

    /// Read file data.
    pub fn read(&self, ino: u64, offset: usize, buf: &mut [u8]) -> Result<usize, SyscallError> {
        let inode = self.inodes.get(&ino)
            .ok_or(SyscallError::ENOENT)?;

        match &inode.data {
            RamInodeData::File(data) => {
                let available = data.len().saturating_sub(offset);
                let to_read = buf.len().min(available);
                buf[..to_read].copy_from_slice(&data[offset..offset + to_read]);
                Ok(to_read)
            }
            RamInodeData::Directory(_) => Err(SyscallError::EISDIR),
            _ => Err(SyscallError::EINVAL),
        }
    }

    /// Write file data.
    pub fn write(&mut self, ino: u64, offset: usize, buf: &[u8]) -> Result<usize, SyscallError> {
        let inode = self.inodes.get_mut(&ino)
            .ok_or(SyscallError::ENOENT)?;

        match &mut inode.data {
            RamInodeData::File(data) => {
                let new_len = offset + buf.len();
                if new_len > data.len() {
                    data.resize(new_len, 0);
                }
                data[offset..offset + buf.len()].copy_from_slice(buf);
                inode.metadata.size = data.len() as u64;
                Ok(buf.len())
            }
            RamInodeData::Directory(_) => Err(SyscallError::EISDIR),
            _ => Err(SyscallError::EINVAL),
        }
    }

    /// Read directory entries.
    pub fn readdir(&self, ino: u64) -> Result<Vec<DirEntry>, SyscallError> {
        let inode = self.inodes.get(&ino)
            .ok_or(SyscallError::ENOENT)?;

        match &inode.data {
            RamInodeData::Directory(entries) => {
                let mut result = Vec::new();
                for (name, &child_ino) in entries {
                    let child = self.inodes.get(&child_ino)
                        .ok_or(SyscallError::EIO)?;
                    let file_type = child.metadata.file_type();
                    result.push(DirEntry {
                        ino: child_ino,
                        name: name.clone(),
                        file_type,
                        is_dir: file_type == FileType::Directory,
                    });
                }
                Ok(result)
            }
            _ => Err(SyscallError::ENOTDIR),
        }
    }

    /// Get inode metadata.
    pub fn stat(&self, ino: u64) -> Result<InodeMetadata, SyscallError> {
        let inode = self.inodes.get(&ino)
            .ok_or(SyscallError::ENOENT)?;
        Ok(inode.metadata.clone())
    }

    /// Check if path exists.
    pub fn exists(&self, path: &str) -> bool {
        self.lookup(path).is_some()
    }
}

/// Not a directory error.
#[derive(Debug)]
pub struct NotDirectory;

impl core::fmt::Display for NotDirectory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Not a directory")
    }
}

/// Add ENOTDIR to SyscallError if needed (simplified).
impl SyscallError {
    pub const ENOTDIR: SyscallError = SyscallError::EINVAL;
}

/// VFS layer.
pub struct Vfs {
    /// Mounted filesystems.
    mounts: BTreeMap<String, Box<RamFs>>,
    /// Root filesystem.
    root: RamFs,
}

impl Vfs {
    /// Create a new VFS.
    pub fn new() -> Self {
        Self {
            mounts: BTreeMap::new(),
            root: RamFs::new(),
        }
    }
}

/// Split path into parent and name.
fn split_path(path: &str) -> (&str, &str) {
    match path.rfind('/') {
        Some(0) => ("/", &path[1..]),
        Some(i) => (&path[..i], &path[i + 1..]),
        None => (".", path),
    }
}

/// Initialize VFS.
pub fn init() {
    let vfs = Vfs::new();
    *VFS.lock() = Some(vfs);
}

/// Check if path exists.
pub fn exists(path: &str) -> bool {
    VFS.lock().as_ref()
        .map(|vfs| vfs.root.exists(path))
        .unwrap_or(false)
}

/// Create a file.
pub fn create_file(path: &str) -> Result<u64, SyscallError> {
    VFS.lock().as_mut()
        .ok_or(SyscallError::EIO)?
        .root.create(path)
}

/// Read file.
pub fn read(path: &str, offset: usize, buf: &mut [u8]) -> Result<usize, SyscallError> {
    let vfs = VFS.lock();
    let vfs = vfs.as_ref().ok_or(SyscallError::EIO)?;
    let ino = vfs.root.lookup(path).ok_or(SyscallError::ENOENT)?;
    vfs.root.read(ino, offset, buf)
}

/// Write file.
pub fn write(path: &str, offset: usize, buf: &[u8]) -> Result<usize, SyscallError> {
    let mut vfs = VFS.lock();
    let vfs = vfs.as_mut().ok_or(SyscallError::EIO)?;
    let ino = vfs.root.lookup(path).ok_or(SyscallError::ENOENT)?;
    vfs.root.write(ino, offset, buf)
}

/// Create directory.
pub fn mkdir(path: &str, _mode: u32) -> Result<u64, SyscallError> {
    VFS.lock().as_mut()
        .ok_or(SyscallError::EIO)?
        .root.mkdir(path)
}

/// Get file metadata.
pub fn stat(path: &str) -> Result<InodeMetadata, SyscallError> {
    let vfs = VFS.lock();
    let vfs = vfs.as_ref().ok_or(SyscallError::EIO)?;
    let ino = vfs.root.lookup(path).ok_or(SyscallError::ENOENT)?;
    vfs.root.stat(ino)
}

/// Read directory.
pub fn readdir(path: &str) -> Result<Vec<DirEntry>, SyscallError> {
    let vfs = VFS.lock();
    let vfs = vfs.as_ref().ok_or(SyscallError::EIO)?;
    let ino = vfs.root.lookup(path).ok_or(SyscallError::ENOENT)?;
    vfs.root.readdir(ino)
}

/// Read entire file contents.
pub fn read_file(path: &str) -> Result<Vec<u8>, SyscallError> {
    let vfs = VFS.lock();
    let vfs = vfs.as_ref().ok_or(SyscallError::EIO)?;
    let ino = vfs.root.lookup(path).ok_or(SyscallError::ENOENT)?;

    // Get file size
    let metadata = vfs.root.stat(ino)?;
    let size = metadata.size as usize;

    // Read entire file
    let mut data = alloc::vec![0u8; size];
    vfs.root.read(ino, 0, &mut data)?;

    Ok(data)
}

/// Write entire file contents.
pub fn write_file(path: &str, data: &[u8]) -> Result<(), SyscallError> {
    let mut vfs = VFS.lock();
    let vfs = vfs.as_mut().ok_or(SyscallError::EIO)?;

    // Create file if it doesn't exist
    let ino = match vfs.root.lookup(path) {
        Some(ino) => ino,
        None => vfs.root.create(path)?,
    };

    // Write data
    vfs.root.write(ino, 0, data)?;

    Ok(())
}

// ============================================================================
// File descriptor-based operations (shell-compatible API)
// ============================================================================

/// Open file entry in global table.
struct OpenFile {
    /// Inode number.
    ino: u64,
    /// Current offset.
    offset: usize,
    /// Open flags.
    flags: u32,
    /// File path (for reference).
    path: String,
}

/// Global open file table.
static OPEN_FILES: Mutex<BTreeMap<Fd, OpenFile>> = Mutex::new(BTreeMap::new());

/// Next file descriptor.
static NEXT_FD: Mutex<Fd> = Mutex::new(3); // 0, 1, 2 reserved

/// Open a file.
///
/// Returns a file descriptor on success.
pub fn open(path: &str, flags: u32, _mode: u32) -> Result<Fd, SyscallError> {
    let mut vfs = VFS.lock();
    let vfs = vfs.as_mut().ok_or(SyscallError::EIO)?;

    // Check if file exists
    let ino = match vfs.root.lookup(path) {
        Some(ino) => ino,
        None => {
            // Create if O_CREAT is set
            if flags & O_CREAT != 0 {
                vfs.root.create(path)?
            } else {
                return Err(SyscallError::ENOENT);
            }
        }
    };

    // Truncate if O_TRUNC is set
    if flags & O_TRUNC != 0 {
        // Clear file contents
        if let Some(inode) = vfs.root.inodes.get_mut(&ino) {
            if let RamInodeData::File(data) = &mut inode.data {
                data.clear();
                inode.metadata.size = 0;
            }
        }
    }

    // Allocate file descriptor
    let mut next_fd = NEXT_FD.lock();
    let fd = *next_fd;
    *next_fd += 1;

    // Add to open file table
    let offset = if flags & O_APPEND != 0 {
        vfs.root.stat(ino)?.size as usize
    } else {
        0
    };

    OPEN_FILES.lock().insert(fd, OpenFile {
        ino,
        offset,
        flags,
        path: String::from(path),
    });

    Ok(fd)
}

/// Close a file descriptor.
pub fn close(fd: Fd) -> Result<(), SyscallError> {
    let mut files = OPEN_FILES.lock();
    files.remove(&fd).ok_or(SyscallError::EBADF)?;
    Ok(())
}

/// Read from a file descriptor.
pub fn read_fd(fd: Fd, buf: &mut [u8]) -> Result<usize, SyscallError> {
    let vfs = VFS.lock();
    let vfs = vfs.as_ref().ok_or(SyscallError::EIO)?;

    let mut files = OPEN_FILES.lock();
    let file = files.get_mut(&fd).ok_or(SyscallError::EBADF)?;

    // Check if readable
    let access = file.flags & 3;
    if access == O_WRONLY {
        return Err(SyscallError::EBADF);
    }

    let bytes_read = vfs.root.read(file.ino, file.offset, buf)?;
    file.offset += bytes_read;

    Ok(bytes_read)
}

/// Write to a file descriptor.
pub fn write_fd(fd: Fd, buf: &[u8]) -> Result<usize, SyscallError> {
    let mut vfs = VFS.lock();
    let vfs = vfs.as_mut().ok_or(SyscallError::EIO)?;

    let mut files = OPEN_FILES.lock();
    let file = files.get_mut(&fd).ok_or(SyscallError::EBADF)?;

    // Check if writable
    let access = file.flags & 3;
    if access == O_RDONLY {
        return Err(SyscallError::EBADF);
    }

    let bytes_written = vfs.root.write(file.ino, file.offset, buf)?;
    file.offset += bytes_written;

    Ok(bytes_written)
}

/// Remove (unlink) a file.
pub fn unlink(path: &str) -> Result<(), SyscallError> {
    let mut vfs = VFS.lock();
    let vfs = vfs.as_mut().ok_or(SyscallError::EIO)?;

    let (parent_path, name) = split_path(path);

    let parent_ino = vfs.root.lookup(parent_path)
        .ok_or(SyscallError::ENOENT)?;

    // Get the inode to remove
    let ino = vfs.root.lookup(path)
        .ok_or(SyscallError::ENOENT)?;

    // Check if it's a directory
    let metadata = vfs.root.stat(ino)?;
    if metadata.file_type() == FileType::Directory {
        return Err(SyscallError::EISDIR);
    }

    // Remove from parent directory
    if let Some(parent) = vfs.root.inodes.get_mut(&parent_ino) {
        if let RamInodeData::Directory(entries) = &mut parent.data {
            entries.remove(name);
        }
    }

    // Remove inode
    vfs.root.inodes.remove(&ino);

    Ok(())
}

/// Check if metadata represents a directory.
pub fn is_directory(metadata: &InodeMetadata) -> bool {
    metadata.file_type() == FileType::Directory
}

/// Create a device node.
pub fn mknod(path: &str, mode: u32, _dev_major: u32, _dev_minor: u32) -> Result<(), SyscallError> {
    let mut vfs = VFS.lock();
    let vfs = vfs.as_mut().ok_or(SyscallError::EIO)?;

    let (parent_path, name) = split_path(path);

    let parent_ino = vfs.root.lookup(parent_path)
        .ok_or(SyscallError::ENOENT)?;

    let parent = vfs.root.inodes.get_mut(&parent_ino)
        .ok_or(SyscallError::ENOENT)?;

    match &mut parent.data {
        RamInodeData::Directory(entries) => {
            if entries.contains_key(name) {
                return Err(SyscallError::EEXIST);
            }

            let ino = vfs.root.alloc_ino();
            entries.insert(String::from(name), ino);

            // Determine file type from mode
            let file_type = if mode & S_IFCHR != 0 {
                FileType::CharDevice
            } else if mode & S_IFBLK != 0 {
                FileType::BlockDevice
            } else if mode & S_IFIFO != 0 {
                FileType::Fifo
            } else {
                FileType::Regular
            };

            let metadata = InodeMetadata::new(ino, file_type);
            vfs.root.inodes.insert(ino, RamInode {
                metadata,
                data: RamInodeData::File(Vec::new()),
            });

            Ok(())
        }
        _ => Err(SyscallError::ENOTDIR),
    }
}

/// Get directory entry info (for ls -l style output).
pub struct DirEntryInfo {
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
    pub mode: u32,
}

/// Read directory with full info.
pub fn readdir_full(path: &str) -> Result<Vec<DirEntryInfo>, SyscallError> {
    let vfs = VFS.lock();
    let vfs = vfs.as_ref().ok_or(SyscallError::EIO)?;
    let ino = vfs.root.lookup(path).ok_or(SyscallError::ENOENT)?;

    let inode = vfs.root.inodes.get(&ino).ok_or(SyscallError::ENOENT)?;

    match &inode.data {
        RamInodeData::Directory(entries) => {
            let mut result = Vec::new();
            for (name, &child_ino) in entries {
                if name == "." || name == ".." {
                    continue;
                }
                let child = vfs.root.inodes.get(&child_ino)
                    .ok_or(SyscallError::EIO)?;
                result.push(DirEntryInfo {
                    name: name.clone(),
                    is_dir: child.metadata.file_type() == FileType::Directory,
                    size: child.metadata.size,
                    mode: child.metadata.mode,
                });
            }
            Ok(result)
        }
        _ => Err(SyscallError::ENOTDIR),
    }
}
