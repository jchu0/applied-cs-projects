# Container Runtime (Docker-lite) - Technical Blueprint

## Executive Summary

This project implements a minimal yet production-capable container runtime that demonstrates deep understanding of Linux kernel primitives, OCI specifications, and systems programming. The runtime provides process isolation through namespaces, resource constraints via cgroups, and layered filesystem support through OverlayFS.

**Primary Goals:**
- Build a compliant OCI runtime from scratch
- Demonstrate mastery of Linux kernel isolation primitives
- Implement production-grade image management with layer caching
- Provide secure, rootless container execution

---

## System Architecture

### High-Level Architecture

```
+------------------+     +-------------------+     +------------------+
|   CLI Interface  |---->|  Container Engine |---->|  Runtime Shim    |
|   (OCI-compliant)|     |   (orchestration) |     |  (runc-like)     |
+------------------+     +-------------------+     +------------------+
                                   |                        |
                    +--------------+--------------+         |
                    |              |              |         |
             +------v----+  +------v----+  +------v----+    |
             |   Image   |  | Storage   |  | Network   |    |
             |   Store   |  |  Driver   |  |  Manager  |    |
             +-----------+  +-----------+  +-----------+    |
                                                           |
                    +--------------------------------------+
                    |
         +----------v-----------+
         |   Linux Kernel APIs  |
         |  (namespaces/cgroups)|
         +----------------------+
```

### Component Breakdown

#### 1. Image Loader & Registry Client

```rust
pub struct ImageStore {
    root_path: PathBuf,
    manifest_cache: HashMap<ImageRef, Manifest>,
    layer_cache: LayerCache,
    registry_client: RegistryClient,
}

pub struct Manifest {
    schema_version: u32,
    media_type: String,
    config: Descriptor,
    layers: Vec<Descriptor>,
    annotations: HashMap<String, String>,
}

pub struct LayerCache {
    layers: HashMap<Digest, LayerMetadata>,
    refs: HashMap<Digest, usize>,  // Reference counting
    total_size: u64,
    max_size: u64,
}

pub struct Descriptor {
    media_type: String,
    digest: Digest,
    size: u64,
    urls: Vec<String>,
    annotations: HashMap<String, String>,
    platform: Option<Platform>,
}
```

**Layer Handling Pipeline:**
1. Parse image manifest (OCI Image Manifest Specification)
2. Resolve layer chain with deduplication
3. Download missing layers with resumable transfers
4. Verify content-addressable integrity (SHA256)
5. Extract and cache layer tarballs
6. Build OverlayFS mount chain

#### 2. Runtime Engine

```rust
pub struct Runtime {
    config: RuntimeConfig,
    container_store: ContainerStore,
    cgroup_manager: CgroupManager,
    network_manager: NetworkManager,
}

pub struct Container {
    id: ContainerId,
    state: ContainerState,
    config: ContainerConfig,
    bundle_path: PathBuf,
    rootfs: PathBuf,
    namespaces: NamespaceSet,
    cgroup_path: PathBuf,
    created_at: SystemTime,
    pid: Option<Pid>,
}

pub enum ContainerState {
    Creating,
    Created,
    Running,
    Stopped,
    Paused,
}

pub struct ContainerConfig {
    process: ProcessSpec,
    root: RootSpec,
    hostname: String,
    mounts: Vec<Mount>,
    linux: LinuxSpec,
    hooks: Option<Hooks>,
}
```

#### 3. Namespace Management

```rust
pub struct NamespaceSet {
    user: Option<NamespaceHandle>,
    mount: Option<NamespaceHandle>,
    pid: Option<NamespaceHandle>,
    network: Option<NamespaceHandle>,
    ipc: Option<NamespaceHandle>,
    uts: Option<NamespaceHandle>,
    cgroup: Option<NamespaceHandle>,
}

impl NamespaceSet {
    pub fn create_all(&mut self, config: &LinuxSpec) -> Result<()> {
        // Order matters: user namespace must be first for rootless
        if config.namespaces.contains(&NamespaceType::User) {
            self.user = Some(self.create_user_namespace(&config.uid_mappings, &config.gid_mappings)?);
        }

        // Mount namespace for isolated filesystem view
        if config.namespaces.contains(&NamespaceType::Mount) {
            self.mount = Some(self.create_mount_namespace()?);
        }

        // PID namespace for process isolation
        if config.namespaces.contains(&NamespaceType::Pid) {
            self.pid = Some(self.create_pid_namespace()?);
        }

        // Network namespace for network isolation
        if config.namespaces.contains(&NamespaceType::Network) {
            self.network = Some(self.create_network_namespace()?);
        }

        // IPC namespace for System V IPC isolation
        if config.namespaces.contains(&NamespaceType::Ipc) {
            self.ipc = Some(self.create_ipc_namespace()?);
        }

        // UTS namespace for hostname isolation
        if config.namespaces.contains(&NamespaceType::Uts) {
            self.uts = Some(self.create_uts_namespace()?);
        }

        // Cgroup namespace for cgroup view isolation
        if config.namespaces.contains(&NamespaceType::Cgroup) {
            self.cgroup = Some(self.create_cgroup_namespace()?);
        }

        Ok(())
    }
}
```

#### 4. Cgroup Management (v2 unified hierarchy)

```rust
pub struct CgroupManager {
    root_path: PathBuf,
    controllers: HashSet<Controller>,
}

pub struct CgroupConfig {
    memory: Option<MemoryConfig>,
    cpu: Option<CpuConfig>,
    io: Option<IoConfig>,
    pids: Option<PidsConfig>,
}

pub struct MemoryConfig {
    limit: u64,
    reservation: u64,
    swap: u64,
    kernel: u64,
    oom_kill_disable: bool,
}

pub struct CpuConfig {
    shares: u64,
    quota: i64,
    period: u64,
    cpus: String,  // CPU affinity mask
    mems: String,  // NUMA memory nodes
}

impl CgroupManager {
    pub fn create_cgroup(&self, container_id: &str, config: &CgroupConfig) -> Result<PathBuf> {
        let cgroup_path = self.root_path.join(container_id);
        fs::create_dir_all(&cgroup_path)?;

        // Configure memory controller
        if let Some(mem) = &config.memory {
            fs::write(cgroup_path.join("memory.max"), mem.limit.to_string())?;
            fs::write(cgroup_path.join("memory.low"), mem.reservation.to_string())?;
            fs::write(cgroup_path.join("memory.swap.max"), mem.swap.to_string())?;
        }

        // Configure CPU controller
        if let Some(cpu) = &config.cpu {
            fs::write(cgroup_path.join("cpu.weight"), cpu.shares.to_string())?;
            if cpu.quota > 0 {
                let max = format!("{} {}", cpu.quota, cpu.period);
                fs::write(cgroup_path.join("cpu.max"), max)?;
            }
            fs::write(cgroup_path.join("cpuset.cpus"), &cpu.cpus)?;
            fs::write(cgroup_path.join("cpuset.mems"), &cpu.mems)?;
        }

        // Configure PIDs controller
        if let Some(pids) = &config.pids {
            fs::write(cgroup_path.join("pids.max"), pids.max.to_string())?;
        }

        Ok(cgroup_path)
    }

    pub fn attach_process(&self, cgroup_path: &Path, pid: Pid) -> Result<()> {
        fs::write(cgroup_path.join("cgroup.procs"), pid.to_string())?;
        Ok(())
    }
}
```

---

## Core Internals

### Container Lifecycle

```
create() -> start() -> [running] -> stop() -> delete()
    |         |           |           |          |
    v         v           v           v          v
  Bundle   Clone()    Exec()      SIGTERM    Cleanup
  Prep     Namespaces  Process    SIGKILL    Resources
```

### Process Execution Flow

```rust
pub fn spawn_container_process(config: &ContainerConfig, namespaces: &NamespaceSet) -> Result<Pid> {
    // 1. Set up pre-clone environment
    let (parent_pipe, child_pipe) = create_pipe()?;

    // 2. Clone with namespace flags
    let clone_flags = compute_clone_flags(&namespaces);
    let pid = unsafe {
        clone(
            Box::new(move || container_init(child_pipe, config)),
            &mut stack,
            clone_flags,
            None,
        )?
    };

    // 3. Parent: set up UID/GID mappings if user namespace
    if namespaces.user.is_some() {
        write_uid_gid_mappings(pid, &config.linux.uid_mappings, &config.linux.gid_mappings)?;
    }

    // 4. Parent: attach to cgroup before child proceeds
    cgroup_manager.attach_process(&cgroup_path, pid)?;

    // 5. Parent: set up network (veth pair)
    if namespaces.network.is_some() {
        network_manager.setup_container_network(pid, &config.network)?;
    }

    // 6. Signal child to proceed
    write_to_pipe(&parent_pipe, &[1])?;

    Ok(pid)
}

fn container_init(pipe: RawFd, config: &ContainerConfig) -> isize {
    // Wait for parent to complete setup
    let mut buf = [0u8; 1];
    read_from_pipe(pipe, &mut buf).unwrap();

    // Set up rootfs
    setup_rootfs(&config.root).unwrap();

    // Pivot root
    pivot_root(&config.root.path).unwrap();

    // Set up mounts
    for mount in &config.mounts {
        setup_mount(mount).unwrap();
    }

    // Drop capabilities
    drop_capabilities(&config.process.capabilities).unwrap();

    // Set seccomp filter
    if let Some(seccomp) = &config.linux.seccomp {
        apply_seccomp_filter(seccomp).unwrap();
    }

    // Execute target process
    let args: Vec<CString> = config.process.args.iter()
        .map(|s| CString::new(s.as_str()).unwrap())
        .collect();
    let env: Vec<CString> = config.process.env.iter()
        .map(|s| CString::new(s.as_str()).unwrap())
        .collect();

    execve(&args[0], &args, &env).unwrap();

    0
}
```

### OverlayFS Mount Strategy

```rust
pub struct OverlayMount {
    lower_dirs: Vec<PathBuf>,  // Image layers (read-only)
    upper_dir: PathBuf,        // Container writable layer
    work_dir: PathBuf,         // OverlayFS work directory
    merged_dir: PathBuf,       // Union mount point
}

impl OverlayMount {
    pub fn mount(&self) -> Result<()> {
        // Construct overlay mount options
        let lower = self.lower_dirs.iter()
            .map(|p| p.to_string_lossy())
            .collect::<Vec<_>>()
            .join(":");

        let options = format!(
            "lowerdir={},upperdir={},workdir={}",
            lower,
            self.upper_dir.display(),
            self.work_dir.display()
        );

        // Create mount point
        fs::create_dir_all(&self.merged_dir)?;

        // Mount overlay filesystem
        nix::mount::mount(
            Some("overlay"),
            &self.merged_dir,
            Some("overlay"),
            nix::mount::MsFlags::empty(),
            Some(options.as_str()),
        )?;

        Ok(())
    }
}
```

### Networking Architecture

```rust
pub struct NetworkManager {
    bridge_name: String,
    subnet: IpNetwork,
    allocator: IpAllocator,
}

pub struct ContainerNetwork {
    veth_host: String,
    veth_container: String,
    ip_address: IpAddr,
    gateway: IpAddr,
    mac_address: MacAddr,
}

impl NetworkManager {
    pub fn setup_container_network(&self, pid: Pid, config: &NetworkConfig) -> Result<ContainerNetwork> {
        let container_id = &config.container_id[..12];

        // 1. Create veth pair
        let veth_host = format!("veth{}", container_id);
        let veth_container = "eth0".to_string();

        netlink::create_veth_pair(&veth_host, &veth_container)?;

        // 2. Move container end to network namespace
        let netns_path = format!("/proc/{}/ns/net", pid);
        netlink::set_link_netns(&veth_container, &netns_path)?;

        // 3. Attach host end to bridge
        netlink::set_link_master(&veth_host, &self.bridge_name)?;
        netlink::set_link_up(&veth_host)?;

        // 4. Configure container interface
        let ip = self.allocator.allocate()?;
        in_namespace(&netns_path, || {
            netlink::add_address(&veth_container, ip, self.subnet.prefix())?;
            netlink::set_link_up(&veth_container)?;
            netlink::add_default_route(self.subnet.network())?;
            Ok(())
        })?;

        // 5. Set up NAT for outbound traffic
        iptables::add_masquerade(&self.subnet)?;

        Ok(ContainerNetwork {
            veth_host,
            veth_container,
            ip_address: ip,
            gateway: self.subnet.network(),
            mac_address: generate_mac(),
        })
    }
}
```

---

## Data Structures

### OCI Runtime Specification Structures

```rust
/// OCI Runtime Specification v1.0
pub struct Spec {
    pub oci_version: String,
    pub root: Root,
    pub mounts: Vec<Mount>,
    pub process: Process,
    pub hostname: String,
    pub linux: Linux,
    pub hooks: Option<Hooks>,
    pub annotations: HashMap<String, String>,
}

pub struct Root {
    pub path: PathBuf,
    pub readonly: bool,
}

pub struct Mount {
    pub destination: PathBuf,
    pub mount_type: String,
    pub source: PathBuf,
    pub options: Vec<String>,
}

pub struct Process {
    pub terminal: bool,
    pub console_size: Option<ConsoleSize>,
    pub user: User,
    pub args: Vec<String>,
    pub env: Vec<String>,
    pub cwd: PathBuf,
    pub capabilities: Option<LinuxCapabilities>,
    pub rlimits: Vec<PosixRlimit>,
    pub no_new_privileges: bool,
    pub oom_score_adj: Option<i32>,
}

pub struct Linux {
    pub namespaces: Vec<LinuxNamespace>,
    pub uid_mappings: Vec<LinuxIdMapping>,
    pub gid_mappings: Vec<LinuxIdMapping>,
    pub devices: Vec<LinuxDevice>,
    pub cgroups_path: String,
    pub resources: Option<LinuxResources>,
    pub seccomp: Option<LinuxSeccomp>,
    pub rootfs_propagation: String,
    pub masked_paths: Vec<String>,
    pub readonly_paths: Vec<String>,
}
```

### Container State Machine

```rust
pub struct StateMachine {
    current: ContainerState,
    transitions: HashMap<(ContainerState, Event), ContainerState>,
}

pub enum Event {
    CreateComplete,
    StartRequested,
    ProcessStarted,
    ProcessExited(i32),
    StopRequested,
    KillRequested,
    DeleteRequested,
    PauseRequested,
    ResumeRequested,
}

impl StateMachine {
    pub fn new() -> Self {
        let mut transitions = HashMap::new();

        // Creating -> Created
        transitions.insert((ContainerState::Creating, Event::CreateComplete), ContainerState::Created);

        // Created -> Running
        transitions.insert((ContainerState::Created, Event::StartRequested), ContainerState::Running);

        // Running -> Stopped
        transitions.insert((ContainerState::Running, Event::ProcessExited(0)), ContainerState::Stopped);
        transitions.insert((ContainerState::Running, Event::StopRequested), ContainerState::Stopped);
        transitions.insert((ContainerState::Running, Event::KillRequested), ContainerState::Stopped);

        // Running -> Paused
        transitions.insert((ContainerState::Running, Event::PauseRequested), ContainerState::Paused);

        // Paused -> Running
        transitions.insert((ContainerState::Paused, Event::ResumeRequested), ContainerState::Running);

        Self { current: ContainerState::Creating, transitions }
    }
}
```

---

## API Design

### OCI-Compliant CLI Interface

```bash
# Container lifecycle
docklet create <container-id> --bundle <bundle-path>
docklet start <container-id>
docklet run <container-id> --bundle <bundle-path>
docklet kill <container-id> [--signal <signal>]
docklet delete <container-id> [--force]

# Container state
docklet state <container-id>
docklet list [--all] [--format <format>]
docklet ps [--all] [--quiet]

# Image management
docklet pull <image-ref>
docklet images [--all]
docklet rmi <image-id>
docklet save <image-ref> -o <output.tar>
docklet load -i <input.tar>

# Execution
docklet exec <container-id> <command> [args...]
docklet attach <container-id>
docklet logs <container-id> [--follow] [--tail <n>]

# Networking
docklet network create <name> [--subnet <cidr>]
docklet network connect <network> <container>
docklet network disconnect <network> <container>
```

### Programmatic API

```rust
pub trait ContainerRuntime {
    /// Create a new container from an OCI bundle
    fn create(&self, id: &str, bundle: &Path, options: CreateOptions) -> Result<Container>;

    /// Start a created container
    fn start(&self, id: &str) -> Result<()>;

    /// Send a signal to a running container
    fn kill(&self, id: &str, signal: Signal) -> Result<()>;

    /// Delete a stopped container
    fn delete(&self, id: &str, force: bool) -> Result<()>;

    /// Get container state
    fn state(&self, id: &str) -> Result<ContainerState>;

    /// List all containers
    fn list(&self, filters: ListFilters) -> Result<Vec<ContainerSummary>>;

    /// Execute a process inside a running container
    fn exec(&self, id: &str, process: &Process, options: ExecOptions) -> Result<ExecResult>;

    /// Attach to container's stdio
    fn attach(&self, id: &str, options: AttachOptions) -> Result<AttachHandle>;
}

pub struct CreateOptions {
    pub console_socket: Option<PathBuf>,
    pub pid_file: Option<PathBuf>,
    pub no_pivot: bool,
    pub no_new_keyring: bool,
    pub rootless: bool,
}

pub struct ExecOptions {
    pub tty: bool,
    pub detach: bool,
    pub user: Option<String>,
    pub env: Vec<String>,
    pub cwd: Option<PathBuf>,
}
```

### Logging Driver Interface

```rust
pub trait LogDriver: Send + Sync {
    fn name(&self) -> &str;
    fn write(&self, entry: &LogEntry) -> Result<()>;
    fn read(&self, options: ReadOptions) -> Result<LogStream>;
    fn close(&self) -> Result<()>;
}

pub struct LogEntry {
    pub timestamp: SystemTime,
    pub stream: Stream,  // Stdout or Stderr
    pub message: Vec<u8>,
    pub partial: bool,
    pub attributes: HashMap<String, String>,
}

// Built-in drivers
pub struct JsonFileDriver { /* ... */ }
pub struct JournaldDriver { /* ... */ }
pub struct SyslogDriver { /* ... */ }
pub struct FluentdDriver { /* ... */ }
```

---

## Enterprise Features

### 1. Rootless Container Execution

```rust
pub struct RootlessConfig {
    pub uid_map: Vec<IdMapping>,
    pub gid_map: Vec<IdMapping>,
    pub subuid_path: PathBuf,
    pub subgid_path: PathBuf,
}

impl RootlessRuntime {
    pub fn setup_user_namespace(&self, config: &RootlessConfig) -> Result<()> {
        // Parse /etc/subuid and /etc/subgid
        let (subuid_start, subuid_count) = parse_subid(&config.subuid_path, getuid())?;
        let (subgid_start, subgid_count) = parse_subid(&config.subgid_path, getgid())?;

        // Create user namespace
        unshare(CloneFlags::CLONE_NEWUSER)?;

        // Write UID mapping: map container root to host user
        let uid_map = format!("0 {} 1\n1 {} {}", getuid(), subuid_start, subuid_count);
        fs::write("/proc/self/uid_map", uid_map)?;

        // Disable setgroups for unprivileged users
        fs::write("/proc/self/setgroups", "deny")?;

        // Write GID mapping
        let gid_map = format!("0 {} 1\n1 {} {}", getgid(), subgid_start, subgid_count);
        fs::write("/proc/self/gid_map", gid_map)?;

        Ok(())
    }

    pub fn setup_rootless_network(&self) -> Result<()> {
        // Use slirp4netns for rootless networking
        let slirp = Command::new("slirp4netns")
            .arg("--configure")
            .arg("--mtu=65520")
            .arg("--disable-host-loopback")
            .arg(format!("{}", getpid()))
            .arg("tap0")
            .spawn()?;

        Ok(())
    }
}
```

### 2. Advanced Logging Drivers

```rust
pub struct LoggingConfig {
    pub driver: String,
    pub options: HashMap<String, String>,
    pub max_size: u64,
    pub max_file: u32,
    pub compress: bool,
    pub labels: Vec<String>,
    pub env: Vec<String>,
}

impl JsonFileDriver {
    pub fn new(config: &LoggingConfig) -> Result<Self> {
        let log_path = config.options.get("path")
            .ok_or_else(|| Error::MissingOption("path"))?;

        Ok(Self {
            file: OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_path)?,
            max_size: config.max_size,
            max_file: config.max_file,
            compress: config.compress,
            current_size: 0,
        })
    }

    fn rotate_if_needed(&mut self) -> Result<()> {
        if self.current_size >= self.max_size {
            // Rotate log files
            for i in (1..self.max_file).rev() {
                let from = format!("{}.{}", self.path, i);
                let to = format!("{}.{}", self.path, i + 1);
                if Path::new(&from).exists() {
                    if self.compress && i == self.max_file - 1 {
                        compress_file(&from)?;
                    }
                    fs::rename(&from, &to)?;
                }
            }
            fs::rename(&self.path, format!("{}.1", self.path))?;
            self.file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)?;
            self.current_size = 0;
        }
        Ok(())
    }
}
```

### 3. Seccomp Security Profiles

```rust
pub struct SeccompProfile {
    pub default_action: Action,
    pub architectures: Vec<Arch>,
    pub syscalls: Vec<SyscallRule>,
}

pub struct SyscallRule {
    pub names: Vec<String>,
    pub action: Action,
    pub args: Vec<ArgCondition>,
}

impl SeccompProfile {
    pub fn compile(&self) -> Result<BpfProgram> {
        let mut builder = SeccompBuilder::new(self.default_action);

        for arch in &self.architectures {
            builder.add_architecture(*arch);
        }

        for rule in &self.syscalls {
            for name in &rule.names {
                let syscall_nr = resolve_syscall_number(name)?;

                if rule.args.is_empty() {
                    builder.add_rule(syscall_nr, rule.action)?;
                } else {
                    for arg in &rule.args {
                        builder.add_rule_with_arg(syscall_nr, rule.action, arg)?;
                    }
                }
            }
        }

        builder.build()
    }

    pub fn default_profile() -> Self {
        // Deny dangerous syscalls by default
        let mut profile = Self {
            default_action: Action::Allow,
            architectures: vec![Arch::X86_64],
            syscalls: vec![],
        };

        // Block kernel module operations
        profile.deny_syscalls(&["init_module", "finit_module", "delete_module"]);

        // Block mount operations (handled by runtime)
        profile.deny_syscalls(&["mount", "umount2", "pivot_root"]);

        // Block raw I/O
        profile.deny_syscalls(&["iopl", "ioperm"]);

        // Block clock modifications
        profile.deny_syscalls(&["settimeofday", "clock_settime", "clock_adjtime"]);

        profile
    }
}
```

---

## Performance Considerations

### Layer Caching Strategy

```rust
pub struct LayerCacheConfig {
    pub max_size: u64,
    pub gc_policy: GcPolicy,
    pub dedup_enabled: bool,
}

pub enum GcPolicy {
    Lru,               // Least recently used
    Fifo,              // First in, first out
    SizeWeighted,      // Consider both size and age
}

impl LayerCache {
    pub fn gc(&mut self) -> Result<u64> {
        let mut freed = 0u64;

        while self.total_size > self.max_size {
            let victim = match self.gc_policy {
                GcPolicy::Lru => self.find_lru_layer(),
                GcPolicy::Fifo => self.find_oldest_layer(),
                GcPolicy::SizeWeighted => self.find_size_weighted_victim(),
            }?;

            if self.refs.get(&victim.digest).copied().unwrap_or(0) == 0 {
                freed += victim.size;
                self.remove_layer(&victim.digest)?;
            } else {
                break;  // Can't free referenced layers
            }
        }

        Ok(freed)
    }
}
```

### Parallel Layer Downloads

```rust
pub async fn pull_image(&self, image_ref: &ImageRef) -> Result<Image> {
    let manifest = self.registry_client.get_manifest(image_ref).await?;

    // Filter already cached layers
    let missing_layers: Vec<_> = manifest.layers.iter()
        .filter(|l| !self.layer_cache.contains(&l.digest))
        .collect();

    // Download layers in parallel with bounded concurrency
    let semaphore = Arc::new(Semaphore::new(4));  // Max 4 concurrent downloads
    let downloads: Vec<_> = missing_layers.iter()
        .map(|layer| {
            let sem = semaphore.clone();
            let client = self.registry_client.clone();
            let cache = self.layer_cache.clone();

            async move {
                let _permit = sem.acquire().await?;
                let data = client.get_blob(&layer.digest).await?;
                cache.store(&layer.digest, data).await?;
                Ok::<_, Error>(())
            }
        })
        .collect();

    futures::future::try_join_all(downloads).await?;

    Ok(Image::from_manifest(manifest))
}
```

### Copy-on-Write Optimization

```rust
// Use reflinks when available for instant layer copies
pub fn copy_layer(&self, src: &Path, dst: &Path) -> Result<()> {
    // Try reflink first (instant, no data copy)
    match reflink::reflink(src, dst) {
        Ok(()) => return Ok(()),
        Err(e) if e.kind() == io::ErrorKind::Unsupported => {
            // Fall back to regular copy
        }
        Err(e) => return Err(e.into()),
    }

    // Regular copy with sparse file support
    let src_file = File::open(src)?;
    let dst_file = File::create(dst)?;

    // Use sendfile for efficient kernel-space copy
    let len = src_file.metadata()?.len();
    let mut offset = 0;
    while offset < len {
        let copied = nix::sys::sendfile::sendfile(
            dst_file.as_raw_fd(),
            src_file.as_raw_fd(),
            Some(&mut (offset as i64)),
            (len - offset) as usize,
        )?;
        offset += copied as u64;
    }

    Ok(())
}
```

---

## Stretch Goals

### 1. CRI (Container Runtime Interface) Shim

```rust
/// Kubernetes CRI implementation
pub struct CriService {
    runtime: Arc<Runtime>,
    image_service: Arc<ImageService>,
    streaming_server: StreamingServer,
}

#[tonic::async_trait]
impl RuntimeService for CriService {
    async fn run_pod_sandbox(
        &self,
        request: Request<RunPodSandboxRequest>,
    ) -> Result<Response<RunPodSandboxResponse>, Status> {
        let config = request.into_inner().config.unwrap();

        // Create pod network namespace
        let netns = self.create_pod_netns(&config)?;

        // Create pause container
        let sandbox_id = generate_id();
        let pause_container = self.runtime.create(
            &sandbox_id,
            &self.create_pause_bundle(&config)?,
            CreateOptions::default(),
        )?;

        self.runtime.start(&sandbox_id)?;

        Ok(Response::new(RunPodSandboxResponse {
            pod_sandbox_id: sandbox_id,
        }))
    }

    async fn create_container(
        &self,
        request: Request<CreateContainerRequest>,
    ) -> Result<Response<CreateContainerResponse>, Status> {
        let req = request.into_inner();
        let pod_id = req.pod_sandbox_id;
        let config = req.config.unwrap();

        // Join pod's namespaces
        let pod = self.get_pod(&pod_id)?;
        let container_id = generate_id();

        let spec = self.create_container_spec(&config, &pod)?;
        let container = self.runtime.create(
            &container_id,
            &spec,
            CreateOptions {
                join_namespaces: vec![
                    pod.netns.clone(),
                    pod.ipcns.clone(),
                ],
                ..Default::default()
            },
        )?;

        Ok(Response::new(CreateContainerResponse {
            container_id,
        }))
    }
}
```

### 2. Multi-Architecture Support

```rust
pub struct MultiArchImage {
    index: ImageIndex,
    platforms: HashMap<Platform, Manifest>,
}

pub struct Platform {
    pub architecture: String,
    pub os: String,
    pub variant: Option<String>,
}

impl MultiArchImage {
    pub fn resolve_for_host(&self) -> Result<&Manifest> {
        let host_platform = Platform {
            architecture: std::env::consts::ARCH.to_string(),
            os: std::env::consts::OS.to_string(),
            variant: detect_cpu_variant(),
        };

        self.platforms.get(&host_platform)
            .or_else(|| {
                // Try without variant
                let generic = Platform { variant: None, ..host_platform };
                self.platforms.get(&generic)
            })
            .ok_or_else(|| Error::PlatformNotFound(host_platform))
    }
}

fn detect_cpu_variant() -> Option<String> {
    #[cfg(target_arch = "arm")]
    {
        // Detect ARM variant (v6, v7, v8)
        let cpuinfo = fs::read_to_string("/proc/cpuinfo").ok()?;
        if cpuinfo.contains("ARMv7") {
            Some("v7".to_string())
        } else if cpuinfo.contains("ARMv6") {
            Some("v6".to_string())
        } else {
            None
        }
    }

    #[cfg(not(target_arch = "arm"))]
    None
}
```

---

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlay_mount_options() {
        let overlay = OverlayMount {
            lower_dirs: vec![
                PathBuf::from("/var/lib/docklet/layers/abc"),
                PathBuf::from("/var/lib/docklet/layers/def"),
            ],
            upper_dir: PathBuf::from("/var/lib/docklet/upper/123"),
            work_dir: PathBuf::from("/var/lib/docklet/work/123"),
            merged_dir: PathBuf::from("/var/lib/docklet/merged/123"),
        };

        let options = overlay.build_options();
        assert!(options.contains("lowerdir=/var/lib/docklet/layers/abc:/var/lib/docklet/layers/def"));
    }

    #[test]
    fn test_cgroup_config_serialization() {
        let config = CgroupConfig {
            memory: Some(MemoryConfig {
                limit: 512 * 1024 * 1024,
                reservation: 256 * 1024 * 1024,
                swap: 0,
                kernel: 0,
                oom_kill_disable: false,
            }),
            cpu: Some(CpuConfig {
                shares: 1024,
                quota: 50000,
                period: 100000,
                cpus: "0-3".to_string(),
                mems: "0".to_string(),
            }),
            io: None,
            pids: Some(PidsConfig { max: 100 }),
        };

        assert_eq!(config.cpu.as_ref().unwrap().cpu_percent(), 50);
    }
}
```

### Integration Tests

```rust
#[test]
fn test_container_lifecycle() {
    let runtime = Runtime::new(test_config()).unwrap();

    // Create container
    let container = runtime.create(
        "test-container",
        &test_bundle_path(),
        CreateOptions::default(),
    ).unwrap();

    assert_eq!(runtime.state("test-container").unwrap(), ContainerState::Created);

    // Start container
    runtime.start("test-container").unwrap();
    assert_eq!(runtime.state("test-container").unwrap(), ContainerState::Running);

    // Execute command
    let result = runtime.exec(
        "test-container",
        &Process {
            args: vec!["echo".to_string(), "hello".to_string()],
            ..Default::default()
        },
        ExecOptions::default(),
    ).unwrap();

    assert_eq!(result.exit_code, 0);
    assert_eq!(result.stdout.trim(), "hello");

    // Stop and delete
    runtime.kill("test-container", Signal::SIGTERM).unwrap();
    runtime.delete("test-container", false).unwrap();
}

#[test]
fn test_network_isolation() {
    let runtime = Runtime::new(test_config()).unwrap();

    // Create two containers on the same network
    let c1 = runtime.create("c1", &bundle1, CreateOptions::default()).unwrap();
    let c2 = runtime.create("c2", &bundle2, CreateOptions::default()).unwrap();

    runtime.start("c1").unwrap();
    runtime.start("c2").unwrap();

    // Verify network connectivity
    let ping_result = runtime.exec(
        "c1",
        &Process {
            args: vec!["ping".to_string(), "-c".to_string(), "1".to_string(), c2.ip_address.to_string()],
            ..Default::default()
        },
        ExecOptions::default(),
    ).unwrap();

    assert_eq!(ping_result.exit_code, 0);

    runtime.delete("c1", true).unwrap();
    runtime.delete("c2", true).unwrap();
}
```

### Conformance Tests

```rust
// OCI Runtime Conformance Test Suite
#[test]
fn test_oci_runtime_create_validation() {
    // Test that invalid bundles are rejected
    let runtime = Runtime::new(test_config()).unwrap();

    // Missing config.json
    let result = runtime.create("test", &empty_bundle, CreateOptions::default());
    assert!(matches!(result, Err(Error::InvalidBundle(_))));

    // Invalid OCI version
    let result = runtime.create("test", &invalid_version_bundle, CreateOptions::default());
    assert!(matches!(result, Err(Error::UnsupportedOciVersion(_))));
}
```

---

## Implementation Phases

### Phase 1: Core Runtime (Weeks 1-3)
- Basic namespace creation (mount, pid, uts)
- Simple rootfs setup with chroot
- Process spawning with clone()
- Basic cgroup v2 support (memory, cpu)

### Phase 2: Image Management (Weeks 4-5)
- OCI image manifest parsing
- Layer download and extraction
- Content-addressable storage
- OverlayFS mounting

### Phase 3: Networking (Weeks 6-7)
- Bridge network creation
- Veth pair setup
- IP address allocation
- NAT configuration with iptables

### Phase 4: Enterprise Features (Weeks 8-10)
- Rootless mode with user namespaces
- Logging drivers
- Seccomp profiles
- Full OCI CLI compliance

### Phase 5: Stretch Goals (Weeks 11-12)
- CRI shim for Kubernetes
- Multi-architecture support
- Performance optimization

---

## Security Considerations

1. **Capability Dropping**: Drop all capabilities except those explicitly required
2. **Seccomp Filtering**: Apply restrictive syscall filters
3. **User Namespaces**: Map container root to unprivileged host user
4. **Read-only Rootfs**: Mount container filesystem read-only by default
5. **No New Privileges**: Prevent privilege escalation through setuid binaries
6. **Resource Limits**: Enforce memory, CPU, and PID limits via cgroups

---

## References

- [OCI Runtime Specification](https://github.com/opencontainers/runtime-spec)
- [OCI Image Format Specification](https://github.com/opencontainers/image-spec)
- [Linux Namespaces man pages](https://man7.org/linux/man-pages/man7/namespaces.7.html)
- [Cgroup v2 Documentation](https://www.kernel.org/doc/Documentation/cgroup-v2.txt)
- [runc Source Code](https://github.com/opencontainers/runc)
