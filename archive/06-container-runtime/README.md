# Docklet - A Minimal Container Runtime

A Docker-lite container runtime implementing OCI container runtime specification using Linux kernel primitives for process isolation.

## Features

### Core Capabilities
- **Linux Namespaces**: PID, network, mount, UTS, IPC, user, and cgroup namespaces
- **Cgroup v2**: Resource limits for memory, CPU, PIDs, and I/O
- **OverlayFS**: Copy-on-write filesystem layers
- **OCI Spec**: Compatible with OCI Runtime Specification

### Architecture

```
┌─────────────────────────────────────────────┐
│              CLI (main.rs)                   │
├─────────────────────────────────────────────┤
│            Runtime (runtime.rs)              │
├──────┬──────┬──────┬──────┬────────┬────────┤
│ Spec │Cgroup│ Proc │ NS   │Overlay │ Image  │
└──────┴──────┴──────┴──────┴────────┴────────┘
```

## Modules

- **spec.rs**: OCI Runtime Specification types
- **runtime.rs**: Container lifecycle management
- **namespace.rs**: Linux namespace operations
- **cgroup.rs**: Cgroup v2 resource management
- **process.rs**: Process execution and setup
- **overlay.rs**: OverlayFS layer management
- **image.rs**: Container image handling
- **container.rs**: Container state machine
- **error.rs**: Error types

## Usage

```bash
# Generate a default spec
docklet spec --bundle ./mycontainer

# Create a container
docklet create mycontainer --bundle ./mycontainer

# Start the container
docklet start mycontainer

# Or create and run in one command
docklet run mycontainer --bundle ./mycontainer

# Check container state
docklet state mycontainer

# List containers
docklet list

# Send signal to container
docklet kill mycontainer SIGTERM

# Delete container
docklet delete mycontainer
```

## OCI Runtime Spec (config.json)

```json
{
  "ociVersion": "1.0.0",
  "root": {
    "path": "rootfs",
    "readonly": false
  },
  "process": {
    "terminal": false,
    "user": {
      "uid": 0,
      "gid": 0
    },
    "args": ["/bin/sh"],
    "env": ["PATH=/usr/bin:/bin", "TERM=xterm"],
    "cwd": "/"
  },
  "hostname": "container",
  "linux": {
    "namespaces": [
      {"type": "pid"},
      {"type": "mount"},
      {"type": "uts"},
      {"type": "ipc"},
      {"type": "network"}
    ],
    "resources": {
      "memory": {
        "limit": 536870912
      },
      "cpu": {
        "shares": 1024
      },
      "pids": {
        "limit": 100
      }
    }
  },
  "mounts": [
    {
      "destination": "/proc",
      "type": "proc",
      "source": "proc"
    },
    {
      "destination": "/sys",
      "type": "sysfs",
      "source": "sysfs",
      "options": ["nosuid", "noexec", "nodev", "ro"]
    }
  ]
}
```

## Container Lifecycle

```
┌──────────┐     create      ┌──────────┐
│  (none)  │ ───────────────▶│ created  │
└──────────┘                 └────┬─────┘
                                  │ start
                                  ▼
                             ┌──────────┐
                             │ running  │
                             └────┬─────┘
                                  │ exit/kill
                                  ▼
                             ┌──────────┐
                             │ stopped  │
                             └────┬─────┘
                                  │ delete
                                  ▼
                             ┌──────────┐
                             │  (none)  │
                             └──────────┘
```

## Resource Limits

### Memory
- `limit`: Hard memory limit in bytes
- `reservation`: Soft limit (memory.low)
- `swap`: Swap limit

### CPU
- `shares`: Relative weight (converted to cpu.weight)
- `quota/period`: CPU bandwidth control
- `cpus`: CPU affinity (cpuset.cpus)

### PIDs
- `limit`: Maximum number of processes

## OverlayFS Layers

```
Container FS = merged view
    ↑
┌─────────┐ ← upper (writable)
├─────────┤
│ layer-3 │ ← lower (read-only)
├─────────┤
│ layer-2 │
├─────────┤
│ layer-1 │
└─────────┘
```

## Building

```bash
cd projects/06-container-runtime
cargo build --release
```

## Requirements

- Linux kernel 4.x+ (for cgroups v2)
- CAP_SYS_ADMIN for namespace/mount operations
- Root or equivalent privileges

## Testing

```bash
cargo test
```

## Implementation Notes

### Namespace Isolation
- Uses `unshare(2)` to create new namespaces
- Handles uid/gid mappings for user namespaces
- Sets hostname in UTS namespace

### Cgroup Management
- Uses cgroup v2 unified hierarchy
- Writes to controller files (memory.max, cpu.weight, etc.)
- Supports freeze/thaw for pausing containers

### Filesystem
- pivot_root for clean filesystem switch
- OverlayFS for efficient layer stacking
- Handles whiteout files for deletions

### Security
- Capabilities dropping (placeholder)
- Seccomp filtering (placeholder)
- Resource limits enforcement

## Future Enhancements

- [ ] User namespace uid/gid mapping
- [ ] Network namespace setup (veth, bridge)
- [ ] Full seccomp BPF filter
- [ ] Capabilities management
- [ ] Checkpoint/restore (CRIU)
- [ ] Rootless containers
- [ ] Image registry pulling

## License

MIT
