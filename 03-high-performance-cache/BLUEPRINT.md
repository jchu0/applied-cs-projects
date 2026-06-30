# Project 3: High-Performance Caching Layer (Redis-lite)

> **Concepts covered:** §01 software-engineering — `rust/00-fundamentals`, `rust/05-async-rust`; §07 infrastructure — `benchmarks/databases`

## Staff-Level Design Document

**Complexity:** ⭐⭐⭐⭐⭐ (Expert)
**Timeline:** 10-12 weeks
**Languages:** Rust (primary) or C++ (alternative)

---

## What This Project Teaches

### Core Concepts
- **In-memory data structures** - Hash tables, skip lists, radix trees, intsets
- **Network programming** - TCP server, event loops, non-blocking I/O
- **Protocol implementation** - RESP (Redis Serialization Protocol)
- **Memory management** - Custom allocators, memory fragmentation, jemalloc
- **Eviction algorithms** - LRU, LFU, random sampling, TTL management
- **Persistence strategies** - Append-only files (AOF), RDB snapshots, fsync policies
- **Replication** - Master-replica, partial resync, replication backlog

### Industry Relevance
This is how Redis, Memcached, and KeyDB work internally. Understanding these patterns is essential for building performance-critical infrastructure and understanding system-level optimization.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Cache Server                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Network   │  │   Command   │  │   Storage   │              │
│  │   Layer     │──│  Processor  │──│   Engine    │              │
│  │  (TCP/TLS)  │  │   (RESP)    │  │             │              │
│  └──────┬──────┘  └─────────────┘  └──────┬──────┘              │
│         │                                  │                     │
│  ┌──────▼──────┐                   ┌──────▼──────┐              │
│  │   Event     │                   │   Data      │              │
│  │   Loop      │                   │ Structures  │              │
│  │(epoll/kqueue)│                  │             │              │
│  └─────────────┘                   └──────┬──────┘              │
│                                           │                     │
│                    ┌──────────────────────┼──────────────────┐  │
│                    │                      │                  │  │
│             ┌──────▼──────┐       ┌───────▼──────┐   ┌───────▼──┐
│             │  Eviction   │       │ Persistence  │   │   TTL    │
│             │  Manager    │       │   (AOF/RDB)  │   │  Manager │
│             └─────────────┘       └──────────────┘   └──────────┘
└─────────────────────────────────────────────────────────────────┘
                            │
                    ┌───────▼───────┐
                    │   Replicas    │
                    └───────────────┘
```

### Component Breakdown

#### 1. Network Layer
**Responsibilities:**
- Accept TCP connections
- Parse RESP protocol
- Handle connection lifecycle
- TLS termination
- Client tracking

**Event Loop Architecture:**
```rust
pub struct EventLoop {
    poll: Poll,
    events: Events,
    connections: HashMap<Token, Connection>,
    next_token: usize,
}

impl EventLoop {
    pub fn run(&mut self, server: &mut Server) -> io::Result<()> {
        loop {
            self.poll.poll(&mut self.events, None)?;

            for event in &self.events {
                match event.token() {
                    LISTENER => self.accept_connection(server)?,
                    token => self.handle_connection(token, event, server)?,
                }
            }
        }
    }

    fn handle_connection(
        &mut self,
        token: Token,
        event: &Event,
        server: &mut Server,
    ) -> io::Result<()> {
        let conn = self.connections.get_mut(&token).unwrap();

        if event.is_readable() {
            // Read data into buffer
            match conn.read() {
                Ok(0) => return self.close_connection(token),
                Ok(_) => {
                    // Parse and execute commands
                    while let Some(cmd) = conn.parse_command()? {
                        let response = server.execute(cmd)?;
                        conn.write_response(response)?;
                    }
                }
                Err(e) if e.kind() == WouldBlock => {}
                Err(e) => return Err(e),
            }
        }

        if event.is_writable() {
            conn.flush()?;
        }

        Ok(())
    }
}
```

#### 2. RESP Protocol Parser
**Protocol Format:**
```
Simple Strings: +OK\r\n
Errors: -ERR unknown command\r\n
Integers: :1000\r\n
Bulk Strings: $5\r\nhello\r\n
Arrays: *2\r\n$3\r\nGET\r\n$3\r\nkey\r\n
Null: $-1\r\n
```

**Parser Implementation:**
```rust
pub enum RespValue {
    SimpleString(String),
    Error(String),
    Integer(i64),
    BulkString(Option<Vec<u8>>),
    Array(Option<Vec<RespValue>>),
}

pub struct RespParser {
    buffer: BytesMut,
}

impl RespParser {
    pub fn parse(&mut self) -> Result<Option<RespValue>, ParseError> {
        if self.buffer.is_empty() {
            return Ok(None);
        }

        let first_byte = self.buffer[0];
        match first_byte {
            b'+' => self.parse_simple_string(),
            b'-' => self.parse_error(),
            b':' => self.parse_integer(),
            b'$' => self.parse_bulk_string(),
            b'*' => self.parse_array(),
            _ => Err(ParseError::InvalidPrefix),
        }
    }

    fn parse_bulk_string(&mut self) -> Result<Option<RespValue>, ParseError> {
        // Skip '$'
        let length_end = self.find_crlf(1)?;
        let length: i64 = self.parse_int(1, length_end)?;

        if length == -1 {
            self.buffer.advance(length_end + 2);
            return Ok(Some(RespValue::BulkString(None)));
        }

        let total_len = length_end + 2 + length as usize + 2;
        if self.buffer.len() < total_len {
            return Ok(None); // Need more data
        }

        let data = self.buffer[length_end + 2..length_end + 2 + length as usize].to_vec();
        self.buffer.advance(total_len);

        Ok(Some(RespValue::BulkString(Some(data))))
    }
}
```

#### 3. Storage Engine
**Core Data Structures:**

```rust
pub struct Database {
    // Main key-value store
    dict: Dict<String, Object>,
    // Keys with expiration
    expires: Dict<String, Instant>,
    // Blocking operations
    blocking_keys: HashMap<String, Vec<ClientId>>,
}

pub enum Object {
    String(StringObject),
    List(ListObject),
    Set(SetObject),
    Hash(HashObject),
    ZSet(ZSetObject),
}

// String: Simple byte array or integer
pub enum StringObject {
    Raw(Vec<u8>),
    Int(i64),
}

// List: Quick list (linked list of ziplists)
pub struct ListObject {
    quicklist: QuickList,
}

// Set: Intset for small integer sets, otherwise hash table
pub enum SetObject {
    IntSet(IntSet),
    HashTable(HashSet<String>),
}

// Hash: Ziplist for small hashes, otherwise hash table
pub enum HashObject {
    ZipList(ZipList),
    HashTable(HashMap<String, String>),
}

// Sorted Set: Skip list + hash table
pub struct ZSetObject {
    dict: HashMap<String, f64>,  // member -> score
    skiplist: SkipList,          // score -> member
}
```

#### 4. Hash Table Implementation

```rust
pub struct Dict<K, V> {
    tables: [HashTable<K, V>; 2],
    rehash_idx: Option<usize>,
    iterators: usize,
}

struct HashTable<K, V> {
    buckets: Vec<Option<Box<Entry<K, V>>>>,
    size: usize,
    mask: usize,
    used: usize,
}

struct Entry<K, V> {
    key: K,
    value: V,
    next: Option<Box<Entry<K, V>>>,
}

impl<K: Hash + Eq, V> Dict<K, V> {
    pub fn get(&self, key: &K) -> Option<&V> {
        // Check if rehashing, search both tables
        if self.is_rehashing() {
            self.rehash_step();
        }

        for table in &self.tables {
            if let Some(entry) = table.find(key) {
                return Some(&entry.value);
            }
        }
        None
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.expand_if_needed();

        if self.is_rehashing() {
            self.rehash_step();
        }

        let table_idx = if self.is_rehashing() { 1 } else { 0 };
        self.tables[table_idx].insert(key, value)
    }

    // Incremental rehashing
    fn rehash_step(&mut self) {
        if let Some(idx) = self.rehash_idx {
            // Move entries from old table to new table
            let mut entries_moved = 0;
            while entries_moved < 10 && idx < self.tables[0].size {
                if let Some(entry) = self.tables[0].buckets[idx].take() {
                    // Rehash to new table
                    self.tables[1].insert_entry(entry);
                    entries_moved += 1;
                }
                self.rehash_idx = Some(idx + 1);
            }

            if idx >= self.tables[0].size {
                // Rehashing complete
                std::mem::swap(&mut self.tables[0], &mut self.tables[1]);
                self.tables[1] = HashTable::new();
                self.rehash_idx = None;
            }
        }
    }
}
```

#### 5. Skip List for Sorted Sets

```rust
const SKIPLIST_MAXLEVEL: usize = 32;
const SKIPLIST_P: f64 = 0.25;

pub struct SkipList {
    head: Box<SkipListNode>,
    tail: *mut SkipListNode,
    length: usize,
    level: usize,
}

struct SkipListNode {
    member: String,
    score: f64,
    backward: *mut SkipListNode,
    levels: Vec<SkipListLevel>,
}

struct SkipListLevel {
    forward: *mut SkipListNode,
    span: usize,
}

impl SkipList {
    pub fn insert(&mut self, score: f64, member: String) -> *mut SkipListNode {
        let mut update: [*mut SkipListNode; SKIPLIST_MAXLEVEL] =
            [std::ptr::null_mut(); SKIPLIST_MAXLEVEL];
        let mut rank: [usize; SKIPLIST_MAXLEVEL] = [0; SKIPLIST_MAXLEVEL];

        // Find insert position
        let mut x = &mut *self.head as *mut SkipListNode;
        for i in (0..self.level).rev() {
            rank[i] = if i == self.level - 1 { 0 } else { rank[i + 1] };

            unsafe {
                while !(*x).levels[i].forward.is_null() {
                    let next = &*(*x).levels[i].forward;
                    if next.score < score
                        || (next.score == score && next.member < member)
                    {
                        rank[i] += (*x).levels[i].span;
                        x = (*x).levels[i].forward;
                    } else {
                        break;
                    }
                }
            }
            update[i] = x;
        }

        // Random level for new node
        let level = self.random_level();
        if level > self.level {
            for i in self.level..level {
                rank[i] = 0;
                update[i] = &mut *self.head;
                self.head.levels[i].span = self.length;
            }
            self.level = level;
        }

        // Create and insert node
        let node = Box::into_raw(Box::new(SkipListNode::new(score, member, level)));
        // ... link node into list

        self.length += 1;
        node
    }

    fn random_level(&self) -> usize {
        let mut level = 1;
        while rand::random::<f64>() < SKIPLIST_P && level < SKIPLIST_MAXLEVEL {
            level += 1;
        }
        level
    }

    pub fn get_rank(&self, score: f64, member: &str) -> Option<usize> {
        let mut rank = 0;
        let mut x = &*self.head as *const SkipListNode;

        for i in (0..self.level).rev() {
            unsafe {
                while !(*x).levels[i].forward.is_null() {
                    let next = &*(*x).levels[i].forward;
                    if next.score < score
                        || (next.score == score && next.member <= member.to_string())
                    {
                        rank += (*x).levels[i].span;
                        x = (*x).levels[i].forward;
                        if next.member == member {
                            return Some(rank);
                        }
                    } else {
                        break;
                    }
                }
            }
        }
        None
    }
}
```

---

## Core Internals

### Eviction Algorithms

#### LRU (Least Recently Used)

```rust
pub struct LRUCache {
    max_memory: usize,
    current_memory: usize,
    sample_size: usize,
}

impl LRUCache {
    pub fn evict_if_needed(&mut self, db: &mut Database) -> usize {
        if self.current_memory <= self.max_memory {
            return 0;
        }

        let mut evicted = 0;

        while self.current_memory > self.max_memory {
            // Sample random keys
            let samples = db.dict.random_keys(self.sample_size);

            if samples.is_empty() {
                break;
            }

            // Find least recently used
            let mut oldest_key = None;
            let mut oldest_time = u64::MAX;

            for key in &samples {
                if let Some(obj) = db.dict.get(key) {
                    if obj.lru_time < oldest_time {
                        oldest_time = obj.lru_time;
                        oldest_key = Some(key.clone());
                    }
                }
            }

            // Evict
            if let Some(key) = oldest_key {
                let freed = db.delete(&key);
                self.current_memory -= freed;
                evicted += 1;
            }
        }

        evicted
    }
}
```

#### LFU (Least Frequently Used)

```rust
// LFU uses logarithmic counter to avoid overflow
pub struct LFUCounter {
    counter: u8,     // 8-bit logarithmic counter
    last_decr: u16,  // Time of last decrement (minutes)
}

impl LFUCounter {
    pub fn increment(&mut self) {
        if self.counter == 255 {
            return;
        }

        let r = rand::random::<f64>();
        let base_val = (self.counter - 5) as f64;  // LFU_INIT_VAL = 5
        let p = 1.0 / (base_val * 10.0 + 1.0);     // lfu_log_factor = 10

        if r < p {
            self.counter += 1;
        }
    }

    pub fn decrement(&mut self, current_minutes: u16) {
        let elapsed = current_minutes.wrapping_sub(self.last_decr);
        let decr = elapsed / 1;  // lfu_decay_time = 1 minute

        if decr > 0 {
            self.counter = self.counter.saturating_sub(decr as u8);
            self.last_decr = current_minutes;
        }
    }
}
```

### Persistence

#### AOF (Append-Only File)

```rust
pub struct AOF {
    file: File,
    buffer: Vec<u8>,
    fsync_policy: FsyncPolicy,
    rewrite_percentage: usize,
    current_size: usize,
    base_size: usize,
}

pub enum FsyncPolicy {
    Always,      // fsync after every command
    EverySecond, // fsync once per second
    No,          // let OS handle it
}

impl AOF {
    pub fn append(&mut self, command: &Command) -> io::Result<()> {
        // Serialize command to RESP
        let resp = command.to_resp();
        self.buffer.extend_from_slice(&resp);

        match self.fsync_policy {
            FsyncPolicy::Always => {
                self.file.write_all(&self.buffer)?;
                self.file.sync_all()?;
                self.buffer.clear();
            }
            FsyncPolicy::EverySecond => {
                // Flush buffer, background thread handles fsync
                self.file.write_all(&self.buffer)?;
                self.buffer.clear();
            }
            FsyncPolicy::No => {
                if self.buffer.len() > 4096 {
                    self.file.write_all(&self.buffer)?;
                    self.buffer.clear();
                }
            }
        }

        self.current_size += resp.len();
        Ok(())
    }

    pub fn rewrite(&mut self, db: &Database) -> io::Result<()> {
        // Create temp file
        let temp_path = format!("{}.rewrite", self.path);
        let mut temp_file = File::create(&temp_path)?;

        // Dump current state as commands
        for (key, value) in db.dict.iter() {
            let cmd = match value {
                Object::String(s) => format!("SET {} {}", key, s),
                Object::List(l) => self.dump_list(key, l),
                Object::Hash(h) => self.dump_hash(key, h),
                Object::Set(s) => self.dump_set(key, s),
                Object::ZSet(z) => self.dump_zset(key, z),
            };

            temp_file.write_all(cmd.as_bytes())?;

            // Include TTL
            if let Some(expire) = db.expires.get(key) {
                let ttl_cmd = format!("PEXPIREAT {} {}\r\n", key, expire);
                temp_file.write_all(ttl_cmd.as_bytes())?;
            }
        }

        // Atomic replace
        temp_file.sync_all()?;
        std::fs::rename(&temp_path, &self.path)?;

        self.base_size = self.current_size;
        Ok(())
    }
}
```

#### RDB (Point-in-time Snapshot)

```rust
pub struct RDB {
    compression: bool,
}

impl RDB {
    pub fn save(&self, path: &str, db: &Database) -> io::Result<()> {
        let temp_path = format!("{}.temp", path);
        let file = File::create(&temp_path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        writer.write_all(b"REDIS0011")?;  // Magic + version

        // Write aux fields
        self.write_aux(&mut writer, "redis-ver", "7.0.0")?;
        self.write_aux(&mut writer, "ctime", &timestamp().to_string())?;

        // Write database selector
        writer.write_all(&[0xFE, 0])?;  // DB 0

        // Write resize info
        self.write_length(&mut writer, db.dict.len())?;
        self.write_length(&mut writer, db.expires.len())?;

        // Write key-value pairs
        for (key, value) in db.dict.iter() {
            // Write expiry if present
            if let Some(expire) = db.expires.get(key) {
                writer.write_all(&[0xFC])?;  // MS timestamp
                writer.write_all(&expire.to_le_bytes())?;
            }

            // Write type
            let type_byte = match value {
                Object::String(_) => 0,
                Object::List(_) => 1,
                Object::Set(_) => 2,
                Object::ZSet(_) => 3,
                Object::Hash(_) => 4,
            };
            writer.write_all(&[type_byte])?;

            // Write key
            self.write_string(&mut writer, key)?;

            // Write value
            self.write_object(&mut writer, value)?;
        }

        // Write EOF
        writer.write_all(&[0xFF])?;

        // Write CRC64 checksum
        let checksum = self.calculate_crc64(&temp_path)?;
        writer.write_all(&checksum.to_le_bytes())?;

        writer.flush()?;
        drop(writer);

        // Atomic replace
        std::fs::rename(&temp_path, path)?;
        Ok(())
    }

    pub fn load(&self, path: &str) -> io::Result<Database> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut db = Database::new();

        // Verify header
        let mut header = [0u8; 9];
        reader.read_exact(&mut header)?;
        if &header[..5] != b"REDIS" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid RDB"));
        }

        // Parse RDB file...
        // (implementation continues)

        Ok(db)
    }
}
```

### Replication

```rust
pub struct ReplicationState {
    role: Role,
    master_host: Option<String>,
    master_port: Option<u16>,
    replid: String,
    offset: u64,
    backlog: ReplicationBacklog,
    replicas: Vec<Replica>,
}

pub struct ReplicationBacklog {
    buffer: VecDeque<u8>,
    offset: u64,
    capacity: usize,
}

impl ReplicationState {
    pub async fn connect_to_master(&mut self) -> io::Result<()> {
        let addr = format!("{}:{}",
            self.master_host.as_ref().unwrap(),
            self.master_port.unwrap());

        let mut stream = TcpStream::connect(&addr).await?;

        // Send PING
        stream.write_all(b"*1\r\n$4\r\nPING\r\n").await?;

        // AUTH if needed
        // ...

        // Send REPLCONF
        let replconf = format!(
            "*3\r\n$8\r\nREPLCONF\r\n$14\r\nlistening-port\r\n$4\r\n6380\r\n"
        );
        stream.write_all(replconf.as_bytes()).await?;

        // Send PSYNC
        let psync = format!(
            "*3\r\n$5\r\nPSYNC\r\n$40\r\n{}\r\n${}\r\n{}\r\n",
            self.replid,
            self.offset.to_string().len(),
            self.offset
        );
        stream.write_all(psync.as_bytes()).await?;

        // Handle response (FULLRESYNC or CONTINUE)
        // ...

        Ok(())
    }

    pub fn propagate(&mut self, command: &[u8]) {
        // Add to backlog
        self.backlog.append(command);
        self.offset += command.len() as u64;

        // Send to all replicas
        for replica in &mut self.replicas {
            replica.send(command);
        }
    }
}
```

---

## Enterprise Features

### 1. Multi-threaded I/O

```rust
pub struct ThreadedIO {
    io_threads: Vec<JoinHandle<()>>,
    pending_reads: Vec<Sender<ReadJob>>,
    pending_writes: Vec<Sender<WriteJob>>,
    results: Receiver<IOResult>,
}

impl ThreadedIO {
    pub fn new(num_threads: usize) -> Self {
        let (result_tx, result_rx) = mpsc::channel();

        let mut io_threads = Vec::with_capacity(num_threads);
        let mut pending_reads = Vec::with_capacity(num_threads);
        let mut pending_writes = Vec::with_capacity(num_threads);

        for i in 0..num_threads {
            let (read_tx, read_rx) = mpsc::channel::<ReadJob>();
            let (write_tx, write_rx) = mpsc::channel::<WriteJob>();
            let result_tx = result_tx.clone();

            let handle = thread::spawn(move || {
                loop {
                    // Handle reads
                    while let Ok(job) = read_rx.try_recv() {
                        let data = job.connection.read();
                        result_tx.send(IOResult::Read(job.token, data)).unwrap();
                    }

                    // Handle writes
                    while let Ok(job) = write_rx.try_recv() {
                        job.connection.write(&job.data);
                        result_tx.send(IOResult::Write(job.token)).unwrap();
                    }

                    thread::yield_now();
                }
            });

            io_threads.push(handle);
            pending_reads.push(read_tx);
            pending_writes.push(write_tx);
        }

        Self {
            io_threads,
            pending_reads,
            pending_writes,
            results: result_rx,
        }
    }
}
```

### 2. Cluster Mode

```rust
pub struct ClusterState {
    myself: ClusterNode,
    nodes: HashMap<String, ClusterNode>,
    slots: [Option<String>; 16384],  // slot -> node_id
    migrating_slots: HashMap<u16, String>,
    importing_slots: HashMap<u16, String>,
}

pub struct ClusterNode {
    id: String,
    ip: String,
    port: u16,
    flags: ClusterNodeFlags,
    master_id: Option<String>,
    slots: Vec<(u16, u16)>,  // Ranges
    ping_sent: u64,
    pong_received: u64,
}

impl ClusterState {
    pub fn key_slot(key: &str) -> u16 {
        // Handle hash tags
        let hash_key = if let (Some(start), Some(end)) =
            (key.find('{'), key.find('}'))
        {
            if end > start + 1 {
                &key[start + 1..end]
            } else {
                key
            }
        } else {
            key
        };

        crc16(hash_key.as_bytes()) % 16384
    }

    pub fn get_node_for_key(&self, key: &str) -> Result<&ClusterNode, ClusterError> {
        let slot = Self::key_slot(key);

        // Check if slot is migrating
        if let Some(target) = self.migrating_slots.get(&slot) {
            return Err(ClusterError::Ask(target.clone(), slot));
        }

        // Get node for slot
        if let Some(node_id) = &self.slots[slot as usize] {
            if let Some(node) = self.nodes.get(node_id) {
                return Ok(node);
            }
        }

        Err(ClusterError::ClusterDown)
    }

    pub fn handle_cluster_message(&mut self, msg: ClusterMessage) {
        match msg {
            ClusterMessage::Ping(node_info) => {
                self.update_node(node_info);
                self.send_pong();
            }
            ClusterMessage::Pong(node_info) => {
                self.update_node(node_info);
                self.mark_node_ok(&node_info.id);
            }
            ClusterMessage::Meet(node_info) => {
                self.add_node(node_info);
            }
            ClusterMessage::Fail(node_id) => {
                self.mark_node_fail(&node_id);
            }
            // ...
        }
    }
}
```

### 3. Keyspace Notifications

```rust
pub struct PubSub {
    channels: HashMap<String, HashSet<ClientId>>,
    patterns: HashMap<String, HashSet<ClientId>>,
    keyspace_events: KeyspaceEventFlags,
}

bitflags! {
    pub struct KeyspaceEventFlags: u32 {
        const KEYSPACE = 0x01;  // __keyspace@<db>__:<key>
        const KEYEVENT = 0x02;  // __keyevent@<db>__:<event>
        const GENERIC = 0x04;   // del, expire, rename, ...
        const STRING = 0x08;    // set, append, ...
        const LIST = 0x10;      // lpush, rpop, ...
        const SET = 0x20;       // sadd, srem, ...
        const HASH = 0x40;      // hset, hdel, ...
        const ZSET = 0x80;      // zadd, zrem, ...
        const EXPIRED = 0x100;  // expired events
        const EVICTED = 0x200;  // evicted events
    }
}

impl PubSub {
    pub fn notify_keyspace_event(
        &self,
        flags: KeyspaceEventFlags,
        event: &str,
        key: &str,
        db: usize,
    ) {
        if !self.keyspace_events.intersects(flags) {
            return;
        }

        // Publish to __keyspace@<db>__:<key> channel
        if self.keyspace_events.contains(KeyspaceEventFlags::KEYSPACE) {
            let channel = format!("__keyspace@{}__:{}", db, key);
            self.publish(&channel, event.as_bytes());
        }

        // Publish to __keyevent@<db>__:<event> channel
        if self.keyspace_events.contains(KeyspaceEventFlags::KEYEVENT) {
            let channel = format!("__keyevent@{}__:{}", db, event);
            self.publish(&channel, key.as_bytes());
        }
    }
}
```

### 4. Memory Optimization

```rust
// Ziplist: Compact encoding for small lists/hashes
pub struct ZipList {
    bytes: Vec<u8>,
}

impl ZipList {
    // Header: zlbytes (4) + zltail (4) + zllen (2)
    // Entry: prevlen (1-5) + encoding (1-5) + data
    // End: 0xFF

    pub fn push_back(&mut self, value: &[u8]) {
        let prevlen = self.last_entry_len();
        let entry = self.encode_entry(prevlen, value);

        // Insert before 0xFF terminator
        let insert_pos = self.bytes.len() - 1;
        self.bytes.splice(insert_pos..insert_pos, entry);

        // Update header
        self.update_bytes_count();
        self.update_tail_offset();
        self.increment_length();
    }

    fn encode_entry(&self, prevlen: usize, value: &[u8]) -> Vec<u8> {
        let mut entry = Vec::new();

        // Encode prevlen
        if prevlen < 254 {
            entry.push(prevlen as u8);
        } else {
            entry.push(0xFE);
            entry.extend_from_slice(&(prevlen as u32).to_le_bytes());
        }

        // Try integer encoding
        if let Ok(int_val) = std::str::from_utf8(value).and_then(|s| s.parse::<i64>().ok()) {
            entry.extend(self.encode_integer(int_val));
        } else {
            // String encoding
            entry.extend(self.encode_string(value));
        }

        entry
    }
}

// IntSet: Memory-efficient set of integers
pub struct IntSet {
    encoding: IntSetEncoding,
    length: u32,
    contents: Vec<u8>,
}

pub enum IntSetEncoding {
    Int16,
    Int32,
    Int64,
}

impl IntSet {
    pub fn add(&mut self, value: i64) -> bool {
        // Upgrade encoding if needed
        let val_enc = Self::value_encoding(value);
        if val_enc as u8 > self.encoding as u8 {
            self.upgrade_encoding(val_enc);
        }

        // Binary search for position
        let (found, pos) = self.search(value);
        if found {
            return false;  // Already exists
        }

        // Insert at position
        self.insert_at(pos, value);
        self.length += 1;
        true
    }
}
```

---

## Performance Considerations

### Memory Efficiency
- **Object encoding:** Use compact encodings (ziplist, intset) for small objects
- **Shared objects:** Reuse common strings (OK, QUEUED, integers 0-9999)
- **Memory allocator:** Use jemalloc for better fragmentation handling
- **Lazy freeing:** Async deletion for large objects

### CPU Optimization
- **Incremental rehashing:** Spread rehashing work across operations
- **Approximate algorithms:** Use sampling for LRU/LFU eviction
- **Lazy expiration:** Only expire keys when accessed or sampled

### Network Optimization
- **Pipelining:** Process multiple commands without waiting for responses
- **Client output buffer limits:** Prevent slow clients from consuming memory
- **TCP_NODELAY:** Disable Nagle's algorithm for latency

### Benchmarks Target
| Operation | Target Throughput | Target Latency (P99) |
|-----------|-------------------|----------------------|
| GET | 500,000 ops/sec | <1ms |
| SET | 500,000 ops/sec | <1ms |
| LPUSH | 400,000 ops/sec | <1ms |
| ZADD | 300,000 ops/sec | <1ms |

---

## Stretch Goals

### 1. Redis Streams

```rust
pub struct Stream {
    rax: RadixTree<StreamEntry>,
    length: u64,
    last_id: StreamID,
    groups: HashMap<String, ConsumerGroup>,
}

pub struct StreamID {
    ms: u64,
    seq: u64,
}

pub struct StreamEntry {
    id: StreamID,
    fields: Vec<(String, String)>,
}

pub struct ConsumerGroup {
    name: String,
    last_delivered_id: StreamID,
    pel: HashMap<StreamID, PendingEntry>,  // Pending entries list
    consumers: HashMap<String, Consumer>,
}

impl Stream {
    pub fn xadd(&mut self, id: Option<StreamID>, fields: Vec<(String, String)>) -> StreamID {
        let id = id.unwrap_or_else(|| self.generate_id());

        let entry = StreamEntry { id, fields };
        self.rax.insert(&id.to_bytes(), entry);
        self.length += 1;
        self.last_id = id;

        id
    }

    pub fn xread(
        &self,
        streams: &[(String, StreamID)],
        count: Option<usize>,
        block: Option<Duration>,
    ) -> Vec<(String, Vec<StreamEntry>)> {
        // Implementation
        todo!()
    }

    pub fn xreadgroup(
        &mut self,
        group: &str,
        consumer: &str,
        streams: &[(String, StreamID)],
        count: Option<usize>,
    ) -> Vec<(String, Vec<StreamEntry>)> {
        // Deliver to consumer, track in PEL
        todo!()
    }
}
```

### 2. RESP3 Protocol

```rust
pub enum Resp3Value {
    // Simple types
    SimpleString(String),
    SimpleError(String),
    Integer(i64),
    BulkString(Option<Vec<u8>>),
    Array(Option<Vec<Resp3Value>>),

    // RESP3 additions
    Null,
    Boolean(bool),
    Double(f64),
    BigNumber(String),
    BulkError(Vec<u8>),
    VerbatimString { format: String, data: Vec<u8> },
    Map(Vec<(Resp3Value, Resp3Value)>),
    Set(Vec<Resp3Value>),
    Attribute(HashMap<String, Resp3Value>),
    Push(Vec<Resp3Value>),
}

impl Resp3Value {
    pub fn serialize(&self) -> Vec<u8> {
        match self {
            Resp3Value::Null => b"_\r\n".to_vec(),
            Resp3Value::Boolean(true) => b"#t\r\n".to_vec(),
            Resp3Value::Boolean(false) => b"#f\r\n".to_vec(),
            Resp3Value::Double(d) => format!(",{}\r\n", d).into_bytes(),
            Resp3Value::Map(pairs) => {
                let mut buf = format!("%{}\r\n", pairs.len()).into_bytes();
                for (k, v) in pairs {
                    buf.extend(k.serialize());
                    buf.extend(v.serialize());
                }
                buf
            }
            // ...
        }
    }
}
```

### 3. Multi-Key Transactions

```rust
pub struct Transaction {
    commands: Vec<QueuedCommand>,
    watched_keys: HashMap<String, u64>,  // key -> version
}

impl Transaction {
    pub fn watch(&mut self, keys: &[String], db: &Database) {
        for key in keys {
            let version = db.get_key_version(key);
            self.watched_keys.insert(key.clone(), version);
        }
    }

    pub fn exec(&mut self, db: &mut Database) -> Result<Vec<RespValue>, TransactionError> {
        // Check watched keys
        for (key, version) in &self.watched_keys {
            if db.get_key_version(key) != *version {
                return Err(TransactionError::WatchModified);
            }
        }

        // Execute all commands atomically
        let mut results = Vec::with_capacity(self.commands.len());
        for cmd in &self.commands {
            let result = db.execute(cmd)?;
            results.push(result);
        }

        Ok(results)
    }
}

// Optimistic locking with CAS
pub fn compare_and_set(
    db: &mut Database,
    key: &str,
    expected: &[u8],
    new_value: &[u8],
) -> bool {
    if let Some(current) = db.get(key) {
        if current == expected {
            db.set(key, new_value);
            return true;
        }
    }
    false
}
```

---

## Testing Strategy

### Unit Tests
- RESP parser edge cases
- Hash table operations
- Skip list correctness
- Eviction algorithm behavior
- Persistence format

### Integration Tests
- Full command coverage
- Replication scenarios
- Cluster operations
- Pub/Sub delivery

### Fuzz Testing
- RESP parser fuzzing
- Command argument fuzzing
- RDB/AOF parser fuzzing

### Performance Tests
- Throughput benchmarks (redis-benchmark)
- Latency distribution
- Memory efficiency
- Replication lag

### Compatibility Tests
- Redis protocol compliance
- Client library compatibility
- Migration from Redis

---

## Implementation Phases

### Phase 1: Core Server (Week 1-3)
- [ ] TCP server with event loop
- [ ] RESP parser and serializer
- [ ] Basic string commands (GET, SET, DEL)
- [ ] Hash table implementation
- [ ] TTL support (EXPIRE, TTL)

### Phase 2: Data Structures (Week 4-5)
- [ ] List commands (LPUSH, RPUSH, LPOP, LRANGE)
- [ ] Set commands (SADD, SMEMBERS, SINTER)
- [ ] Hash commands (HSET, HGET, HGETALL)
- [ ] Sorted set with skip list (ZADD, ZRANGE, ZRANK)

### Phase 3: Persistence (Week 6-7)
- [ ] RDB snapshot save/load
- [ ] AOF logging
- [ ] AOF rewrite
- [ ] Background saving (BGSAVE)

### Phase 4: Advanced Features (Week 8-9)
- [ ] LRU/LFU eviction
- [ ] Pub/Sub
- [ ] Transactions (MULTI/EXEC)
- [ ] Lua scripting (optional)

### Phase 5: Replication & Scale (Week 10-12)
- [ ] Master-replica replication
- [ ] Partial resync
- [ ] Cluster mode basics
- [ ] Multi-threaded I/O

---

## References

- [Redis Internals](https://redis.io/docs/reference/internals/)
- [Redis Source Code](https://github.com/redis/redis)
- [Skip Lists: A Probabilistic Alternative to Balanced Trees](https://15721.courses.cs.cmu.edu/spring2018/papers/08-oltpindexes1/pugh-skiplists-cacm1990.pdf)
- [The Design and Implementation of a Log-Structured File System](https://people.eecs.berkeley.edu/~brewer/cs262/LFS.pdf)
- [jemalloc](https://jemalloc.net/)
