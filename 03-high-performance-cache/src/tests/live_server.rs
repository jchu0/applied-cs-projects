//! Integration tests that drive the *live* server (`Server::run`) over a real
//! TCP socket, exercising the wired-in eviction, AOF persistence, RDB
//! round-trips, multi-database `SELECT`, and periodic expiration paths.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::{Duration, Instant};

use redis_lite::config::Config;
use redis_lite::server::Server;

/// Grab a free TCP port by binding to port 0 and immediately releasing it.
fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

/// Start a live server on a background thread and return its port.
fn start_server(mut config: Config) -> u16 {
    let port = free_port();
    config.bind = "127.0.0.1".to_string();
    config.port = port;

    thread::spawn(move || {
        let mut server = Server::new(config).expect("server should start");
        let _ = server.run();
    });

    // Wait until the server accepts connections.
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            break;
        }
        assert!(Instant::now() < deadline, "server did not come up in time");
        thread::sleep(Duration::from_millis(20));
    }
    port
}

/// A minimal RESP client over a persistent TCP connection.
struct Client {
    stream: TcpStream,
    buf: Vec<u8>,
}

impl Client {
    fn connect(port: u16) -> Self {
        let stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        Self { stream, buf: Vec::new() }
    }

    /// Send a command as a RESP array of bulk strings and return the raw reply.
    fn cmd(&mut self, args: &[&str]) -> String {
        let mut out = format!("*{}\r\n", args.len());
        for a in args {
            out.push_str(&format!("${}\r\n{}\r\n", a.len(), a));
        }
        self.stream.write_all(out.as_bytes()).unwrap();
        self.stream.flush().unwrap();
        self.read_reply()
    }

    /// Read exactly one RESP reply. This handles the reply shapes produced by
    /// the commands under test: simple strings, errors, integers, bulk
    /// strings, and (nested) arrays.
    fn read_reply(&mut self) -> String {
        let line = self.read_line();
        match line.as_bytes()[0] {
            b'+' | b'-' | b':' => line,
            b'$' => {
                let len: i64 = line[1..].parse().unwrap();
                if len < 0 {
                    return line; // null bulk string
                }
                let mut data = vec![0u8; len as usize + 2]; // + CRLF
                self.read_exact(&mut data);
                format!("{}{}", line, String::from_utf8_lossy(&data[..len as usize]))
            }
            b'*' => {
                let count: i64 = line[1..].parse().unwrap();
                let mut acc = line.clone();
                if count > 0 {
                    for _ in 0..count {
                        acc.push('|');
                        acc.push_str(&self.read_reply());
                    }
                }
                acc
            }
            _ => line,
        }
    }

    fn fill(&mut self) {
        let mut tmp = [0u8; 1024];
        let n = self.stream.read(&mut tmp).unwrap();
        assert!(n > 0, "connection closed unexpectedly");
        self.buf.extend_from_slice(&tmp[..n]);
    }

    fn read_line(&mut self) -> String {
        loop {
            if let Some(pos) = self.buf.windows(2).position(|w| w == b"\r\n") {
                let line: Vec<u8> = self.buf.drain(..pos + 2).collect();
                return String::from_utf8_lossy(&line[..line.len() - 2]).into_owned();
            }
            self.fill();
        }
    }

    fn read_exact(&mut self, out: &mut [u8]) {
        while self.buf.len() < out.len() {
            self.fill();
        }
        let taken: Vec<u8> = self.buf.drain(..out.len()).collect();
        out.copy_from_slice(&taken);
    }
}

// ==================== Eviction ====================

#[test]
fn eviction_noeviction_errors_when_full() {
    let config = Config {
        maxmemory: 2048, // tiny cap
        maxmemory_policy: "noeviction".to_string(),
        ..Config::default()
    };
    let port = start_server(config);
    let mut c = Client::connect(port);

    // Write large values until the server rejects with OOM.
    let big = "x".repeat(512);
    let mut saw_oom = false;
    for i in 0..200 {
        let reply = c.cmd(&["SET", &format!("k{}", i), &big]);
        if reply.starts_with("-OOM") {
            saw_oom = true;
            break;
        }
    }
    assert!(saw_oom, "noeviction policy should reject writes once over maxmemory");
}

#[test]
fn eviction_allkeys_lru_frees_memory() {
    let config = Config {
        maxmemory: 4096,
        maxmemory_policy: "allkeys-lru".to_string(),
        ..Config::default()
    };
    let port = start_server(config);
    let mut c = Client::connect(port);

    let big = "y".repeat(256);
    // Insert far more data than fits; eviction must keep it accepting writes.
    for i in 0..300 {
        let reply = c.cmd(&["SET", &format!("key{}", i), &big]);
        assert!(
            reply.starts_with("+OK"),
            "allkeys-lru should never OOM, got: {}",
            reply
        );
    }

    // Some earlier keys must have been evicted (DBSIZE stays bounded).
    let dbsize = c.cmd(&["DBSIZE"]);
    let count: i64 = dbsize.trim_start_matches(':').parse().unwrap();
    assert!(count < 300, "expected eviction to bound DBSIZE, got {}", count);
    assert!(count > 0, "database should still hold recent keys");
}

// ==================== AOF persistence + real replay path ====================

#[test]
fn aof_persists_and_replays_on_restart() {
    let dir = tempfile::tempdir().unwrap();
    let make_config = || Config {
        appendonly: true,
        dir: dir.path().to_string_lossy().into_owned(),
        ..Config::default()
    };

    // First server instance: write keys through the live path.
    {
        let port = start_server(make_config());
        let mut c = Client::connect(port);
        assert!(c.cmd(&["SET", "alpha", "1"]).starts_with("+OK"));
        assert!(c.cmd(&["SET", "beta", "2"]).starts_with("+OK"));
        assert!(c.cmd(&["RPUSH", "mylist", "a", "b", "c"]).starts_with(":3"));
        assert!(c.cmd(&["INCR", "counter"]).starts_with(":1"));
        assert!(c.cmd(&["INCR", "counter"]).starts_with(":2"));
        // Give the everysec fsync a moment, then force durability.
        thread::sleep(Duration::from_millis(50));
        drop(c);
        // Let the background fsync flush the buffer to disk.
        thread::sleep(Duration::from_millis(1200));
    }

    // Second server instance over the same AOF directory: keys must be present,
    // having been replayed through the real command execution path.
    {
        let port = start_server(make_config());
        let mut c = Client::connect(port);
        assert_eq!(c.cmd(&["GET", "alpha"]), "$11");
        assert_eq!(c.cmd(&["GET", "beta"]), "$12");
        assert!(c.cmd(&["LLEN", "mylist"]).starts_with(":3"));
        // INCR replay must have applied cumulatively -> 2.
        assert!(c.cmd(&["GET", "counter"]).contains("2"));
    }
}

// ==================== Multi-DB SELECT isolation ====================

#[test]
fn select_isolates_keys_across_databases() {
    let config = Config {
        databases: 4,
        ..Config::default()
    };
    let port = start_server(config);
    let mut c = Client::connect(port);

    // Put a key in DB 0.
    assert!(c.cmd(&["SET", "shared", "in-db-0"]).starts_with("+OK"));

    // Switch to DB 1: the key must not be visible there.
    assert!(c.cmd(&["SELECT", "1"]).starts_with("+OK"));
    assert_eq!(c.cmd(&["GET", "shared"]), "$-1");
    assert!(c.cmd(&["SET", "shared", "in-db-1"]).starts_with("+OK"));
    assert_eq!(c.cmd(&["GET", "shared"]), "$7in-db-1");

    // Back to DB 0: original value intact.
    assert!(c.cmd(&["SELECT", "0"]).starts_with("+OK"));
    assert_eq!(c.cmd(&["GET", "shared"]), "$7in-db-0");

    // Out-of-range SELECT is rejected.
    assert!(c.cmd(&["SELECT", "99"]).starts_with("-ERR"));
}

// ==================== Periodic (active) expiration ====================

#[test]
fn active_expiration_reclaims_untouched_keys() {
    let config = Config::default();
    let port = start_server(config);
    let mut c = Client::connect(port);

    // Key with a 1-second TTL that we never touch again.
    assert!(c.cmd(&["SET", "ephemeral", "v"]).starts_with("+OK"));
    assert!(c.cmd(&["EXPIRE", "ephemeral", "1"]).starts_with(":1"));
    assert!(c.cmd(&["SET", "permanent", "v"]).starts_with("+OK"));

    // Wait past the TTL; the server's periodic cycle should drop it even though
    // we never GET it again.
    thread::sleep(Duration::from_millis(1500));

    // DBSIZE reflects active expiration without a prior access to the key.
    let dbsize = c.cmd(&["DBSIZE"]);
    assert!(
        dbsize.starts_with(":1"),
        "active expiration should have removed the expired key, DBSIZE={}",
        dbsize
    );
    assert_eq!(c.cmd(&["GET", "ephemeral"]), "$-1");
    assert_eq!(c.cmd(&["GET", "permanent"]), "$1v");
}
