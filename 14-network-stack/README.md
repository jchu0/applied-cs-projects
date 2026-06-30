# Network Stack (TCP + HTTP)

A userspace TCP/IP stack and HTTP/1.1 parser written in Rust, implementing the full TCP state machine, sliding-window flow control, Reno congestion control, and an HTTP connection pool with keep-alive and reverse-proxy support.

> **Status:** reference implementation / teaching scaffold built to a strong blueprint — not production-grade. See [../PROJECTS_STATUS.md](../PROJECTS_STATUS.md) and the [2026-06 audit](../../docs/AUDIT_2026-06_public-readiness.md).

> **Concepts covered:** §01 software-engineering — `rust/*`; §07 infrastructure — `benchmarks/languages`

---

## What's real vs simulated

The TCP state machine, congestion control (slow start / congestion avoidance / fast recovery), RTT estimation, retransmission timers, and HTTP parser are fully implemented and exercised by 60 passing tests. The **TUN/TAP interface is simulated** — the implementation is not wired to a real kernel network device, so the stack runs in-process only and does not send or receive actual IP packets on the host network.

---

## Layout

```
src/
  lib.rs       — crate root, shared Error / Result types
  tcp.rs       — TCP connection state machine, send/recv buffers, timers
  http.rs      — HTTP/1.1 request/response parser (chunked encoding, keep-alive)
  pool.rs      — HTTP connection pool (idle timeout, per-host limits)
  proxy.rs     — reverse proxy with weighted load balancing and health checks

tests/
  tcp_test.rs  — state machine, handshake, retransmission (39 tests)
  http_test.rs — request parsing, chunked bodies, pipelining (13 tests)
  pool_test.rs — connection pool lifecycle (8 tests)

BLUEPRINT.md   — full architecture, data structures, and implementation phases
Cargo.toml     — dependencies (tokio, bytes, bitflags, etherparse, thiserror)
```

---

## Build & test

```bash
cd 06-real-world-projects/14-network-stack
cargo build
cargo test
```

All 60 tests pass without external dependencies or a running kernel TUN/TAP device.

---

## Key concepts implemented

- **TCP state machine** — all 11 RFC 793 states, simultaneous open/close
- **Sequence-number arithmetic** — wrapping u32 comparisons, ISN generation
- **Sliding-window flow control** — send window clamped by receiver advertisement and cwnd
- **Retransmission** — RTO via RFC 6298 SRTT/RTTVAR, exponential back-off, fast retransmit on 3 duplicate ACKs
- **Congestion control** — TCP Reno (slow start, congestion avoidance, fast recovery); CUBIC skeleton in blueprint
- **HTTP/1.1 parsing** — request line, headers, Content-Length and chunked transfer encoding
- **Connection pooling** — idle-timeout eviction, per-host caps, keep-alive reuse
- **Reverse proxy** — weighted round-robin backend selection, connection forwarding
