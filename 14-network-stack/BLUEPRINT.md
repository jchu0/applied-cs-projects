# Network Stack (TCP + HTTP)

> **Concepts covered:** §01 software-engineering — `rust/*`; §07 infrastructure — `benchmarks/languages`

## Executive Summary

A userspace network stack implementation covering TCP protocol internals and HTTP/1.1 parsing. This project provides deep understanding of TCP's three-way handshake, sequence/acknowledgment numbers, sliding window flow control, congestion control algorithms (slow start, Reno), retransmission mechanisms, and HTTP protocol parsing with keep-alive connections.

---

## System Architecture

```
                    Application Layer
                          |
              +-----------+-----------+
              |                       |
        +-----v-----+           +-----v-----+
        |  HTTP/1.1 |           |  HTTP/2   |  (Stretch)
        |  Parser   |           |  Streams  |
        +-----------+           +-----------+
              |                       |
              +-----------+-----------+
                          |
                    +-----v-----+
                    |    TCP    |
                    | Transport |
                    +-----+-----+
                          |
        +-----------------+-----------------+
        |                 |                 |
   +----v----+      +-----v----+     +-----v-----+
   |  Send   |      | Receive  |     | Congestion|
   | Buffer  |      | Buffer   |     | Control   |
   +---------+      +----------+     +-----------+
                          |
                    +-----v-----+
                    |    IP     |
                    | (Simulated)|
                    +-----+-----+
                          |
                    +-----v-----+
                    |   TUN/TAP |
                    |  Interface|
                    +-----------+

TCP Connection State Machine:
                              +---------+
                              |  CLOSED |
                              +----+----+
                     passive       |       active
                     open          |       open
                   +---------------+---------------+
                   |                               |
              +----v----+                     +----v----+
              |  LISTEN |                     | SYN-SENT|
              +----+----+                     +----+----+
                   |                               |
             rcv SYN,                        rcv SYN,ACK
             snd SYN,ACK                     snd ACK
                   |                               |
              +----v----+                     +----v----+
              |SYN-RCVD |                     |  ESTAB  |<--------+
              +----+----+                     +----+----+         |
                   |                               |              |
              rcv ACK                         close, snd FIN      |
                   |                               |              |
                   +-----------> ESTAB <-----------+              |
                                   |                              |
                               close                              |
                                   |                              |
                              +----v----+                         |
                              | FIN-WAIT|                         |
                              +----+----+                         |
                                   |                              |
                              rcv FIN,ACK                         |
                              snd ACK                             |
                                   |                              |
                              +----v----+                         |
                              |TIME-WAIT|                         |
                              +----+----+                         |
                                   |                              |
                              2MSL timeout                        |
                                   |                              |
                              +----v----+                         |
                              |  CLOSED |-------------------------+
                              +---------+
```

---

## Core Data Structures

### TCP Control Block

```rust
pub struct TcpConnection {
    // Connection identifier
    local_addr: SocketAddr,
    remote_addr: SocketAddr,
    state: TcpState,

    // Sequence numbers
    send: SendSequenceSpace,
    recv: RecvSequenceSpace,

    // Buffers
    send_buffer: SendBuffer,
    recv_buffer: RecvBuffer,

    // Timers
    timers: TcpTimers,

    // Congestion control
    congestion: CongestionControl,

    // Options
    mss: u16,                      // Maximum Segment Size
    window_scale: u8,
    timestamps_enabled: bool,
}

pub struct SendSequenceSpace {
    una: u32,      // Oldest unacknowledged sequence number
    nxt: u32,      // Next sequence number to send
    wnd: u16,      // Send window (receiver's advertised window)
    up: u32,       // Urgent pointer
    wl1: u32,      // Segment sequence used for last window update
    wl2: u32,      // Segment ack used for last window update
    iss: u32,      // Initial send sequence number
}

pub struct RecvSequenceSpace {
    nxt: u32,      // Next expected sequence number
    wnd: u16,      // Receive window
    up: u32,       // Urgent pointer
    irs: u32,      // Initial receive sequence number
}

pub enum TcpState {
    Closed,
    Listen,
    SynSent,
    SynReceived,
    Established,
    FinWait1,
    FinWait2,
    CloseWait,
    Closing,
    LastAck,
    TimeWait,
}
```

### TCP Segment

```rust
pub struct TcpSegment {
    // Header fields
    src_port: u16,
    dst_port: u16,
    seq: u32,
    ack: u32,
    data_offset: u8,      // Header length in 32-bit words
    flags: TcpFlags,
    window: u16,
    checksum: u16,
    urgent_ptr: u16,
    options: Vec<TcpOption>,

    // Payload
    payload: Bytes,
}

bitflags! {
    pub struct TcpFlags: u8 {
        const FIN = 0b00000001;
        const SYN = 0b00000010;
        const RST = 0b00000100;
        const PSH = 0b00001000;
        const ACK = 0b00010000;
        const URG = 0b00100000;
        const ECE = 0b01000000;
        const CWR = 0b10000000;
    }
}

pub enum TcpOption {
    EndOfList,
    NoOp,
    Mss(u16),
    WindowScale(u8),
    SackPermitted,
    Sack(Vec<(u32, u32)>),
    Timestamp { ts_val: u32, ts_ecr: u32 },
}
```

### Send and Receive Buffers

```rust
pub struct SendBuffer {
    buffer: VecDeque<u8>,
    unacked: VecDeque<SentSegment>,
    capacity: usize,
}

pub struct SentSegment {
    seq: u32,
    len: u32,
    sent_at: Instant,
    retransmissions: u32,
}

pub struct RecvBuffer {
    buffer: BTreeMap<u32, Bytes>,  // seq -> data (handles out-of-order)
    assembled: VecDeque<u8>,       // In-order data ready for app
    capacity: usize,
}

impl RecvBuffer {
    pub fn insert(&mut self, seq: u32, data: Bytes) -> bool {
        // Check if we can accept this segment
        let space = self.capacity - self.assembled.len();
        if data.len() > space {
            return false;
        }

        // Store out-of-order segment
        self.buffer.insert(seq, data);

        // Try to assemble contiguous data
        self.assemble();
        true
    }

    fn assemble(&mut self) {
        let mut next_seq = self.next_expected_seq();

        while let Some(data) = self.buffer.remove(&next_seq) {
            self.assembled.extend(data.iter());
            next_seq = next_seq.wrapping_add(data.len() as u32);
        }
    }
}
```

---

## TCP Connection Lifecycle

### Three-Way Handshake

```rust
impl TcpConnection {
    pub fn connect(&mut self, remote: SocketAddr) -> Result<()> {
        // Generate ISS
        self.send.iss = generate_isn();
        self.send.una = self.send.iss;
        self.send.nxt = self.send.iss.wrapping_add(1);

        // Send SYN
        let syn = TcpSegment {
            seq: self.send.iss,
            flags: TcpFlags::SYN,
            options: vec![
                TcpOption::Mss(self.mss),
                TcpOption::WindowScale(self.window_scale),
                TcpOption::SackPermitted,
            ],
            ..Default::default()
        };

        self.send_segment(syn)?;
        self.state = TcpState::SynSent;

        // Start retransmission timer
        self.timers.start_retransmit();

        Ok(())
    }

    pub fn handle_segment(&mut self, segment: TcpSegment) -> Result<()> {
        match self.state {
            TcpState::Listen => self.handle_listen(segment),
            TcpState::SynSent => self.handle_syn_sent(segment),
            TcpState::SynReceived => self.handle_syn_received(segment),
            TcpState::Established => self.handle_established(segment),
            TcpState::FinWait1 => self.handle_fin_wait1(segment),
            TcpState::FinWait2 => self.handle_fin_wait2(segment),
            TcpState::CloseWait => self.handle_close_wait(segment),
            TcpState::Closing => self.handle_closing(segment),
            TcpState::LastAck => self.handle_last_ack(segment),
            TcpState::TimeWait => self.handle_time_wait(segment),
            TcpState::Closed => Err(Error::ConnectionClosed),
        }
    }

    fn handle_syn_sent(&mut self, segment: TcpSegment) -> Result<()> {
        if !segment.flags.contains(TcpFlags::ACK) {
            // Simultaneous open: SYN without ACK
            if segment.flags.contains(TcpFlags::SYN) {
                self.recv.irs = segment.seq;
                self.recv.nxt = segment.seq.wrapping_add(1);

                // Send SYN+ACK
                let synack = TcpSegment {
                    seq: self.send.iss,
                    ack: self.recv.nxt,
                    flags: TcpFlags::SYN | TcpFlags::ACK,
                    ..Default::default()
                };
                self.send_segment(synack)?;
                self.state = TcpState::SynReceived;
            }
            return Ok(());
        }

        // Check ACK validity
        if !self.is_valid_ack(segment.ack) {
            // Send RST
            return self.send_rst(segment.ack);
        }

        if segment.flags.contains(TcpFlags::SYN) {
            // SYN+ACK received
            self.recv.irs = segment.seq;
            self.recv.nxt = segment.seq.wrapping_add(1);
            self.send.una = segment.ack;

            // Process options
            self.process_options(&segment.options);

            // Send ACK
            let ack = TcpSegment {
                seq: self.send.nxt,
                ack: self.recv.nxt,
                flags: TcpFlags::ACK,
                window: self.recv.wnd,
                ..Default::default()
            };
            self.send_segment(ack)?;

            self.state = TcpState::Established;
            self.timers.cancel_retransmit();
        }

        Ok(())
    }
}
```

### Data Transfer

```rust
impl TcpConnection {
    pub fn send(&mut self, data: &[u8]) -> Result<usize> {
        if self.state != TcpState::Established {
            return Err(Error::NotConnected);
        }

        // Add to send buffer
        let added = self.send_buffer.write(data);

        // Try to send
        self.send_data()?;

        Ok(added)
    }

    fn send_data(&mut self) -> Result<()> {
        let effective_window = std::cmp::min(
            self.send.wnd as usize,
            self.congestion.cwnd as usize,
        );

        let in_flight = self.send.nxt.wrapping_sub(self.send.una) as usize;
        let available = effective_window.saturating_sub(in_flight);

        if available == 0 {
            return Ok(());  // Window full
        }

        // Calculate how much to send
        let to_send = std::cmp::min(
            std::cmp::min(available, self.mss as usize),
            self.send_buffer.len(),
        );

        if to_send == 0 {
            return Ok(());
        }

        // Get data from buffer
        let data = self.send_buffer.read(to_send);

        // Create segment
        let segment = TcpSegment {
            seq: self.send.nxt,
            ack: self.recv.nxt,
            flags: TcpFlags::ACK | TcpFlags::PSH,
            window: self.recv.wnd,
            payload: data,
            ..Default::default()
        };

        // Send and track
        self.send_segment(segment.clone())?;
        self.send_buffer.mark_sent(SentSegment {
            seq: self.send.nxt,
            len: to_send as u32,
            sent_at: Instant::now(),
            retransmissions: 0,
        });

        self.send.nxt = self.send.nxt.wrapping_add(to_send as u32);

        // Start retransmit timer if not running
        if !self.timers.retransmit_running() {
            self.timers.start_retransmit();
        }

        Ok(())
    }

    fn handle_established(&mut self, segment: TcpSegment) -> Result<()> {
        // Check sequence number validity
        if !self.is_valid_seq(segment.seq, segment.payload.len()) {
            return self.send_ack();  // Send duplicate ACK
        }

        // Process ACK
        if segment.flags.contains(TcpFlags::ACK) {
            self.process_ack(segment.ack)?;
        }

        // Process data
        if !segment.payload.is_empty() {
            self.process_data(segment.seq, segment.payload)?;
        }

        // Check for FIN
        if segment.flags.contains(TcpFlags::FIN) {
            self.recv.nxt = self.recv.nxt.wrapping_add(1);
            self.send_ack()?;
            self.state = TcpState::CloseWait;
        }

        Ok(())
    }

    fn process_ack(&mut self, ack: u32) -> Result<()> {
        if !self.is_valid_ack(ack) {
            return Ok(());  // Ignore invalid ACK
        }

        // New data acknowledged
        if ack.wrapping_sub(self.send.una) > 0 {
            let acked = ack.wrapping_sub(self.send.una);

            // Remove from unacked queue
            self.send_buffer.ack_bytes(acked as usize);

            // Update congestion control
            self.congestion.on_ack(acked);

            // Update UNA
            self.send.una = ack;

            // Restart retransmit timer
            if self.send.una != self.send.nxt {
                self.timers.restart_retransmit();
            } else {
                self.timers.cancel_retransmit();
            }

            // Try to send more data
            self.send_data()?;
        }

        Ok(())
    }

    fn is_valid_seq(&self, seq: u32, len: usize) -> bool {
        let len = if len == 0 { 1 } else { len as u32 };  // Empty segments take 1 seq
        let seg_end = seq.wrapping_add(len - 1);

        // Check if segment fits in receive window
        let wnd_end = self.recv.nxt.wrapping_add(self.recv.wnd as u32);

        // SEG.SEQ >= RCV.NXT and SEG.SEQ < RCV.NXT + RCV.WND
        // or SEG.SEQ + SEG.LEN - 1 >= RCV.NXT and SEG.SEQ + SEG.LEN - 1 < RCV.NXT + RCV.WND
        (seq.wrapping_sub(self.recv.nxt) < self.recv.wnd as u32) ||
        (seg_end.wrapping_sub(self.recv.nxt) < self.recv.wnd as u32)
    }
}
```

---

## Retransmission

### Timer Management

```rust
pub struct TcpTimers {
    retransmit: Option<Instant>,
    time_wait: Option<Instant>,
    persist: Option<Instant>,
    keepalive: Option<Instant>,

    // RTT estimation
    srtt: Duration,           // Smoothed RTT
    rttvar: Duration,         // RTT variance
    rto: Duration,            // Retransmission timeout
}

impl TcpTimers {
    pub fn update_rtt(&mut self, rtt: Duration) {
        // RFC 6298 RTT estimation
        if self.srtt == Duration::ZERO {
            // First measurement
            self.srtt = rtt;
            self.rttvar = rtt / 2;
        } else {
            // Subsequent measurements
            let alpha = 0.125;
            let beta = 0.25;

            let diff = if rtt > self.srtt {
                rtt - self.srtt
            } else {
                self.srtt - rtt
            };

            self.rttvar = Duration::from_secs_f64(
                (1.0 - beta) * self.rttvar.as_secs_f64() +
                beta * diff.as_secs_f64()
            );

            self.srtt = Duration::from_secs_f64(
                (1.0 - alpha) * self.srtt.as_secs_f64() +
                alpha * rtt.as_secs_f64()
            );
        }

        // RTO = SRTT + max(G, 4 * RTTVAR)
        // G is clock granularity, assume 1ms
        self.rto = self.srtt + std::cmp::max(
            Duration::from_millis(1),
            self.rttvar * 4
        );

        // Clamp RTO
        self.rto = std::cmp::max(self.rto, Duration::from_secs(1));
        self.rto = std::cmp::min(self.rto, Duration::from_secs(60));
    }
}

impl TcpConnection {
    pub fn handle_timeout(&mut self) -> Result<()> {
        // Retransmit oldest unacked segment
        if let Some(segment) = self.send_buffer.oldest_unacked() {
            // Exponential backoff
            self.timers.rto *= 2;
            self.timers.rto = std::cmp::min(self.timers.rto, Duration::from_secs(60));

            // Congestion control: enter slow start
            self.congestion.on_timeout();

            // Retransmit
            self.retransmit_segment(segment)?;

            // Restart timer
            self.timers.start_retransmit();
        }

        Ok(())
    }

    pub fn handle_duplicate_ack(&mut self) -> Result<()> {
        // Fast retransmit after 3 duplicate ACKs
        self.congestion.dup_acks += 1;

        if self.congestion.dup_acks == 3 {
            // Fast retransmit
            if let Some(segment) = self.send_buffer.oldest_unacked() {
                self.retransmit_segment(segment)?;
            }

            // Fast recovery
            self.congestion.on_fast_retransmit();
        } else if self.congestion.dup_acks > 3 {
            // In fast recovery, inflate window
            self.congestion.cwnd += self.mss as u32;
        }

        Ok(())
    }
}
```

---

## Congestion Control

### Slow Start and Congestion Avoidance

```rust
pub struct CongestionControl {
    cwnd: u32,           // Congestion window
    ssthresh: u32,       // Slow start threshold
    mss: u16,
    dup_acks: u32,
    state: CongestionState,
}

pub enum CongestionState {
    SlowStart,
    CongestionAvoidance,
    FastRecovery,
}

impl CongestionControl {
    pub fn new(mss: u16) -> Self {
        Self {
            cwnd: mss as u32 * 10,  // Initial window: 10 MSS (RFC 6928)
            ssthresh: u32::MAX,
            mss,
            dup_acks: 0,
            state: CongestionState::SlowStart,
        }
    }

    pub fn on_ack(&mut self, acked: u32) {
        self.dup_acks = 0;

        match self.state {
            CongestionState::SlowStart => {
                // Exponential growth
                self.cwnd += self.mss as u32;

                if self.cwnd >= self.ssthresh {
                    self.state = CongestionState::CongestionAvoidance;
                }
            }
            CongestionState::CongestionAvoidance => {
                // Linear growth: increase by MSS per RTT
                // Approximation: cwnd += MSS * MSS / cwnd
                self.cwnd += (self.mss as u32 * self.mss as u32) / self.cwnd;
            }
            CongestionState::FastRecovery => {
                // Exit fast recovery
                self.cwnd = self.ssthresh;
                self.state = CongestionState::CongestionAvoidance;
            }
        }
    }

    pub fn on_timeout(&mut self) {
        // Timeout: severe congestion
        self.ssthresh = std::cmp::max(self.cwnd / 2, 2 * self.mss as u32);
        self.cwnd = self.mss as u32;  // Reset to 1 MSS
        self.state = CongestionState::SlowStart;
        self.dup_acks = 0;
    }

    pub fn on_fast_retransmit(&mut self) {
        // 3 duplicate ACKs: moderate congestion
        self.ssthresh = std::cmp::max(self.cwnd / 2, 2 * self.mss as u32);
        self.cwnd = self.ssthresh + 3 * self.mss as u32;
        self.state = CongestionState::FastRecovery;
    }
}
```

### TCP Reno vs CUBIC (Stretch)

```rust
// TCP Reno (implemented above) is the baseline

// TCP CUBIC (modern default)
pub struct CubicCongestion {
    cwnd: u32,
    ssthresh: u32,
    mss: u16,

    // CUBIC-specific
    w_max: f64,           // Window size before last reduction
    k: f64,               // Time to reach w_max
    epoch_start: Instant,
    origin_point: f64,
    last_max_cwnd: f64,

    // Constants
    beta: f64,            // Multiplicative decrease (0.7)
    c: f64,               // Scaling constant (0.4)
}

impl CubicCongestion {
    pub fn on_ack(&mut self, rtt: Duration) {
        let t = self.epoch_start.elapsed().as_secs_f64();

        // CUBIC window function: W(t) = C * (t - K)^3 + W_max
        let target = self.c * (t - self.k).powi(3) + self.w_max;

        // TCP friendliness: ensure we're at least as aggressive as Reno
        let reno_cwnd = self.cwnd as f64 +
            (self.mss as f64 * self.mss as f64) / self.cwnd as f64;

        self.cwnd = std::cmp::max(target as u32, reno_cwnd as u32);
    }

    pub fn on_loss(&mut self) {
        // Record w_max and reduce
        self.epoch_start = Instant::now();
        self.w_max = self.cwnd as f64;
        self.ssthresh = (self.cwnd as f64 * self.beta) as u32;
        self.cwnd = self.ssthresh;

        // Calculate K: time to reach w_max
        self.k = ((self.w_max * (1.0 - self.beta)) / self.c).cbrt();
    }
}
```

---

## HTTP/1.1 Parser

### Request/Response Structures

```rust
pub struct HttpRequest {
    pub method: Method,
    pub uri: String,
    pub version: HttpVersion,
    pub headers: HashMap<String, String>,
    pub body: Option<Bytes>,
}

pub struct HttpResponse {
    pub version: HttpVersion,
    pub status: u16,
    pub reason: String,
    pub headers: HashMap<String, String>,
    pub body: Option<Bytes>,
}

pub enum HttpVersion {
    Http10,
    Http11,
}

pub enum Method {
    Get,
    Post,
    Put,
    Delete,
    Head,
    Options,
    Patch,
}
```

### Parser Implementation

```rust
pub struct HttpParser {
    state: ParserState,
    buffer: BytesMut,
    request: PartialRequest,
}

enum ParserState {
    RequestLine,
    Headers,
    Body,
    Complete,
}

impl HttpParser {
    pub fn parse(&mut self, data: &[u8]) -> Result<Option<HttpRequest>> {
        self.buffer.extend_from_slice(data);

        loop {
            match self.state {
                ParserState::RequestLine => {
                    if let Some(line_end) = self.find_crlf() {
                        let line = self.buffer.split_to(line_end + 2);
                        self.parse_request_line(&line[..line.len()-2])?;
                        self.state = ParserState::Headers;
                    } else {
                        return Ok(None);  // Need more data
                    }
                }
                ParserState::Headers => {
                    if let Some(line_end) = self.find_crlf() {
                        if line_end == 0 {
                            // Empty line: end of headers
                            self.buffer.advance(2);
                            self.state = if self.has_body() {
                                ParserState::Body
                            } else {
                                ParserState::Complete
                            };
                        } else {
                            let line = self.buffer.split_to(line_end + 2);
                            self.parse_header(&line[..line.len()-2])?;
                        }
                    } else {
                        return Ok(None);
                    }
                }
                ParserState::Body => {
                    if self.parse_body()? {
                        self.state = ParserState::Complete;
                    } else {
                        return Ok(None);
                    }
                }
                ParserState::Complete => {
                    let request = self.build_request();
                    self.reset();
                    return Ok(Some(request));
                }
            }
        }
    }

    fn parse_request_line(&mut self, line: &[u8]) -> Result<()> {
        let line = std::str::from_utf8(line)?;
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() != 3 {
            return Err(Error::BadRequest);
        }

        self.request.method = match parts[0] {
            "GET" => Method::Get,
            "POST" => Method::Post,
            "PUT" => Method::Put,
            "DELETE" => Method::Delete,
            "HEAD" => Method::Head,
            "OPTIONS" => Method::Options,
            "PATCH" => Method::Patch,
            _ => return Err(Error::MethodNotAllowed),
        };

        self.request.uri = parts[1].to_string();

        self.request.version = match parts[2] {
            "HTTP/1.0" => HttpVersion::Http10,
            "HTTP/1.1" => HttpVersion::Http11,
            _ => return Err(Error::VersionNotSupported),
        };

        Ok(())
    }

    fn parse_body(&mut self) -> Result<bool> {
        // Check for Content-Length
        if let Some(len) = self.request.headers.get("content-length") {
            let len: usize = len.parse()?;
            if self.buffer.len() >= len {
                self.request.body = Some(self.buffer.split_to(len).freeze());
                return Ok(true);
            }
        }

        // Check for Transfer-Encoding: chunked
        if self.request.headers.get("transfer-encoding")
            .map(|v| v == "chunked")
            .unwrap_or(false)
        {
            return self.parse_chunked_body();
        }

        // No body
        Ok(true)
    }

    fn parse_chunked_body(&mut self) -> Result<bool> {
        let mut body = BytesMut::new();

        loop {
            // Find chunk size line
            if let Some(line_end) = self.find_crlf() {
                let size_line = &self.buffer[..line_end];
                let size = usize::from_str_radix(
                    std::str::from_utf8(size_line)?.trim(),
                    16
                )?;

                if size == 0 {
                    // Last chunk
                    self.buffer.advance(line_end + 2);
                    // Skip trailer and final CRLF
                    if let Some(end) = self.buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                        self.buffer.advance(end + 4);
                    }
                    self.request.body = Some(body.freeze());
                    return Ok(true);
                }

                // Check if we have the whole chunk
                if self.buffer.len() >= line_end + 2 + size + 2 {
                    self.buffer.advance(line_end + 2);
                    body.extend_from_slice(&self.buffer[..size]);
                    self.buffer.advance(size + 2);  // Skip chunk data and trailing CRLF
                } else {
                    return Ok(false);  // Need more data
                }
            } else {
                return Ok(false);
            }
        }
    }
}
```

### Keep-Alive Connection Pool

```rust
pub struct ConnectionPool {
    connections: HashMap<SocketAddr, Vec<PooledConnection>>,
    max_per_host: usize,
    idle_timeout: Duration,
}

struct PooledConnection {
    connection: TcpConnection,
    last_used: Instant,
}

impl ConnectionPool {
    pub async fn get(&mut self, addr: SocketAddr) -> Result<TcpConnection> {
        // Check for existing idle connection
        if let Some(connections) = self.connections.get_mut(&addr) {
            // Remove expired connections
            connections.retain(|c| c.last_used.elapsed() < self.idle_timeout);

            // Return first available
            if let Some(pooled) = connections.pop() {
                return Ok(pooled.connection);
            }
        }

        // Create new connection
        let mut conn = TcpConnection::new();
        conn.connect(addr)?;

        Ok(conn)
    }

    pub fn put(&mut self, addr: SocketAddr, connection: TcpConnection) {
        let connections = self.connections.entry(addr).or_insert_with(Vec::new);

        if connections.len() < self.max_per_host {
            connections.push(PooledConnection {
                connection,
                last_used: Instant::now(),
            });
        }
        // Otherwise drop the connection
    }
}
```

---

## Reverse Proxy (Enterprise Feature)

```rust
pub struct ReverseProxy {
    listeners: Vec<TcpListener>,
    backends: Arc<RwLock<Vec<Backend>>>,
    connection_pool: ConnectionPool,
    load_balancer: LoadBalancer,
}

pub struct Backend {
    addr: SocketAddr,
    weight: u32,
    health: AtomicBool,
}

impl ReverseProxy {
    pub async fn handle_connection(&self, client: TcpConnection) -> Result<()> {
        // Parse HTTP request
        let request = self.read_request(&client).await?;

        // Select backend
        let backend = self.load_balancer.select(&self.backends.read())?;

        // Get or create backend connection
        let mut backend_conn = self.connection_pool.get(backend.addr).await?;

        // Forward request
        self.forward_request(&mut backend_conn, &request).await?;

        // Read response
        let response = self.read_response(&backend_conn).await?;

        // Return connection to pool if keep-alive
        if self.should_keep_alive(&request, &response) {
            self.connection_pool.put(backend.addr, backend_conn);
        }

        // Send response to client
        self.send_response(&client, &response).await?;

        Ok(())
    }
}
```

---

## Implementation Phases

### Phase 1: TCP Foundation (Week 1-2)
- [ ] TCP segment parsing/serialization
- [ ] Connection state machine
- [ ] Three-way handshake
- [ ] Basic send/receive
- [ ] TUN/TAP interface

### Phase 2: Flow Control (Week 3)
- [ ] Send and receive buffers
- [ ] Sliding window
- [ ] Window updates
- [ ] Zero window probing

### Phase 3: Reliability (Week 4)
- [ ] Retransmission timer
- [ ] RTT estimation
- [ ] Fast retransmit
- [ ] Duplicate ACK handling

### Phase 4: Congestion Control (Week 5)
- [ ] Slow start
- [ ] Congestion avoidance
- [ ] Fast recovery
- [ ] ECN (optional)

### Phase 5: HTTP/1.1 Parser (Week 6)
- [ ] Request line parsing
- [ ] Header parsing
- [ ] Content-Length body
- [ ] Chunked transfer encoding

### Phase 6: HTTP Features (Week 7)
- [ ] Keep-alive connections
- [ ] Connection pooling
- [ ] Pipelining

### Phase 7: Enterprise Features (Week 8)
- [ ] Reverse proxy
- [ ] Load balancing
- [ ] Access logging
- [ ] Health checks

---

## Testing Strategy

### Unit Tests
- Sequence number arithmetic (wrapping)
- State transitions
- Buffer operations
- HTTP parsing

### Integration Tests
```rust
#[test]
fn test_three_way_handshake() {
    let (client, server) = create_test_pair();

    // Client initiates
    client.connect(server.addr()).unwrap();

    // Run both state machines
    let segment = client.take_outgoing();  // SYN
    server.handle_segment(segment).unwrap();

    let segment = server.take_outgoing();  // SYN+ACK
    client.handle_segment(segment).unwrap();

    let segment = client.take_outgoing();  // ACK
    server.handle_segment(segment).unwrap();

    assert_eq!(client.state(), TcpState::Established);
    assert_eq!(server.state(), TcpState::Established);
}

#[test]
fn test_data_transfer() {
    let (mut client, mut server) = established_connection();

    // Send data
    client.send(b"Hello, World!").unwrap();

    // Transfer segments
    while let Some(seg) = client.take_outgoing() {
        server.handle_segment(seg).unwrap();
    }

    while let Some(seg) = server.take_outgoing() {
        client.handle_segment(seg).unwrap();
    }

    // Receive data
    let received = server.recv().unwrap();
    assert_eq!(&received, b"Hello, World!");
}
```

### Network Simulation
- Packet loss
- Reordering
- Latency
- Bandwidth limits

---

## Stretch Goals

### HTTP/2
- Binary framing
- Streams and multiplexing
- HPACK header compression
- Flow control per stream

### TLS Handshake
- Client/Server Hello
- Key exchange
- Certificate validation

### TCP BBR Congestion Control
- Bottleneck Bandwidth
- Round-trip propagation time
- Pacing-based approach

---

## Dependencies

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
bytes = "1"
bitflags = "2"
tun-tap = "0.1"                  # TUN/TAP interface
etherparse = "0.13"              # Packet parsing
rand = "0.8"
```

---

## References

- [RFC 793: TCP](https://www.rfc-editor.org/rfc/rfc793)
- [RFC 5681: TCP Congestion Control](https://www.rfc-editor.org/rfc/rfc5681)
- [RFC 6298: TCP RTO](https://www.rfc-editor.org/rfc/rfc6298)
- [RFC 7230: HTTP/1.1 Message Syntax](https://www.rfc-editor.org/rfc/rfc7230)
- [TCP/IP Illustrated, Volume 1](https://www.amazon.com/TCP-Illustrated-Protocols-Addison-Wesley-Professional/dp/0321336313)
