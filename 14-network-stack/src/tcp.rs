//! TCP protocol implementation.
//!
//! Implements TCP state machine, three-way handshake, flow control,
//! and congestion control algorithms.

use crate::{Error, Result};
use bitflags::bitflags;
use bytes::{Buf, BufMut, BytesMut};
use parking_lot::Mutex;
use std::collections::{BTreeMap, VecDeque};
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// TCP sequence number.
pub type SeqNum = u32;

/// TCP acknowledgment number.
pub type AckNum = u32;

/// Window size.
pub type WindowSize = u16;

bitflags! {
    /// TCP flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// TCP connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcpState {
    /// No connection.
    Closed,
    /// Waiting for connection request.
    Listen,
    /// SYN sent, waiting for SYN-ACK.
    SynSent,
    /// SYN received, sent SYN-ACK.
    SynReceived,
    /// Connection established.
    Established,
    /// FIN sent, waiting for ACK.
    FinWait1,
    /// FIN sent, ACK received, waiting for FIN.
    FinWait2,
    /// Received FIN while in Established.
    CloseWait,
    /// FIN sent after receiving FIN.
    Closing,
    /// FIN sent and received, waiting for ACK.
    LastAck,
    /// Waiting for timeout after sending final ACK.
    TimeWait,
}

/// TCP segment header.
#[derive(Debug, Clone)]
pub struct TcpSegment {
    /// Source port.
    pub src_port: u16,
    /// Destination port.
    pub dst_port: u16,
    /// Sequence number.
    pub seq_num: SeqNum,
    /// Acknowledgment number.
    pub ack_num: AckNum,
    /// Data offset (header length in 32-bit words).
    pub data_offset: u8,
    /// Flags.
    pub flags: TcpFlags,
    /// Window size.
    pub window: WindowSize,
    /// Checksum.
    pub checksum: u16,
    /// Urgent pointer.
    pub urgent_ptr: u16,
    /// Options.
    pub options: Vec<TcpOption>,
    /// Payload data.
    pub payload: BytesMut,
}

impl TcpSegment {
    /// Create a new TCP segment.
    pub fn new(src_port: u16, dst_port: u16) -> Self {
        Self {
            src_port,
            dst_port,
            seq_num: 0,
            ack_num: 0,
            data_offset: 5, // Minimum header size
            flags: TcpFlags::empty(),
            window: 65535,
            checksum: 0,
            urgent_ptr: 0,
            options: Vec::new(),
            payload: BytesMut::new(),
        }
    }

    /// Parse a TCP segment from bytes.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 20 {
            return Err(Error::Parse("TCP segment too short".into()));
        }

        let src_port = u16::from_be_bytes([data[0], data[1]]);
        let dst_port = u16::from_be_bytes([data[2], data[3]]);
        let seq_num = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        let ack_num = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
        let data_offset = (data[12] >> 4) & 0x0F;
        let flags = TcpFlags::from_bits_truncate(data[13]);
        let window = u16::from_be_bytes([data[14], data[15]]);
        let checksum = u16::from_be_bytes([data[16], data[17]]);
        let urgent_ptr = u16::from_be_bytes([data[18], data[19]]);

        let header_len = (data_offset as usize) * 4;
        if data.len() < header_len {
            return Err(Error::Parse("Invalid data offset".into()));
        }

        // Parse options
        let mut options = Vec::new();
        let mut opt_offset = 20;
        while opt_offset < header_len {
            match data[opt_offset] {
                0 => break, // End of options
                1 => opt_offset += 1, // NOP
                2 => {
                    // MSS
                    if opt_offset + 4 <= header_len {
                        let mss = u16::from_be_bytes([data[opt_offset + 2], data[opt_offset + 3]]);
                        options.push(TcpOption::Mss(mss));
                    }
                    opt_offset += 4;
                }
                3 => {
                    // Window Scale
                    if opt_offset + 3 <= header_len {
                        options.push(TcpOption::WindowScale(data[opt_offset + 2]));
                    }
                    opt_offset += 3;
                }
                8 => {
                    // Timestamps
                    if opt_offset + 10 <= header_len {
                        let ts_val = u32::from_be_bytes([
                            data[opt_offset + 2],
                            data[opt_offset + 3],
                            data[opt_offset + 4],
                            data[opt_offset + 5],
                        ]);
                        let ts_ecr = u32::from_be_bytes([
                            data[opt_offset + 6],
                            data[opt_offset + 7],
                            data[opt_offset + 8],
                            data[opt_offset + 9],
                        ]);
                        options.push(TcpOption::Timestamps { ts_val, ts_ecr });
                    }
                    opt_offset += 10;
                }
                _ => {
                    if opt_offset + 1 < header_len {
                        let len = data[opt_offset + 1] as usize;
                        opt_offset += len.max(2);
                    } else {
                        break;
                    }
                }
            }
        }

        let payload = BytesMut::from(&data[header_len..]);

        Ok(Self {
            src_port,
            dst_port,
            seq_num,
            ack_num,
            data_offset,
            flags,
            window,
            checksum,
            urgent_ptr,
            options,
            payload,
        })
    }

    /// Serialize the segment to bytes.
    pub fn serialize(&self) -> BytesMut {
        let mut buf = BytesMut::with_capacity(20 + self.payload.len());

        buf.put_u16(self.src_port);
        buf.put_u16(self.dst_port);
        buf.put_u32(self.seq_num);
        buf.put_u32(self.ack_num);
        buf.put_u8((self.data_offset << 4) | 0);
        buf.put_u8(self.flags.bits());
        buf.put_u16(self.window);
        buf.put_u16(self.checksum);
        buf.put_u16(self.urgent_ptr);

        // Add options padding if needed
        let header_len = (self.data_offset as usize) * 4;
        while buf.len() < header_len {
            buf.put_u8(0);
        }

        buf.extend_from_slice(&self.payload);
        buf
    }

    /// Calculate checksum.
    pub fn calculate_checksum(&self, src_ip: IpAddr, dst_ip: IpAddr) -> u16 {
        let mut sum: u32 = 0;

        // Pseudo header
        match (src_ip, dst_ip) {
            (IpAddr::V4(src), IpAddr::V4(dst)) => {
                let src_bytes = src.octets();
                let dst_bytes = dst.octets();
                sum += u16::from_be_bytes([src_bytes[0], src_bytes[1]]) as u32;
                sum += u16::from_be_bytes([src_bytes[2], src_bytes[3]]) as u32;
                sum += u16::from_be_bytes([dst_bytes[0], dst_bytes[1]]) as u32;
                sum += u16::from_be_bytes([dst_bytes[2], dst_bytes[3]]) as u32;
            }
            _ => {}
        }

        sum += 6u32; // TCP protocol number
        let tcp_len = (self.data_offset as usize) * 4 + self.payload.len();
        sum += tcp_len as u32;

        // TCP header and data
        let data = self.serialize();
        let mut i = 0;
        while i + 1 < data.len() {
            sum += u16::from_be_bytes([data[i], data[i + 1]]) as u32;
            i += 2;
        }
        if i < data.len() {
            sum += (data[i] as u32) << 8;
        }

        // Fold 32-bit sum to 16 bits
        while sum >> 16 != 0 {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }

        !sum as u16
    }
}

/// TCP option.
#[derive(Debug, Clone)]
pub enum TcpOption {
    /// Maximum Segment Size.
    Mss(u16),
    /// Window Scale factor.
    WindowScale(u8),
    /// SACK permitted.
    SackPermitted,
    /// Selective Acknowledgment.
    Sack(Vec<(SeqNum, SeqNum)>),
    /// Timestamps.
    Timestamps { ts_val: u32, ts_ecr: u32 },
}

/// Send sequence space variables.
#[derive(Debug, Clone)]
pub struct SendSequenceSpace {
    /// Send unacknowledged.
    pub una: SeqNum,
    /// Send next.
    pub nxt: SeqNum,
    /// Send window.
    pub wnd: WindowSize,
    /// Segment sequence number used for last window update.
    pub wl1: SeqNum,
    /// Segment acknowledgment number used for last window update.
    pub wl2: AckNum,
    /// Initial send sequence number.
    pub iss: SeqNum,
}

impl SendSequenceSpace {
    /// Create new send sequence space.
    pub fn new(iss: SeqNum) -> Self {
        Self {
            una: iss,
            nxt: iss,
            wnd: 0,
            wl1: 0,
            wl2: 0,
            iss,
        }
    }
}

/// Receive sequence space variables.
#[derive(Debug, Clone)]
pub struct RecvSequenceSpace {
    /// Receive next.
    pub nxt: SeqNum,
    /// Receive window.
    pub wnd: WindowSize,
    /// Initial receive sequence number.
    pub irs: SeqNum,
}

impl RecvSequenceSpace {
    /// Create new receive sequence space.
    pub fn new(irs: SeqNum) -> Self {
        Self {
            nxt: irs.wrapping_add(1),
            wnd: 65535,
            irs,
        }
    }
}

/// Congestion control state.
#[derive(Debug, Clone)]
pub struct CongestionControl {
    /// Congestion window.
    pub cwnd: u32,
    /// Slow start threshold.
    pub ssthresh: u32,
    /// Round-trip time estimate.
    pub srtt: Duration,
    /// RTT variance.
    pub rttvar: Duration,
    /// Retransmission timeout.
    pub rto: Duration,
    /// Duplicate ACK count.
    pub dup_acks: u32,
    /// In fast recovery.
    pub fast_recovery: bool,
}

impl Default for CongestionControl {
    fn default() -> Self {
        Self {
            cwnd: 10 * 1460, // Initial window: 10 MSS
            ssthresh: u32::MAX,
            srtt: Duration::from_millis(100),
            rttvar: Duration::from_millis(50),
            rto: Duration::from_secs(1),
            dup_acks: 0,
            fast_recovery: false,
        }
    }
}

impl CongestionControl {
    /// Update RTT estimate.
    pub fn update_rtt(&mut self, measured_rtt: Duration) {
        // RFC 6298 algorithm
        let alpha = 0.125;
        let beta = 0.25;

        let diff = if measured_rtt > self.srtt {
            measured_rtt - self.srtt
        } else {
            self.srtt - measured_rtt
        };

        self.rttvar = Duration::from_secs_f64(
            (1.0 - beta) * self.rttvar.as_secs_f64() + beta * diff.as_secs_f64(),
        );

        self.srtt = Duration::from_secs_f64(
            (1.0 - alpha) * self.srtt.as_secs_f64() + alpha * measured_rtt.as_secs_f64(),
        );

        self.rto = self.srtt + 4 * self.rttvar;

        // Clamp RTO
        if self.rto < Duration::from_millis(200) {
            self.rto = Duration::from_millis(200);
        }
        if self.rto > Duration::from_secs(60) {
            self.rto = Duration::from_secs(60);
        }
    }

    /// Handle ACK (congestion avoidance).
    pub fn on_ack(&mut self, bytes_acked: u32) {
        if self.fast_recovery {
            // Exit fast recovery
            self.cwnd = self.ssthresh;
            self.fast_recovery = false;
        } else if self.cwnd < self.ssthresh {
            // Slow start
            self.cwnd += bytes_acked;
        } else {
            // Congestion avoidance
            self.cwnd += (1460 * bytes_acked) / self.cwnd;
        }
        self.dup_acks = 0;
    }

    /// Handle duplicate ACK.
    pub fn on_dup_ack(&mut self) {
        self.dup_acks += 1;

        if self.dup_acks == 3 && !self.fast_recovery {
            // Enter fast recovery
            self.ssthresh = self.cwnd / 2;
            self.cwnd = self.ssthresh + 3 * 1460;
            self.fast_recovery = true;
        } else if self.fast_recovery {
            // Inflate window
            self.cwnd += 1460;
        }
    }

    /// Handle timeout.
    pub fn on_timeout(&mut self) {
        const MAX_RTO: Duration = Duration::from_secs(60);

        self.ssthresh = self.cwnd / 2;
        self.cwnd = 1460; // Reset to 1 MSS
        self.dup_acks = 0;
        self.fast_recovery = false;
        self.rto *= 2; // Exponential backoff
        if self.rto > MAX_RTO {
            self.rto = MAX_RTO;
        }
    }
}

/// Retransmission queue entry.
#[derive(Debug, Clone)]
pub struct RetransmitEntry {
    /// Sequence number.
    pub seq: SeqNum,
    /// Segment data.
    pub data: BytesMut,
    /// Send time.
    pub sent_at: Instant,
    /// Retransmit count.
    pub retransmits: u32,
}

/// TCP connection.
pub struct TcpConnection {
    /// Connection state.
    pub state: TcpState,
    /// Local address.
    pub local_addr: SocketAddr,
    /// Remote address.
    pub remote_addr: SocketAddr,
    /// Send sequence space.
    pub send: SendSequenceSpace,
    /// Receive sequence space.
    pub recv: RecvSequenceSpace,
    /// Congestion control.
    pub congestion: CongestionControl,
    /// Send buffer.
    pub send_buffer: VecDeque<u8>,
    /// Receive buffer.
    pub recv_buffer: VecDeque<u8>,
    /// Retransmission queue.
    pub retransmit_queue: BTreeMap<SeqNum, RetransmitEntry>,
    /// Out-of-order segments.
    pub ooo_segments: BTreeMap<SeqNum, BytesMut>,
    /// Maximum segment size.
    pub mss: u16,
    /// Window scale factor.
    pub window_scale: u8,
    /// Last activity time.
    pub last_activity: Instant,
}

impl TcpConnection {
    /// Create a new TCP connection.
    pub fn new(local_addr: SocketAddr, remote_addr: SocketAddr) -> Self {
        let iss = rand::random::<u32>();

        Self {
            state: TcpState::Closed,
            local_addr,
            remote_addr,
            send: SendSequenceSpace::new(iss),
            recv: RecvSequenceSpace::new(0),
            congestion: CongestionControl::default(),
            send_buffer: VecDeque::new(),
            recv_buffer: VecDeque::new(),
            retransmit_queue: BTreeMap::new(),
            ooo_segments: BTreeMap::new(),
            mss: 1460,
            window_scale: 0,
            last_activity: Instant::now(),
        }
    }

    /// Initiate connection (active open).
    pub fn connect(&mut self) -> Result<TcpSegment> {
        if self.state != TcpState::Closed {
            return Err(Error::InvalidState("Cannot connect in current state".into()));
        }

        self.state = TcpState::SynSent;

        // Create SYN segment
        let mut syn = TcpSegment::new(
            self.local_addr.port(),
            self.remote_addr.port(),
        );
        syn.seq_num = self.send.iss;
        syn.flags = TcpFlags::SYN;
        syn.options.push(TcpOption::Mss(self.mss));
        syn.options.push(TcpOption::WindowScale(7));

        self.send.nxt = self.send.iss.wrapping_add(1);

        Ok(syn)
    }

    /// Accept connection (passive open).
    pub fn listen(&mut self) -> Result<()> {
        if self.state != TcpState::Closed {
            return Err(Error::InvalidState("Cannot listen in current state".into()));
        }
        self.state = TcpState::Listen;
        Ok(())
    }

    /// Process incoming segment.
    pub fn on_segment(&mut self, segment: TcpSegment) -> Result<Option<TcpSegment>> {
        self.last_activity = Instant::now();

        match self.state {
            TcpState::Closed => Ok(None),

            TcpState::Listen => {
                if segment.flags.contains(TcpFlags::SYN) {
                    // Received SYN, send SYN-ACK
                    self.recv = RecvSequenceSpace::new(segment.seq_num);
                    self.state = TcpState::SynReceived;

                    let mut syn_ack = TcpSegment::new(
                        self.local_addr.port(),
                        self.remote_addr.port(),
                    );
                    syn_ack.seq_num = self.send.iss;
                    syn_ack.ack_num = self.recv.nxt;
                    syn_ack.flags = TcpFlags::SYN | TcpFlags::ACK;
                    syn_ack.window = self.recv.wnd;
                    syn_ack.options.push(TcpOption::Mss(self.mss));

                    self.send.nxt = self.send.iss.wrapping_add(1);

                    // Process MSS option
                    for opt in &segment.options {
                        if let TcpOption::Mss(mss) = opt {
                            self.mss = self.mss.min(*mss);
                        }
                    }

                    Ok(Some(syn_ack))
                } else {
                    Ok(None)
                }
            }

            TcpState::SynSent => {
                if segment.flags.contains(TcpFlags::SYN | TcpFlags::ACK) {
                    // Received SYN-ACK
                    if segment.ack_num != self.send.nxt {
                        return Err(Error::Protocol("Invalid ACK in SYN-ACK".into()));
                    }

                    self.recv = RecvSequenceSpace::new(segment.seq_num);
                    self.send.una = segment.ack_num;
                    self.send.wnd = segment.window;
                    self.state = TcpState::Established;

                    // Send ACK
                    let mut ack = TcpSegment::new(
                        self.local_addr.port(),
                        self.remote_addr.port(),
                    );
                    ack.seq_num = self.send.nxt;
                    ack.ack_num = self.recv.nxt;
                    ack.flags = TcpFlags::ACK;
                    ack.window = self.recv.wnd;

                    // Process MSS option
                    for opt in &segment.options {
                        if let TcpOption::Mss(mss) = opt {
                            self.mss = self.mss.min(*mss);
                        }
                    }

                    Ok(Some(ack))
                } else if segment.flags.contains(TcpFlags::SYN) {
                    // Simultaneous open
                    self.recv = RecvSequenceSpace::new(segment.seq_num);
                    self.state = TcpState::SynReceived;

                    let mut syn_ack = TcpSegment::new(
                        self.local_addr.port(),
                        self.remote_addr.port(),
                    );
                    syn_ack.seq_num = self.send.iss;
                    syn_ack.ack_num = self.recv.nxt;
                    syn_ack.flags = TcpFlags::SYN | TcpFlags::ACK;
                    syn_ack.window = self.recv.wnd;

                    Ok(Some(syn_ack))
                } else {
                    Ok(None)
                }
            }

            TcpState::SynReceived => {
                if segment.flags.contains(TcpFlags::ACK) {
                    if segment.ack_num == self.send.nxt {
                        self.send.una = segment.ack_num;
                        self.send.wnd = segment.window;
                        self.state = TcpState::Established;
                    }
                }
                Ok(None)
            }

            TcpState::Established => {
                let mut response = None;

                // Process ACK
                if segment.flags.contains(TcpFlags::ACK) {
                    self.process_ack(segment.ack_num, segment.window)?;
                }

                // Process data
                if !segment.payload.is_empty() {
                    response = Some(self.process_data(segment.seq_num, segment.payload)?);
                }

                // Process FIN
                if segment.flags.contains(TcpFlags::FIN) {
                    self.recv.nxt = segment.seq_num.wrapping_add(1);
                    self.state = TcpState::CloseWait;

                    let mut ack = TcpSegment::new(
                        self.local_addr.port(),
                        self.remote_addr.port(),
                    );
                    ack.seq_num = self.send.nxt;
                    ack.ack_num = self.recv.nxt;
                    ack.flags = TcpFlags::ACK;
                    ack.window = self.recv.wnd;
                    response = Some(ack);
                }

                Ok(response)
            }

            TcpState::FinWait1 => {
                if segment.flags.contains(TcpFlags::ACK) {
                    if segment.ack_num == self.send.nxt {
                        self.state = TcpState::FinWait2;
                    }
                }
                if segment.flags.contains(TcpFlags::FIN) {
                    self.recv.nxt = segment.seq_num.wrapping_add(1);
                    if self.state == TcpState::FinWait2 {
                        self.state = TcpState::TimeWait;
                    } else {
                        self.state = TcpState::Closing;
                    }

                    let mut ack = TcpSegment::new(
                        self.local_addr.port(),
                        self.remote_addr.port(),
                    );
                    ack.seq_num = self.send.nxt;
                    ack.ack_num = self.recv.nxt;
                    ack.flags = TcpFlags::ACK;
                    ack.window = self.recv.wnd;
                    return Ok(Some(ack));
                }
                Ok(None)
            }

            TcpState::FinWait2 => {
                if segment.flags.contains(TcpFlags::FIN) {
                    self.recv.nxt = segment.seq_num.wrapping_add(1);
                    self.state = TcpState::TimeWait;

                    let mut ack = TcpSegment::new(
                        self.local_addr.port(),
                        self.remote_addr.port(),
                    );
                    ack.seq_num = self.send.nxt;
                    ack.ack_num = self.recv.nxt;
                    ack.flags = TcpFlags::ACK;
                    ack.window = self.recv.wnd;
                    return Ok(Some(ack));
                }
                Ok(None)
            }

            TcpState::CloseWait => Ok(None),

            TcpState::Closing => {
                if segment.flags.contains(TcpFlags::ACK) {
                    self.state = TcpState::TimeWait;
                }
                Ok(None)
            }

            TcpState::LastAck => {
                if segment.flags.contains(TcpFlags::ACK) {
                    self.state = TcpState::Closed;
                }
                Ok(None)
            }

            TcpState::TimeWait => Ok(None),
        }
    }

    /// Process acknowledgment.
    fn process_ack(&mut self, ack_num: AckNum, window: WindowSize) -> Result<()> {
        // Check if ACK is valid
        if !self.is_valid_ack(ack_num) {
            return Ok(());
        }

        if ack_num == self.send.una {
            // Duplicate ACK
            self.congestion.on_dup_ack();
        } else {
            // New ACK
            let bytes_acked = ack_num.wrapping_sub(self.send.una);
            self.send.una = ack_num;
            self.send.wnd = window;

            // Update congestion control
            self.congestion.on_ack(bytes_acked);

            // Remove acknowledged segments from retransmit queue
            // Inline sequence comparison to avoid borrowing self in closure
            self.retransmit_queue.retain(|seq, entry| {
                let end = seq.wrapping_add(entry.data.len() as u32);
                // Check if end is NOT before or equal to ack_num (i.e., end > ack_num)
                // is_seq_before_or_equal: seq1 == seq2 || (seq1 - seq2 as i32) < 0
                let is_before_or_equal = end == ack_num || (end.wrapping_sub(ack_num) as i32) < 0;
                !is_before_or_equal
            });

            // Calculate RTT if possible
            if let Some(entry) = self.retransmit_queue.values().next() {
                if entry.retransmits == 0 {
                    let rtt = entry.sent_at.elapsed();
                    self.congestion.update_rtt(rtt);
                }
            }
        }

        Ok(())
    }

    /// Process incoming data.
    fn process_data(&mut self, seq: SeqNum, data: BytesMut) -> Result<TcpSegment> {
        if seq == self.recv.nxt {
            // In-order data
            self.recv_buffer.extend(data.iter());
            self.recv.nxt = self.recv.nxt.wrapping_add(data.len() as u32);

            // Check for out-of-order segments that can now be delivered
            while let Some(entry) = self.ooo_segments.remove(&self.recv.nxt) {
                self.recv_buffer.extend(entry.iter());
                self.recv.nxt = self.recv.nxt.wrapping_add(entry.len() as u32);
            }
        } else if self.is_seq_after(seq, self.recv.nxt) {
            // Out-of-order data
            self.ooo_segments.insert(seq, data);
        }

        // Send ACK
        let mut ack = TcpSegment::new(
            self.local_addr.port(),
            self.remote_addr.port(),
        );
        ack.seq_num = self.send.nxt;
        ack.ack_num = self.recv.nxt;
        ack.flags = TcpFlags::ACK;
        ack.window = self.available_window();

        Ok(ack)
    }

    /// Send data.
    pub fn send(&mut self, data: &[u8]) -> Result<Vec<TcpSegment>> {
        if self.state != TcpState::Established {
            return Err(Error::InvalidState("Cannot send in current state".into()));
        }

        self.send_buffer.extend(data.iter());
        self.flush_send_buffer()
    }

    /// Flush send buffer.
    fn flush_send_buffer(&mut self) -> Result<Vec<TcpSegment>> {
        let mut segments = Vec::new();

        while !self.send_buffer.is_empty() {
            // Calculate available window
            let in_flight = self.send.nxt.wrapping_sub(self.send.una);
            let window = (self.send.wnd as u32).min(self.congestion.cwnd);

            if in_flight >= window {
                break; // Window full
            }

            let available = window - in_flight;
            let to_send = available.min(self.mss as u32).min(self.send_buffer.len() as u32);

            if to_send == 0 {
                break;
            }

            // Extract data from buffer
            let data: Vec<u8> = self.send_buffer.drain(..to_send as usize).collect();

            // Create segment
            let mut segment = TcpSegment::new(
                self.local_addr.port(),
                self.remote_addr.port(),
            );
            segment.seq_num = self.send.nxt;
            segment.ack_num = self.recv.nxt;
            segment.flags = TcpFlags::ACK | TcpFlags::PSH;
            segment.window = self.available_window();
            segment.payload = BytesMut::from(&data[..]);

            // Add to retransmit queue
            self.retransmit_queue.insert(
                self.send.nxt,
                RetransmitEntry {
                    seq: self.send.nxt,
                    data: segment.payload.clone(),
                    sent_at: Instant::now(),
                    retransmits: 0,
                },
            );

            self.send.nxt = self.send.nxt.wrapping_add(to_send);
            segments.push(segment);
        }

        Ok(segments)
    }

    /// Close connection.
    pub fn close(&mut self) -> Result<Option<TcpSegment>> {
        match self.state {
            TcpState::Established => {
                self.state = TcpState::FinWait1;
            }
            TcpState::CloseWait => {
                self.state = TcpState::LastAck;
            }
            _ => {
                return Err(Error::InvalidState("Cannot close in current state".into()));
            }
        }

        // Send FIN
        let mut fin = TcpSegment::new(
            self.local_addr.port(),
            self.remote_addr.port(),
        );
        fin.seq_num = self.send.nxt;
        fin.ack_num = self.recv.nxt;
        fin.flags = TcpFlags::FIN | TcpFlags::ACK;
        fin.window = self.available_window();

        self.send.nxt = self.send.nxt.wrapping_add(1);

        Ok(Some(fin))
    }

    /// Get segments that need retransmission.
    pub fn get_retransmits(&mut self) -> Vec<TcpSegment> {
        let mut segments = Vec::new();
        let now = Instant::now();

        // Pre-calculate values needed in the loop to avoid borrowing issues
        let local_port = self.local_addr.port();
        let remote_port = self.remote_addr.port();
        let recv_nxt = self.recv.nxt;
        let window = self.available_window();
        let rto = self.congestion.rto;

        for entry in self.retransmit_queue.values_mut() {
            if now.duration_since(entry.sent_at) > rto {
                // Timeout - need to retransmit
                // Note: congestion.on_timeout() called after loop to avoid borrow issues

                let mut segment = TcpSegment::new(local_port, remote_port);
                segment.seq_num = entry.seq;
                segment.ack_num = recv_nxt;
                segment.flags = TcpFlags::ACK | TcpFlags::PSH;
                segment.window = window;
                segment.payload = entry.data.clone();

                entry.sent_at = now;
                entry.retransmits += 1;

                segments.push(segment);
            }
        }

        // Update congestion control after loop
        if !segments.is_empty() {
            self.congestion.on_timeout();
        }

        segments
    }

    /// Read received data.
    pub fn read(&mut self) -> Vec<u8> {
        self.recv_buffer.drain(..).collect()
    }

    /// Available receive window.
    fn available_window(&self) -> u16 {
        let buffered = self.recv_buffer.len() as u32;
        let window = 65535u32.saturating_sub(buffered);
        window.min(u16::MAX as u32) as u16
    }

    /// Check if sequence number is valid ACK.
    fn is_valid_ack(&self, ack: AckNum) -> bool {
        // ACK must be within [SND.UNA, SND.NXT]
        self.is_seq_between_or_equal(ack, self.send.una, self.send.nxt)
    }

    /// Check if seq1 is before seq2 (with wrap-around).
    fn is_seq_before(&self, seq1: SeqNum, seq2: SeqNum) -> bool {
        (seq1.wrapping_sub(seq2) as i32) < 0
    }

    /// Check if seq1 is before or equal to seq2.
    fn is_seq_before_or_equal(&self, seq1: SeqNum, seq2: SeqNum) -> bool {
        seq1 == seq2 || self.is_seq_before(seq1, seq2)
    }

    /// Check if seq1 is after seq2.
    fn is_seq_after(&self, seq1: SeqNum, seq2: SeqNum) -> bool {
        (seq1.wrapping_sub(seq2) as i32) > 0
    }

    /// Check if seq is between start and end (inclusive).
    fn is_seq_between_or_equal(&self, seq: SeqNum, start: SeqNum, end: SeqNum) -> bool {
        if start <= end {
            seq >= start && seq <= end
        } else {
            seq >= start || seq <= end
        }
    }
}

/// TCP listener for accepting connections.
pub struct TcpListener {
    /// Listening address.
    pub local_addr: SocketAddr,
    /// Pending connections.
    pending: Arc<Mutex<VecDeque<TcpConnection>>>,
    /// Backlog size.
    backlog: usize,
}

impl TcpListener {
    /// Create a new TCP listener.
    pub fn new(local_addr: SocketAddr, backlog: usize) -> Self {
        Self {
            local_addr,
            pending: Arc::new(Mutex::new(VecDeque::with_capacity(backlog))),
            backlog,
        }
    }

    /// Accept a new connection.
    pub fn accept(&self) -> Option<TcpConnection> {
        self.pending.lock().pop_front()
    }

    /// Queue a pending connection.
    pub fn queue_connection(&self, conn: TcpConnection) -> bool {
        let mut pending = self.pending.lock();
        if pending.len() < self.backlog {
            pending.push_back(conn);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tcp_segment_parse() {
        let data = [
            0x00, 0x50, // src port (80)
            0x01, 0xBB, // dst port (443)
            0x00, 0x00, 0x00, 0x01, // seq
            0x00, 0x00, 0x00, 0x02, // ack
            0x50, 0x12, // data offset + flags (SYN|ACK)
            0xFF, 0xFF, // window
            0x00, 0x00, // checksum
            0x00, 0x00, // urgent
        ];

        let segment = TcpSegment::parse(&data).unwrap();
        assert_eq!(segment.src_port, 80);
        assert_eq!(segment.dst_port, 443);
        assert_eq!(segment.seq_num, 1);
        assert_eq!(segment.ack_num, 2);
        assert!(segment.flags.contains(TcpFlags::SYN | TcpFlags::ACK));
    }

    #[test]
    fn test_three_way_handshake() {
        let client_addr: SocketAddr = "127.0.0.1:12345".parse().unwrap();
        let server_addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        let mut client = TcpConnection::new(client_addr, server_addr);
        let mut server = TcpConnection::new(server_addr, client_addr);

        // Server listens
        server.listen().unwrap();

        // Client sends SYN
        let syn = client.connect().unwrap();
        assert_eq!(client.state, TcpState::SynSent);

        // Server receives SYN, sends SYN-ACK
        let syn_ack = server.on_segment(syn).unwrap().unwrap();
        assert_eq!(server.state, TcpState::SynReceived);

        // Client receives SYN-ACK, sends ACK
        let ack = client.on_segment(syn_ack).unwrap().unwrap();
        assert_eq!(client.state, TcpState::Established);

        // Server receives ACK
        server.on_segment(ack).unwrap();
        assert_eq!(server.state, TcpState::Established);
    }

    #[test]
    fn test_congestion_control() {
        let mut cc = CongestionControl::default();

        // Test slow start
        let initial_cwnd = cc.cwnd;
        cc.on_ack(1460);
        assert!(cc.cwnd > initial_cwnd);

        // Test timeout
        cc.on_timeout();
        assert_eq!(cc.cwnd, 1460);
    }
}
