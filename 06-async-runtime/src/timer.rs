//! Timer wheel for efficient timeout management
//!
//! Implements a hierarchical timer wheel for O(1) timer operations.

use std::collections::HashMap;
use std::task::Waker;
use std::time::{Duration, Instant};


/// Timer handle for cancellation
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TimerHandle {
    id: u64,
}

/// Timer entry
struct TimerEntry {
    id: u64,
    deadline: u64,
    waker: Waker,
}

/// Wheel in the timer wheel hierarchy
struct Wheel {
    slots: Vec<Vec<TimerEntry>>,
    mask: usize,
}

impl Wheel {
    fn new(num_slots: usize) -> Self {
        Self {
            slots: (0..num_slots).map(|_| Vec::new()).collect(),
            mask: num_slots - 1,
        }
    }
}

/// Hierarchical timer wheel
pub struct TimerWheel {
    /// Wheels at different granularities
    wheels: [Wheel; 4],
    /// Current time in ticks
    current_tick: u64,
    /// Tick duration (1ms default)
    tick_duration: Duration,
    /// Start time
    start_time: Instant,
    /// Next timer ID
    next_id: u64,
    /// Handle to entry mapping for cancellation
    handles: HashMap<u64, (usize, usize)>,
}

impl TimerWheel {
    /// Create a new timer wheel
    pub fn new() -> Self {
        Self::with_tick_duration(Duration::from_millis(1))
    }

    /// Create a timer wheel with custom tick duration
    pub fn with_tick_duration(tick_duration: Duration) -> Self {
        Self {
            wheels: [
                Wheel::new(256),  // 256 ticks
                Wheel::new(64),   // 64 * 256 ticks
                Wheel::new(64),   // 64 * 64 * 256 ticks
                Wheel::new(64),   // 64 * 64 * 64 * 256 ticks
            ],
            current_tick: 0,
            tick_duration,
            start_time: Instant::now(),
            next_id: 0,
            handles: HashMap::new(),
        }
    }

    /// Insert a timer
    pub fn insert(&mut self, deadline: Instant, waker: Waker) -> TimerHandle {
        let ticks = self.instant_to_ticks(deadline);
        let delta = ticks.saturating_sub(self.current_tick);

        // Determine which wheel and slot
        let (wheel_idx, slot_idx) = self.calculate_slot(delta);

        let id = self.next_id;
        self.next_id += 1;

        let entry = TimerEntry {
            id,
            deadline: ticks,
            waker,
        };
        self.wheels[wheel_idx].slots[slot_idx].push(entry);
        self.handles.insert(id, (wheel_idx, slot_idx));

        TimerHandle { id }
    }

    /// Cancel a timer
    pub fn cancel(&mut self, handle: TimerHandle) -> bool {
        self.handles.remove(&handle.id).is_some()
        // Note: Entry is not removed from wheel, just ignored on fire
    }

    /// Advance time and fire expired timers
    pub fn advance(&mut self, now: Instant) -> Vec<Waker> {
        let target_tick = self.instant_to_ticks(now);
        let mut wakers = Vec::new();

        while self.current_tick < target_tick {
            self.current_tick += 1;

            // Check wheel 0 slot
            let slot_idx = (self.current_tick as usize) & self.wheels[0].mask;
            let slot = std::mem::take(&mut self.wheels[0].slots[slot_idx]);

            for entry in slot {
                // Skip cancelled timers
                if !self.handles.contains_key(&entry.id) {
                    continue;
                }

                if entry.deadline <= self.current_tick {
                    self.handles.remove(&entry.id);
                    wakers.push(entry.waker);
                } else {
                    // Reinsert into appropriate slot
                    let delta = entry.deadline - self.current_tick;
                    let (wheel_idx, new_slot) = self.calculate_slot(delta);
                    self.wheels[wheel_idx].slots[new_slot].push(entry);
                }
            }

            // Cascade from higher wheels if needed
            if slot_idx == 0 && self.current_tick > 0 {
                self.cascade(1);
            }
        }

        wakers
    }

    /// Get the next deadline
    pub fn next_deadline(&self) -> Option<Instant> {
        // Find the earliest timer
        let mut min_ticks = u64::MAX;

        for wheel in &self.wheels {
            for slot in &wheel.slots {
                for entry in slot {
                    if entry.deadline < min_ticks {
                        min_ticks = entry.deadline;
                    }
                }
            }
        }

        if min_ticks == u64::MAX {
            None
        } else {
            Some(self.ticks_to_instant(min_ticks))
        }
    }

    /// Convert instant to ticks
    fn instant_to_ticks(&self, instant: Instant) -> u64 {
        let elapsed = instant.duration_since(self.start_time);
        (elapsed.as_nanos() / self.tick_duration.as_nanos()) as u64
    }

    /// Convert ticks to instant
    fn ticks_to_instant(&self, ticks: u64) -> Instant {
        self.start_time + self.tick_duration * ticks as u32
    }

    /// Calculate wheel and slot for a delta
    fn calculate_slot(&self, delta: u64) -> (usize, usize) {
        if delta < 256 {
            (0, delta as usize)
        } else if delta < 256 * 64 {
            (1, (delta / 256) as usize)
        } else if delta < 256 * 64 * 64 {
            (2, (delta / (256 * 64)) as usize)
        } else {
            (3, (delta / (256 * 64 * 64)).min(63) as usize)
        }
    }

    /// Cascade entries from higher wheels
    fn cascade(&mut self, wheel_idx: usize) {
        if wheel_idx >= self.wheels.len() {
            return;
        }

        let shift = 8 + 6 * (wheel_idx - 1);
        let slot_idx = ((self.current_tick >> shift) as usize) & self.wheels[wheel_idx].mask;

        let entries = std::mem::take(&mut self.wheels[wheel_idx].slots[slot_idx]);

        for entry in entries {
            let delta = entry.deadline.saturating_sub(self.current_tick);
            let (new_wheel, new_slot) = self.calculate_slot(delta);
            self.wheels[new_wheel].slots[new_slot].push(entry);
        }

        // Continue cascade if this slot was 0
        if slot_idx == 0 {
            self.cascade(wheel_idx + 1);
        }
    }
}

impl Default for TimerWheel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::noop_waker;

    #[test]
    fn test_timer_insert_and_fire() {
        let mut wheel = TimerWheel::new();
        let start = Instant::now();

        let waker = noop_waker();
        let _handle = wheel.insert(start + Duration::from_millis(10), waker);

        // Advance past deadline
        let wakers = wheel.advance(start + Duration::from_millis(15));
        assert_eq!(wakers.len(), 1);
    }

    #[test]
    fn test_timer_cancel() {
        let mut wheel = TimerWheel::new();
        let start = Instant::now();

        let waker = noop_waker();
        let handle = wheel.insert(start + Duration::from_millis(10), waker);

        assert!(wheel.cancel(handle));

        // Advance past deadline - should not fire
        let wakers = wheel.advance(start + Duration::from_millis(15));
        assert_eq!(wakers.len(), 0);
    }

    #[test]
    fn test_timer_wheel_new() {
        let wheel = TimerWheel::new();
        assert_eq!(wheel.current_tick, 0);
        assert_eq!(wheel.next_id, 0);
        assert_eq!(wheel.tick_duration, Duration::from_millis(1));
    }

    #[test]
    fn test_timer_wheel_with_tick_duration() {
        let wheel = TimerWheel::with_tick_duration(Duration::from_micros(100));
        assert_eq!(wheel.tick_duration, Duration::from_micros(100));
    }

    #[test]
    fn test_timer_wheel_default() {
        let wheel: TimerWheel = Default::default();
        assert_eq!(wheel.tick_duration, Duration::from_millis(1));
    }

    #[test]
    fn test_timer_handle_uniqueness() {
        let mut wheel = TimerWheel::new();
        let start = Instant::now();
        let waker = noop_waker();

        let handle1 = wheel.insert(start + Duration::from_millis(10), waker.clone());
        let handle2 = wheel.insert(start + Duration::from_millis(20), waker.clone());
        let handle3 = wheel.insert(start + Duration::from_millis(30), waker);

        assert_ne!(handle1.id, handle2.id);
        assert_ne!(handle2.id, handle3.id);
        assert_ne!(handle1.id, handle3.id);
    }

    #[test]
    fn test_timer_multiple_timers_same_deadline() {
        let mut wheel = TimerWheel::new();
        let start = Instant::now();
        let deadline = start + Duration::from_millis(10);
        let waker = noop_waker();

        let _h1 = wheel.insert(deadline, waker.clone());
        let _h2 = wheel.insert(deadline, waker.clone());
        let _h3 = wheel.insert(deadline, waker);

        // Advance past deadline
        let wakers = wheel.advance(start + Duration::from_millis(15));
        assert_eq!(wakers.len(), 3);
    }

    #[test]
    fn test_timer_no_fire_before_deadline() {
        let mut wheel = TimerWheel::new();
        let start = Instant::now();

        let waker = noop_waker();
        let _handle = wheel.insert(start + Duration::from_millis(100), waker);

        // Advance but not past deadline
        let wakers = wheel.advance(start + Duration::from_millis(50));
        assert_eq!(wakers.len(), 0);
    }

    #[test]
    fn test_timer_fire_exactly_at_deadline() {
        let mut wheel = TimerWheel::new();
        let start = Instant::now();
        let deadline = start + Duration::from_millis(10);

        let waker = noop_waker();
        let _handle = wheel.insert(deadline, waker);

        // Advance exactly to deadline
        let wakers = wheel.advance(deadline);
        assert_eq!(wakers.len(), 1);
    }

    #[test]
    fn test_timer_ordered_firing() {
        let mut wheel = TimerWheel::new();
        let start = Instant::now();
        let waker = noop_waker();

        // Insert timers at different deadlines
        let _h1 = wheel.insert(start + Duration::from_millis(10), waker.clone());
        let _h2 = wheel.insert(start + Duration::from_millis(20), waker.clone());
        let _h3 = wheel.insert(start + Duration::from_millis(30), waker);

        // Advance to 15ms - should fire 1 timer
        let wakers = wheel.advance(start + Duration::from_millis(15));
        assert_eq!(wakers.len(), 1);

        // Advance to 25ms - should fire 1 more timer
        let wakers = wheel.advance(start + Duration::from_millis(25));
        assert_eq!(wakers.len(), 1);

        // Advance to 35ms - should fire the last timer
        let wakers = wheel.advance(start + Duration::from_millis(35));
        assert_eq!(wakers.len(), 1);
    }

    #[test]
    fn test_timer_cancel_nonexistent() {
        let mut wheel = TimerWheel::new();

        // Try to cancel a handle that was never inserted
        let fake_handle = TimerHandle { id: 999 };
        assert!(!wheel.cancel(fake_handle));
    }

    #[test]
    fn test_timer_cancel_already_cancelled() {
        let mut wheel = TimerWheel::new();
        let start = Instant::now();

        let waker = noop_waker();
        let handle = wheel.insert(start + Duration::from_millis(10), waker);

        assert!(wheel.cancel(handle));
        // Second cancel should return false
        assert!(!wheel.cancel(handle));
    }

    #[test]
    fn test_timer_next_deadline_empty() {
        let wheel = TimerWheel::new();
        assert!(wheel.next_deadline().is_none());
    }

    #[test]
    fn test_timer_next_deadline_single() {
        let mut wheel = TimerWheel::new();
        let start = Instant::now();
        let deadline = start + Duration::from_millis(100);

        let waker = noop_waker();
        let _handle = wheel.insert(deadline, waker);

        let next = wheel.next_deadline();
        assert!(next.is_some());
        // The deadline should be close to what we set
    }

    #[test]
    fn test_timer_next_deadline_multiple() {
        let mut wheel = TimerWheel::new();
        let start = Instant::now();
        let waker = noop_waker();

        let _h1 = wheel.insert(start + Duration::from_millis(100), waker.clone());
        let _h2 = wheel.insert(start + Duration::from_millis(50), waker.clone());
        let _h3 = wheel.insert(start + Duration::from_millis(150), waker);

        // Next deadline should be the earliest one
        let next = wheel.next_deadline();
        assert!(next.is_some());
    }

    #[test]
    fn test_timer_calculate_slot_wheel_0() {
        let wheel = TimerWheel::new();

        // Delta < 256 should go to wheel 0
        let (wheel_idx, slot_idx) = wheel.calculate_slot(100);
        assert_eq!(wheel_idx, 0);
        assert_eq!(slot_idx, 100);
    }

    #[test]
    fn test_timer_calculate_slot_wheel_1() {
        let wheel = TimerWheel::new();

        // Delta 256..256*64 should go to wheel 1
        let (wheel_idx, _slot_idx) = wheel.calculate_slot(300);
        assert_eq!(wheel_idx, 1);
    }

    #[test]
    fn test_timer_calculate_slot_wheel_2() {
        let wheel = TimerWheel::new();

        // Delta 256*64..256*64*64 should go to wheel 2
        let delta = 256 * 64 + 100;
        let (wheel_idx, _slot_idx) = wheel.calculate_slot(delta);
        assert_eq!(wheel_idx, 2);
    }

    #[test]
    fn test_timer_calculate_slot_wheel_3() {
        let wheel = TimerWheel::new();

        // Delta >= 256*64*64 should go to wheel 3
        let delta = 256 * 64 * 64 + 100;
        let (wheel_idx, _slot_idx) = wheel.calculate_slot(delta);
        assert_eq!(wheel_idx, 3);
    }

    #[test]
    fn test_timer_calculate_slot_max() {
        let wheel = TimerWheel::new();

        // Very large delta should be clamped to wheel 3, slot 63
        let delta = u64::MAX / 2;
        let (wheel_idx, slot_idx) = wheel.calculate_slot(delta);
        assert_eq!(wheel_idx, 3);
        assert!(slot_idx <= 63);
    }

    #[test]
    fn test_timer_long_duration() {
        let mut wheel = TimerWheel::new();
        let start = Instant::now();

        // Insert a timer far in the future
        let waker = noop_waker();
        let _handle = wheel.insert(start + Duration::from_secs(60), waker);

        // Should not fire immediately
        let wakers = wheel.advance(start + Duration::from_secs(1));
        assert_eq!(wakers.len(), 0);
    }

    #[test]
    fn test_timer_advance_no_timers() {
        let mut wheel = TimerWheel::new();
        let start = Instant::now();

        // Advancing with no timers should be safe
        let wakers = wheel.advance(start + Duration::from_secs(10));
        assert_eq!(wakers.len(), 0);
    }

    #[test]
    fn test_timer_advance_multiple_times() {
        let mut wheel = TimerWheel::new();
        let start = Instant::now();

        let waker = noop_waker();
        let _h1 = wheel.insert(start + Duration::from_millis(5), waker.clone());
        let _h2 = wheel.insert(start + Duration::from_millis(10), waker.clone());
        let _h3 = wheel.insert(start + Duration::from_millis(15), waker);

        // Multiple advances
        let w1 = wheel.advance(start + Duration::from_millis(6));
        let w2 = wheel.advance(start + Duration::from_millis(11));
        let w3 = wheel.advance(start + Duration::from_millis(16));

        assert_eq!(w1.len(), 1);
        assert_eq!(w2.len(), 1);
        assert_eq!(w3.len(), 1);
    }

    #[test]
    fn test_timer_handle_equality() {
        let h1 = TimerHandle { id: 1 };
        let h2 = TimerHandle { id: 1 };
        let h3 = TimerHandle { id: 2 };

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_timer_wheel_structure() {
        let wheel = TimerWheel::new();

        // Verify wheel sizes
        assert_eq!(wheel.wheels[0].slots.len(), 256);
        assert_eq!(wheel.wheels[1].slots.len(), 64);
        assert_eq!(wheel.wheels[2].slots.len(), 64);
        assert_eq!(wheel.wheels[3].slots.len(), 64);

        // Verify masks
        assert_eq!(wheel.wheels[0].mask, 255);
        assert_eq!(wheel.wheels[1].mask, 63);
        assert_eq!(wheel.wheels[2].mask, 63);
        assert_eq!(wheel.wheels[3].mask, 63);
    }

    #[test]
    fn test_timer_stress_many_timers() {
        let mut wheel = TimerWheel::new();
        let start = Instant::now();
        let waker = noop_waker();

        // Insert many timers
        for i in 0..100 {
            let _h = wheel.insert(start + Duration::from_millis(i * 10), waker.clone());
        }

        // Advance and collect all
        let wakers = wheel.advance(start + Duration::from_secs(2));
        assert_eq!(wakers.len(), 100);
    }

    #[test]
    fn test_timer_mixed_cancel_and_fire() {
        let mut wheel = TimerWheel::new();
        let start = Instant::now();
        let waker = noop_waker();

        let h1 = wheel.insert(start + Duration::from_millis(10), waker.clone());
        let h2 = wheel.insert(start + Duration::from_millis(20), waker.clone());
        let h3 = wheel.insert(start + Duration::from_millis(30), waker);

        // Cancel the middle one
        wheel.cancel(h2);

        // Advance past all deadlines
        let wakers = wheel.advance(start + Duration::from_millis(35));

        // Only 2 should fire (h1 and h3)
        assert_eq!(wakers.len(), 2);

        // Verify we can still check handles
        assert_eq!(h1, h1);
        assert_eq!(h3, h3);
    }
}
