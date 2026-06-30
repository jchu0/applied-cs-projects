//! Time utilities
//!
//! Provides async sleep, timeout, and interval functions.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::timer::{TimerHandle, TimerWheel};

/// Global timer wheel (thread-local for now)
thread_local! {
    static TIMER_WHEEL: Arc<Mutex<TimerWheel>> = Arc::new(Mutex::new(TimerWheel::new()));
}

/// Get the timer wheel
pub fn timer_wheel() -> Arc<Mutex<TimerWheel>> {
    TIMER_WHEEL.with(|tw| tw.clone())
}

/// Process expired timers
pub fn process_timers() {
    TIMER_WHEEL.with(|tw| {
        let wakers = tw.lock().advance(Instant::now());
        for waker in wakers {
            waker.wake();
        }
    });
}

/// Sleep for a duration
pub fn sleep(duration: Duration) -> Sleep {
    Sleep {
        deadline: Instant::now() + duration,
        handle: None,
    }
}

/// Sleep until an instant
pub fn sleep_until(deadline: Instant) -> Sleep {
    Sleep {
        deadline,
        handle: None,
    }
}

/// Future for sleeping
pub struct Sleep {
    deadline: Instant,
    handle: Option<TimerHandle>,
}

impl Future for Sleep {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Check if deadline has passed
        if Instant::now() >= self.deadline {
            return Poll::Ready(());
        }

        // Register with timer wheel if not already
        if self.handle.is_none() {
            TIMER_WHEEL.with(|tw| {
                let handle = tw.lock().insert(self.deadline, cx.waker().clone());
                self.handle = Some(handle);
            });
        } else {
            // Update waker in case it changed
            TIMER_WHEEL.with(|tw| {
                // Cancel old and re-register with new waker
                if let Some(handle) = self.handle.take() {
                    tw.lock().cancel(handle);
                }
                let handle = tw.lock().insert(self.deadline, cx.waker().clone());
                self.handle = Some(handle);
            });
        }

        Poll::Pending
    }
}

impl Drop for Sleep {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            TIMER_WHEEL.with(|tw| {
                tw.lock().cancel(handle);
            });
        }
    }
}

/// Timeout future combinator
pub fn timeout<F>(duration: Duration, future: F) -> Timeout<F> {
    Timeout {
        future,
        sleep: sleep(duration),
    }
}

/// Timeout until a specific instant
pub fn timeout_at<F>(deadline: Instant, future: F) -> Timeout<F> {
    Timeout {
        future,
        sleep: sleep_until(deadline),
    }
}

/// Future that times out
pub struct Timeout<F> {
    future: F,
    sleep: Sleep,
}

/// Error returned when timeout expires
#[derive(Debug, Clone, Copy)]
pub struct Elapsed;

impl std::fmt::Display for Elapsed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "timeout elapsed")
    }
}

impl std::error::Error for Elapsed {}

impl<F: Future> Future for Timeout<F> {
    type Output = Result<F::Output, Elapsed>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Safety: We don't move the future
        let this = unsafe { self.get_unchecked_mut() };

        // Poll the inner future first
        let future = unsafe { Pin::new_unchecked(&mut this.future) };
        if let Poll::Ready(result) = future.poll(cx) {
            return Poll::Ready(Ok(result));
        }

        // Poll the sleep timer
        let sleep = unsafe { Pin::new_unchecked(&mut this.sleep) };
        if let Poll::Ready(()) = sleep.poll(cx) {
            return Poll::Ready(Err(Elapsed));
        }

        Poll::Pending
    }
}

/// Interval timer
pub struct Interval {
    period: Duration,
    next: Instant,
    handle: Option<TimerHandle>,
}

impl Interval {
    /// Create a new interval
    pub fn new(period: Duration) -> Self {
        Self {
            period,
            next: Instant::now() + period,
            handle: None,
        }
    }

    /// Wait for the next tick
    pub async fn tick(&mut self) {
        // Wait until next deadline
        let sleep = sleep_until(self.next);
        sleep.await;

        // Update next deadline
        self.next += self.period;
    }
}

/// Create an interval timer
pub fn interval(period: Duration) -> Interval {
    Interval::new(period)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sleep_immediate() {
        // Sleep for 0 duration should complete immediately
        let sleep = sleep(Duration::from_secs(0));
        let waker = crate::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut pinned = std::pin::pin!(sleep);

        // First poll might register, second should complete
        let _ = pinned.as_mut().poll(&mut cx);
        // Give it time to complete
        std::thread::sleep(Duration::from_millis(1));
        assert!(pinned.as_mut().poll(&mut cx).is_ready());
    }

    #[test]
    fn test_timeout_structure() {
        // Just test that timeout creates properly
        let fut = async { 42 };
        let _timeout = timeout(Duration::from_millis(100), fut);
    }

    #[test]
    fn test_interval_creation() {
        let _interval = interval(Duration::from_millis(100));
    }
}
