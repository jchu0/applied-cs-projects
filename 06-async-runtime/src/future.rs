//! Future combinators
//!
//! Provides utilities for combining and manipulating futures.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Select between two futures, returning the first to complete
pub fn select<A, B>(a: A, b: B) -> Select<A, B>
where
    A: Future,
    B: Future,
{
    Select { a, b }
}

/// Future that races two futures
pub struct Select<A, B> {
    a: A,
    b: B,
}

/// Result of select operation
pub enum Either<A, B> {
    Left(A),
    Right(B),
}

impl<A: Future, B: Future> Future for Select<A, B> {
    type Output = Either<A::Output, B::Output>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Safety: We don't move the futures
        let this = unsafe { self.get_unchecked_mut() };

        // Poll first future
        let a = unsafe { Pin::new_unchecked(&mut this.a) };
        if let Poll::Ready(result) = a.poll(cx) {
            return Poll::Ready(Either::Left(result));
        }

        // Poll second future
        let b = unsafe { Pin::new_unchecked(&mut this.b) };
        if let Poll::Ready(result) = b.poll(cx) {
            return Poll::Ready(Either::Right(result));
        }

        Poll::Pending
    }
}

/// Join two futures, waiting for both to complete
pub fn join<A, B>(a: A, b: B) -> Join<A, B>
where
    A: Future,
    B: Future,
{
    Join {
        a: MaybeDone::Future(a),
        b: MaybeDone::Future(b),
    }
}

/// Future that waits for two futures to complete
pub struct Join<A, B>
where
    A: Future,
    B: Future,
{
    a: MaybeDone<A>,
    b: MaybeDone<B>,
}

enum MaybeDone<F: Future> {
    Future(F),
    Done(F::Output),
    Gone,
}

impl<F: Future> MaybeDone<F> {
    fn poll(&mut self, cx: &mut Context<'_>) -> bool {
        match self {
            MaybeDone::Future(f) => {
                // Safety: We don't move the future
                let f = unsafe { Pin::new_unchecked(f) };
                if let Poll::Ready(result) = f.poll(cx) {
                    *self = MaybeDone::Done(result);
                    true
                } else {
                    false
                }
            }
            MaybeDone::Done(_) => true,
            MaybeDone::Gone => true,
        }
    }

    fn take(&mut self) -> F::Output {
        match std::mem::replace(self, MaybeDone::Gone) {
            MaybeDone::Done(v) => v,
            _ => panic!("MaybeDone polled after completion"),
        }
    }
}

impl<A: Future, B: Future> Future for Join<A, B> {
    type Output = (A::Output, B::Output);

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Safety: We don't move the futures
        let this = unsafe { self.get_unchecked_mut() };

        let a_done = this.a.poll(cx);
        let b_done = this.b.poll(cx);

        if a_done && b_done {
            Poll::Ready((this.a.take(), this.b.take()))
        } else {
            Poll::Pending
        }
    }
}

/// Join three futures
pub fn join3<A, B, C>(a: A, b: B, c: C) -> Join3<A, B, C>
where
    A: Future,
    B: Future,
    C: Future,
{
    Join3 {
        a: MaybeDone::Future(a),
        b: MaybeDone::Future(b),
        c: MaybeDone::Future(c),
    }
}

/// Future that waits for three futures
pub struct Join3<A, B, C>
where
    A: Future,
    B: Future,
    C: Future,
{
    a: MaybeDone<A>,
    b: MaybeDone<B>,
    c: MaybeDone<C>,
}

impl<A: Future, B: Future, C: Future> Future for Join3<A, B, C> {
    type Output = (A::Output, B::Output, C::Output);

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };

        let a_done = this.a.poll(cx);
        let b_done = this.b.poll(cx);
        let c_done = this.c.poll(cx);

        if a_done && b_done && c_done {
            Poll::Ready((this.a.take(), this.b.take(), this.c.take()))
        } else {
            Poll::Pending
        }
    }
}

/// Yield execution to allow other tasks to run
pub async fn yield_now() {
    struct YieldNow {
        yielded: bool,
    }

    impl Future for YieldNow {
        type Output = ();

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            if self.yielded {
                Poll::Ready(())
            } else {
                self.yielded = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }

    YieldNow { yielded: false }.await
}

/// Poll a future once, returning None if not ready
///
/// Note: This consumes the future and returns the result or None.
/// The future cannot be polled again if it returns Pending.
pub async fn poll_once<F: Future>(future: F) -> Option<F::Output> {
    let mut future = std::pin::pin!(future);

    struct PollOnce<'a, F> {
        future: &'a mut std::pin::Pin<&'a mut F>,
    }

    impl<'a, F: Future> Future for PollOnce<'a, F> {
        type Output = Option<F::Output>;

        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            let this = self.get_mut();
            match this.future.as_mut().poll(cx) {
                Poll::Ready(result) => Poll::Ready(Some(result)),
                Poll::Pending => Poll::Ready(None),
            }
        }
    }

    PollOnce { future: &mut future }.await
}

/// Future that returns a value immediately
pub fn ready<T>(value: T) -> Ready<T> {
    Ready { value: Some(value) }
}

/// Immediately ready future
pub struct Ready<T> {
    value: Option<T>,
}

// Ready is safe to unpin
impl<T> Unpin for Ready<T> {}

impl<T> Future for Ready<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(self.get_mut().value.take().expect("Ready polled after completion"))
    }
}

/// Future that never completes
pub fn pending<T>() -> Pending<T> {
    Pending {
        _marker: std::marker::PhantomData,
    }
}

/// Never-completing future
pub struct Pending<T> {
    _marker: std::marker::PhantomData<T>,
}

impl<T> Future for Pending<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ready() {
        let fut = ready(42);
        let waker = crate::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut pinned = std::pin::pin!(fut);

        match pinned.as_mut().poll(&mut cx) {
            Poll::Ready(v) => assert_eq!(v, 42),
            Poll::Pending => panic!("ready should be immediately ready"),
        }
    }

    #[test]
    fn test_select_left() {
        let a = ready(1);
        let b = pending::<i32>();
        let fut = select(a, b);

        let waker = crate::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut pinned = std::pin::pin!(fut);

        match pinned.as_mut().poll(&mut cx) {
            Poll::Ready(Either::Left(v)) => assert_eq!(v, 1),
            _ => panic!("expected left"),
        }
    }

    #[test]
    fn test_join_both() {
        let a = ready(1);
        let b = ready(2);
        let fut = join(a, b);

        let waker = crate::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut pinned = std::pin::pin!(fut);

        match pinned.as_mut().poll(&mut cx) {
            Poll::Ready((a, b)) => {
                assert_eq!(a, 1);
                assert_eq!(b, 2);
            }
            Poll::Pending => panic!("join should be ready"),
        }
    }
}
