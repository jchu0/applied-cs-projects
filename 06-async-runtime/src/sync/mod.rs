//! Synchronization primitives
//!
//! Provides async-aware synchronization primitives.

pub mod oneshot;
pub mod mpsc;
pub mod cancellation;
pub mod mutex;
pub mod notify;

pub use oneshot::{channel as oneshot_channel, Sender as OneshotSender, Receiver as OneshotReceiver};
pub use mpsc::{channel as mpsc_channel, Sender as MpscSender, Receiver as MpscReceiver};
pub use cancellation::{CancellationToken, Cancelled, DropGuard};
pub use mutex::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
pub use notify::{Notify, Notified, Semaphore, SemaphorePermit, Barrier, BarrierWaitResult};
