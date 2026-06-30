//! Channel demonstration
//!
//! Shows how to use oneshot and mpsc channels.

use async_runtime::sync::{mpsc, oneshot};
use async_runtime::{block_on, spawn};

fn main() {
    block_on(async {
        println!("Testing channel functionality...\n");

        // Test oneshot channel
        println!("Test 1: Oneshot channel");
        let (tx, rx) = oneshot::channel();

        spawn(async move {
            tx.send(42).unwrap();
        });

        let value = rx.await.unwrap();
        println!("  Received: {}\n", value);

        // Test mpsc channel
        println!("Test 2: MPSC channel");
        let (tx, mut rx) = mpsc::channel(16);

        // Spawn multiple producers
        for i in 0..3 {
            let tx = tx.clone();
            spawn(async move {
                tx.try_send(i * 10).unwrap();
            });
        }

        // Drop the original sender
        drop(tx);

        // Receive all messages
        let mut values = Vec::new();
        while let Ok(value) = rx.try_recv() {
            values.push(value);
        }
        values.sort();
        println!("  Received: {:?}\n", values);

        println!("Channel demo complete!");
    });
}
