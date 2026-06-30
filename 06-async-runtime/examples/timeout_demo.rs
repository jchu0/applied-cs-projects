//! Timeout demonstration
//!
//! Shows how to use timeout combinators.

use async_runtime::{block_on, sleep, timeout};
use std::time::Duration;

fn main() {
    block_on(async {
        println!("Testing timeout functionality...\n");

        // Test 1: Operation completes before timeout
        println!("Test 1: Fast operation with timeout");
        let result = timeout(Duration::from_millis(100), async {
            sleep(Duration::from_millis(10)).await;
            "completed"
        })
        .await;

        match result {
            Ok(msg) => println!("  Result: {} (as expected)\n", msg),
            Err(_) => println!("  Unexpected timeout!\n"),
        }

        // Test 2: Operation times out
        println!("Test 2: Slow operation with timeout");
        let result = timeout(Duration::from_millis(10), async {
            sleep(Duration::from_millis(100)).await;
            "completed"
        })
        .await;

        match result {
            Ok(_) => println!("  Unexpected success!\n"),
            Err(e) => println!("  Result: {} (as expected)\n", e),
        }

        println!("Timeout demo complete!");
    });
}
