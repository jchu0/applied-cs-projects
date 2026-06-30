//! Tests for CRDT implementations.

use crdt_collaboration::crdt::*;
use crdt_collaboration::ClientId;

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create deterministic client IDs for testing
    fn client_id(n: u8) -> ClientId {
        ClientId::from_bytes([n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    }

    // ===================
    // LWWRegister Tests
    // ===================

    #[test]
    fn test_lww_register_basic_operations() {
        let client = client_id(1);
        let mut register = LWWRegister::new("initial".to_string(), 0, client);

        // Test initial state
        assert_eq!(register.get(), "initial");

        // Test set operation
        register.set("hello".to_string(), 1, client);
        assert_eq!(register.get(), "hello");

        // Test overwrite with higher timestamp
        register.set("world".to_string(), 2, client);
        assert_eq!(register.get(), "world");
    }

    #[test]
    fn test_lww_register_merge_conflict_resolution() {
        let client1 = client_id(1);
        let client2 = client_id(2);

        let mut reg1 = LWWRegister::new("value1".to_string(), 1, client1);
        let reg2 = LWWRegister::new("value2".to_string(), 2, client2);

        // Merge - reg2 has higher timestamp, should win
        reg1.merge(&reg2);
        assert_eq!(reg1.get(), "value2");
    }

    #[test]
    fn test_lww_register_same_timestamp_tiebreaker() {
        let client1 = client_id(1);
        let client2 = client_id(2);

        // Same timestamp, higher client_id wins
        let mut reg1 = LWWRegister::new("value1".to_string(), 1, client1);
        let reg2 = LWWRegister::new("value2".to_string(), 1, client2);

        reg1.merge(&reg2);
        // client2 > client1, so value2 wins
        assert_eq!(reg1.get(), "value2");
    }

    // ===================
    // GCounter Tests
    // ===================

    #[test]
    fn test_gcounter_basic_increment() {
        let client = client_id(1);
        let mut counter = GCounter::new();

        assert_eq!(counter.value(), 0);

        counter.increment(client, 1);
        assert_eq!(counter.value(), 1);

        counter.increment(client, 5);
        assert_eq!(counter.value(), 6);
    }

    #[test]
    fn test_gcounter_multiple_clients() {
        let client1 = client_id(1);
        let client2 = client_id(2);
        let mut counter = GCounter::new();

        counter.increment(client1, 3);
        counter.increment(client2, 5);

        assert_eq!(counter.value(), 8);
    }

    #[test]
    fn test_gcounter_merge() {
        let client1 = client_id(1);
        let client2 = client_id(2);

        let mut counter1 = GCounter::new();
        let mut counter2 = GCounter::new();

        counter1.increment(client1, 3);
        counter2.increment(client2, 5);

        counter1.merge(&counter2);
        assert_eq!(counter1.value(), 8);

        // Merge is idempotent
        counter1.merge(&counter2);
        assert_eq!(counter1.value(), 8);
    }

    #[test]
    fn test_gcounter_merge_takes_max() {
        let client1 = client_id(1);

        let mut counter1 = GCounter::new();
        let mut counter2 = GCounter::new();

        counter1.increment(client1, 3);
        counter2.increment(client1, 5);

        counter1.merge(&counter2);
        // Should take max(3, 5) = 5
        assert_eq!(counter1.value(), 5);
    }

    // ===================
    // PNCounter Tests
    // ===================

    #[test]
    fn test_pncounter_basic_operations() {
        let client = client_id(1);
        let mut counter = PNCounter::new();

        assert_eq!(counter.value(), 0);

        counter.increment(client, 5);
        assert_eq!(counter.value(), 5);

        counter.decrement(client, 2);
        assert_eq!(counter.value(), 3);
    }

    #[test]
    fn test_pncounter_negative_value() {
        let client = client_id(1);
        let mut counter = PNCounter::new();

        counter.decrement(client, 5);
        assert_eq!(counter.value(), -5);
    }

    #[test]
    fn test_pncounter_merge() {
        let client1 = client_id(1);
        let client2 = client_id(2);

        let mut counter1 = PNCounter::new();
        let mut counter2 = PNCounter::new();

        counter1.increment(client1, 10);
        counter1.decrement(client1, 3);

        counter2.increment(client2, 5);
        counter2.decrement(client2, 2);

        counter1.merge(&counter2);
        // (10 - 3) + (5 - 2) = 7 + 3 = 10
        assert_eq!(counter1.value(), 10);
    }

    // ===================
    // VectorClock Tests
    // ===================

    #[test]
    fn test_vector_clock_basic_increment() {
        let client1 = client_id(1);
        let client2 = client_id(2);
        let mut clock = VectorClock::new();

        let t1 = clock.increment(client1);
        assert_eq!(t1, 1);
        assert_eq!(clock.get(&client1), 1);

        let t2 = clock.increment(client1);
        assert_eq!(t2, 2);
        assert_eq!(clock.get(&client1), 2);

        let t3 = clock.increment(client2);
        assert_eq!(t3, 1);
        assert_eq!(clock.get(&client2), 1);
    }

    #[test]
    fn test_vector_clock_merge() {
        let client1 = client_id(1);
        let client2 = client_id(2);
        let client3 = client_id(3);

        let mut clock1 = VectorClock::new();
        let mut clock2 = VectorClock::new();

        clock1.increment(client1);
        clock1.increment(client1);
        clock1.increment(client2);

        clock2.increment(client2);
        clock2.increment(client2);
        clock2.increment(client3);

        clock1.merge(&clock2);
        assert_eq!(clock1.get(&client1), 2);
        assert_eq!(clock1.get(&client2), 2); // max(1, 2) = 2
        assert_eq!(clock1.get(&client3), 1);
    }

    #[test]
    fn test_vector_clock_happens_before() {
        let client1 = client_id(1);
        let client2 = client_id(2);

        let mut clock1 = VectorClock::new();
        let mut clock2 = VectorClock::new();

        clock1.increment(client1);

        clock2.increment(client1);
        clock2.increment(client2);

        // clock1 happens before clock2
        assert!(clock1.happens_before(&clock2));
        assert!(!clock2.happens_before(&clock1));
    }

    #[test]
    fn test_vector_clock_concurrent() {
        let client1 = client_id(1);
        let client2 = client_id(2);

        let mut clock1 = VectorClock::new();
        let mut clock2 = VectorClock::new();

        clock1.increment(client1);
        clock2.increment(client2);

        // Neither happens before the other - concurrent
        assert!(!clock1.happens_before(&clock2));
        assert!(!clock2.happens_before(&clock1));
    }

    #[test]
    fn test_vector_clock_is_concurrent() {
        let client1 = client_id(1);
        let client2 = client_id(2);

        let mut clock1 = VectorClock::new();
        let mut clock2 = VectorClock::new();

        clock1.increment(client1);
        clock2.increment(client2);

        assert!(clock1.is_concurrent(&clock2));
    }

    // ===================
    // PositionId Tests
    // ===================

    #[test]
    fn test_position_id_ordering() {
        let client1 = client_id(1);
        let client2 = client_id(2);

        let pos1 = PositionId::new(1, client1, 0);
        let pos2 = PositionId::new(2, client2, 0);

        assert!(pos1 < pos2);
    }

    #[test]
    fn test_position_id_same_lamport_different_client() {
        let client1 = client_id(1);
        let client2 = client_id(2);

        let pos1 = PositionId::new(1, client1, 0);
        let pos2 = PositionId::new(1, client2, 0);

        // Same lamport, different client - should still have total ordering
        assert!(pos1 != pos2);
        assert!(pos1 < pos2 || pos2 < pos1);
    }
}
