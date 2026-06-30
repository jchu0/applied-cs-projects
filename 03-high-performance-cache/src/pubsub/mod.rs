//! Pub/Sub messaging system
//!
//! Implements Redis-compatible publish/subscribe functionality
//! with support for channel subscriptions and pattern matching.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use crate::resp::RespValue;

/// Client ID type
pub type ClientId = usize;

/// Pub/Sub manager
pub struct PubSub {
    /// Channel subscriptions: channel -> set of client IDs
    channels: RwLock<HashMap<String, HashSet<ClientId>>>,
    /// Pattern subscriptions: pattern -> set of client IDs
    patterns: RwLock<HashMap<String, HashSet<ClientId>>>,
    /// Client subscriptions: client ID -> set of channels
    client_channels: RwLock<HashMap<ClientId, HashSet<String>>>,
    /// Client patterns: client ID -> set of patterns
    client_patterns: RwLock<HashMap<ClientId, HashSet<String>>>,
    /// Message buffer for clients
    messages: RwLock<HashMap<ClientId, Vec<RespValue>>>,
}

impl PubSub {
    /// Create a new Pub/Sub manager
    pub fn new() -> Self {
        Self {
            channels: RwLock::new(HashMap::new()),
            patterns: RwLock::new(HashMap::new()),
            client_channels: RwLock::new(HashMap::new()),
            client_patterns: RwLock::new(HashMap::new()),
            messages: RwLock::new(HashMap::new()),
        }
    }

    /// Subscribe a client to channels
    pub fn subscribe(&self, client_id: ClientId, channels: &[String]) -> Vec<RespValue> {
        let mut responses = Vec::new();
        let mut channel_map = self.channels.write().unwrap();
        let mut client_map = self.client_channels.write().unwrap();

        for channel in channels {
            // Add client to channel subscribers
            channel_map
                .entry(channel.clone())
                .or_insert_with(HashSet::new)
                .insert(client_id);

            // Track client's subscriptions
            client_map
                .entry(client_id)
                .or_insert_with(HashSet::new)
                .insert(channel.clone());

            // Count total subscriptions for this client
            let count = client_map.get(&client_id).map(|s| s.len()).unwrap_or(0);

            // Build response
            responses.push(RespValue::array(vec![
                RespValue::bulk_string("subscribe"),
                RespValue::bulk_string(channel),
                RespValue::integer(count as i64),
            ]));
        }

        responses
    }

    /// Unsubscribe a client from channels
    pub fn unsubscribe(&self, client_id: ClientId, channels: Option<&[String]>) -> Vec<RespValue> {
        let mut responses = Vec::new();
        let mut channel_map = self.channels.write().unwrap();
        let mut client_map = self.client_channels.write().unwrap();

        let channels_to_remove: Vec<String> = match channels {
            Some(chs) => chs.to_vec(),
            None => {
                // Unsubscribe from all channels
                client_map
                    .get(&client_id)
                    .map(|s| s.iter().cloned().collect())
                    .unwrap_or_default()
            }
        };

        for channel in channels_to_remove {
            // Remove client from channel
            if let Some(clients) = channel_map.get_mut(&channel) {
                clients.remove(&client_id);
                if clients.is_empty() {
                    channel_map.remove(&channel);
                }
            }

            // Update client's subscriptions
            if let Some(subs) = client_map.get_mut(&client_id) {
                subs.remove(&channel);
            }

            let count = client_map.get(&client_id).map(|s| s.len()).unwrap_or(0);

            responses.push(RespValue::array(vec![
                RespValue::bulk_string("unsubscribe"),
                RespValue::bulk_string(&channel),
                RespValue::integer(count as i64),
            ]));
        }

        responses
    }

    /// Subscribe to patterns
    pub fn psubscribe(&self, client_id: ClientId, patterns: &[String]) -> Vec<RespValue> {
        let mut responses = Vec::new();
        let mut pattern_map = self.patterns.write().unwrap();
        let mut client_map = self.client_patterns.write().unwrap();

        for pattern in patterns {
            pattern_map
                .entry(pattern.clone())
                .or_insert_with(HashSet::new)
                .insert(client_id);

            client_map
                .entry(client_id)
                .or_insert_with(HashSet::new)
                .insert(pattern.clone());

            let count = client_map.get(&client_id).map(|s| s.len()).unwrap_or(0);

            responses.push(RespValue::array(vec![
                RespValue::bulk_string("psubscribe"),
                RespValue::bulk_string(pattern),
                RespValue::integer(count as i64),
            ]));
        }

        responses
    }

    /// Unsubscribe from patterns
    pub fn punsubscribe(&self, client_id: ClientId, patterns: Option<&[String]>) -> Vec<RespValue> {
        let mut responses = Vec::new();
        let mut pattern_map = self.patterns.write().unwrap();
        let mut client_map = self.client_patterns.write().unwrap();

        let patterns_to_remove: Vec<String> = match patterns {
            Some(pats) => pats.to_vec(),
            None => {
                client_map
                    .get(&client_id)
                    .map(|s| s.iter().cloned().collect())
                    .unwrap_or_default()
            }
        };

        for pattern in patterns_to_remove {
            if let Some(clients) = pattern_map.get_mut(&pattern) {
                clients.remove(&client_id);
                if clients.is_empty() {
                    pattern_map.remove(&pattern);
                }
            }

            if let Some(pats) = client_map.get_mut(&client_id) {
                pats.remove(&pattern);
            }

            let count = client_map.get(&client_id).map(|s| s.len()).unwrap_or(0);

            responses.push(RespValue::array(vec![
                RespValue::bulk_string("punsubscribe"),
                RespValue::bulk_string(&pattern),
                RespValue::integer(count as i64),
            ]));
        }

        responses
    }

    /// Publish a message to a channel
    /// Returns the number of clients that received the message
    pub fn publish(&self, channel: &str, message: &[u8]) -> usize {
        let mut receivers = 0;

        // Send to direct subscribers
        let channel_map = self.channels.read().unwrap();
        if let Some(clients) = channel_map.get(channel) {
            let mut messages = self.messages.write().unwrap();
            for client_id in clients {
                let msg = RespValue::array(vec![
                    RespValue::bulk_string("message"),
                    RespValue::bulk_string(channel),
                    RespValue::bulk(message.to_vec()),
                ]);
                messages.entry(*client_id).or_insert_with(Vec::new).push(msg);
                receivers += 1;
            }
        }
        drop(channel_map);

        // Send to pattern subscribers
        let pattern_map = self.patterns.read().unwrap();
        for (pattern, clients) in pattern_map.iter() {
            if Self::match_pattern(pattern, channel) {
                let mut messages = self.messages.write().unwrap();
                for client_id in clients {
                    let msg = RespValue::array(vec![
                        RespValue::bulk_string("pmessage"),
                        RespValue::bulk_string(pattern),
                        RespValue::bulk_string(channel),
                        RespValue::bulk(message.to_vec()),
                    ]);
                    messages.entry(*client_id).or_insert_with(Vec::new).push(msg);
                    receivers += 1;
                }
            }
        }

        receivers
    }

    /// Get pending messages for a client
    pub fn get_messages(&self, client_id: ClientId) -> Vec<RespValue> {
        let mut messages = self.messages.write().unwrap();
        messages.remove(&client_id).unwrap_or_default()
    }

    /// Check if client has subscriptions
    pub fn has_subscriptions(&self, client_id: ClientId) -> bool {
        let channels = self.client_channels.read().unwrap();
        let patterns = self.client_patterns.read().unwrap();

        channels.get(&client_id).map(|s| !s.is_empty()).unwrap_or(false)
            || patterns.get(&client_id).map(|s| !s.is_empty()).unwrap_or(false)
    }

    /// Get number of subscriptions for a client
    pub fn subscription_count(&self, client_id: ClientId) -> usize {
        let channels = self.client_channels.read().unwrap();
        let patterns = self.client_patterns.read().unwrap();

        let ch_count = channels.get(&client_id).map(|s| s.len()).unwrap_or(0);
        let pat_count = patterns.get(&client_id).map(|s| s.len()).unwrap_or(0);

        ch_count + pat_count
    }

    /// Clean up client subscriptions when disconnected
    pub fn client_disconnected(&self, client_id: ClientId) {
        self.unsubscribe(client_id, None);
        self.punsubscribe(client_id, None);

        let mut messages = self.messages.write().unwrap();
        messages.remove(&client_id);
    }

    /// Get number of subscribers for a channel
    pub fn numsub(&self, channel: &str) -> usize {
        let channels = self.channels.read().unwrap();
        channels.get(channel).map(|s| s.len()).unwrap_or(0)
    }

    /// Get number of pattern subscriptions
    pub fn numpat(&self) -> usize {
        let patterns = self.patterns.read().unwrap();
        patterns.values().map(|s| s.len()).sum()
    }

    /// List active channels
    pub fn channels_list(&self, pattern: Option<&str>) -> Vec<String> {
        let channels = self.channels.read().unwrap();

        match pattern {
            Some(pat) => channels
                .keys()
                .filter(|ch| Self::match_pattern(pat, ch))
                .cloned()
                .collect(),
            None => channels.keys().cloned().collect(),
        }
    }

    /// Match a pattern against a string (glob-style)
    fn match_pattern(pattern: &str, s: &str) -> bool {
        let mut p_chars = pattern.chars().peekable();
        let mut s_chars = s.chars().peekable();

        while let Some(p) = p_chars.next() {
            match p {
                '*' => {
                    // Match any sequence
                    if p_chars.peek().is_none() {
                        return true;
                    }
                    // Try matching rest of pattern at each position
                    let rest_pattern: String = p_chars.collect();
                    let mut rest_s: String = s_chars.collect();
                    while !rest_s.is_empty() {
                        if Self::match_pattern(&rest_pattern, &rest_s) {
                            return true;
                        }
                        rest_s = rest_s[1..].to_string();
                    }
                    return Self::match_pattern(&rest_pattern, "");
                }
                '?' => {
                    // Match any single character
                    if s_chars.next().is_none() {
                        return false;
                    }
                }
                '[' => {
                    // Character class (simplified)
                    let c = match s_chars.next() {
                        Some(c) => c,
                        None => return false,
                    };
                    let mut matched = false;
                    let mut negate = false;

                    if p_chars.peek() == Some(&'^') {
                        negate = true;
                        p_chars.next();
                    }

                    while let Some(pc) = p_chars.next() {
                        if pc == ']' {
                            break;
                        }
                        if pc == c {
                            matched = true;
                        }
                    }

                    if negate {
                        matched = !matched;
                    }
                    if !matched {
                        return false;
                    }
                }
                '\\' => {
                    // Escape next character
                    let escaped = p_chars.next().unwrap_or('\\');
                    if s_chars.next() != Some(escaped) {
                        return false;
                    }
                }
                c => {
                    // Literal match
                    if s_chars.next() != Some(c) {
                        return false;
                    }
                }
            }
        }

        s_chars.next().is_none()
    }
}

impl Default for PubSub {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Subscribe/Publish Tests ====================

    #[test]
    fn test_subscribe_publish() {
        let pubsub = PubSub::new();

        // Subscribe client 1 to channel
        pubsub.subscribe(1, &["news".to_string()]);

        // Publish message
        let receivers = pubsub.publish("news", b"hello");
        assert_eq!(receivers, 1);

        // Get message
        let messages = pubsub.get_messages(1);
        assert_eq!(messages.len(), 1);
    }

    #[test]
    fn test_subscribe_multiple_channels() {
        let pubsub = PubSub::new();

        // Subscribe client to multiple channels
        let responses = pubsub.subscribe(1, &[
            "news".to_string(),
            "sports".to_string(),
            "weather".to_string(),
        ]);

        assert_eq!(responses.len(), 3);

        // Verify subscription counts in responses
        for (i, resp) in responses.iter().enumerate() {
            if let RespValue::Array(Some(arr)) = resp {
                assert_eq!(arr.len(), 3);
                // Third element is count
                if let RespValue::Integer(count) = &arr[2] {
                    assert_eq!(*count as usize, i + 1);
                }
            }
        }
    }

    #[test]
    fn test_subscribe_multiple_clients() {
        let pubsub = PubSub::new();

        // Multiple clients subscribe to same channel
        pubsub.subscribe(1, &["news".to_string()]);
        pubsub.subscribe(2, &["news".to_string()]);
        pubsub.subscribe(3, &["news".to_string()]);

        // Publish message
        let receivers = pubsub.publish("news", b"hello");
        assert_eq!(receivers, 3);

        // Each client should have the message
        assert_eq!(pubsub.get_messages(1).len(), 1);
        assert_eq!(pubsub.get_messages(2).len(), 1);
        assert_eq!(pubsub.get_messages(3).len(), 1);
    }

    #[test]
    fn test_publish_no_subscribers() {
        let pubsub = PubSub::new();

        // Publish to channel with no subscribers
        let receivers = pubsub.publish("empty_channel", b"hello");
        assert_eq!(receivers, 0);
    }

    #[test]
    fn test_publish_wrong_channel() {
        let pubsub = PubSub::new();

        pubsub.subscribe(1, &["news".to_string()]);

        // Publish to different channel
        let receivers = pubsub.publish("sports", b"hello");
        assert_eq!(receivers, 0);

        // No messages for client
        assert_eq!(pubsub.get_messages(1).len(), 0);
    }

    // ==================== Unsubscribe Tests ====================

    #[test]
    fn test_unsubscribe_specific() {
        let pubsub = PubSub::new();

        pubsub.subscribe(1, &[
            "news".to_string(),
            "sports".to_string(),
        ]);

        // Unsubscribe from one channel
        pubsub.unsubscribe(1, Some(&["news".to_string()]));

        // Publish to both channels
        let news_receivers = pubsub.publish("news", b"hello");
        let sports_receivers = pubsub.publish("sports", b"hello");

        assert_eq!(news_receivers, 0);
        assert_eq!(sports_receivers, 1);
    }

    #[test]
    fn test_unsubscribe_all() {
        let pubsub = PubSub::new();

        pubsub.subscribe(1, &[
            "news".to_string(),
            "sports".to_string(),
            "weather".to_string(),
        ]);

        // Unsubscribe from all channels
        pubsub.unsubscribe(1, None);

        assert!(!pubsub.has_subscriptions(1));
        assert_eq!(pubsub.subscription_count(1), 0);
    }

    // ==================== Pattern Subscription Tests ====================

    #[test]
    fn test_psubscribe() {
        let pubsub = PubSub::new();

        // Subscribe to pattern
        pubsub.psubscribe(1, &["news.*".to_string()]);

        // Publish to matching channel
        let receivers = pubsub.publish("news.sports", b"hello");
        assert_eq!(receivers, 1);

        let messages = pubsub.get_messages(1);
        assert_eq!(messages.len(), 1);

        // Verify pmessage format
        if let RespValue::Array(Some(arr)) = &messages[0] {
            assert_eq!(arr.len(), 4);
            assert_eq!(arr[0].as_str(), Some("pmessage"));
            assert_eq!(arr[1].as_str(), Some("news.*"));
            assert_eq!(arr[2].as_str(), Some("news.sports"));
        }
    }

    #[test]
    fn test_psubscribe_multiple_patterns() {
        let pubsub = PubSub::new();

        pubsub.psubscribe(1, &[
            "news.*".to_string(),
            "weather.*".to_string(),
        ]);

        let receivers1 = pubsub.publish("news.sports", b"hello");
        let receivers2 = pubsub.publish("weather.today", b"sunny");
        let receivers3 = pubsub.publish("politics", b"no match");

        assert_eq!(receivers1, 1);
        assert_eq!(receivers2, 1);
        assert_eq!(receivers3, 0);
    }

    #[test]
    fn test_punsubscribe() {
        let pubsub = PubSub::new();

        pubsub.psubscribe(1, &[
            "news.*".to_string(),
            "sports.*".to_string(),
        ]);

        // Unsubscribe from one pattern
        pubsub.punsubscribe(1, Some(&["news.*".to_string()]));

        let news_receivers = pubsub.publish("news.sports", b"hello");
        let sports_receivers = pubsub.publish("sports.soccer", b"goal");

        assert_eq!(news_receivers, 0);
        assert_eq!(sports_receivers, 1);
    }

    #[test]
    fn test_punsubscribe_all() {
        let pubsub = PubSub::new();

        pubsub.psubscribe(1, &[
            "news.*".to_string(),
            "sports.*".to_string(),
        ]);

        pubsub.punsubscribe(1, None);

        assert!(!pubsub.has_subscriptions(1));
    }

    // ==================== Pattern Matching Tests ====================

    #[test]
    fn test_pattern_matching() {
        assert!(PubSub::match_pattern("*", "anything"));
        assert!(PubSub::match_pattern("news.*", "news.sports"));
        assert!(PubSub::match_pattern("news.?", "news.a"));
        assert!(!PubSub::match_pattern("news.?", "news.ab"));
        assert!(PubSub::match_pattern("h[ae]llo", "hello"));
        assert!(PubSub::match_pattern("h[ae]llo", "hallo"));
    }

    #[test]
    fn test_pattern_matching_asterisk() {
        assert!(PubSub::match_pattern("*", ""));
        assert!(PubSub::match_pattern("*", "anything"));
        assert!(PubSub::match_pattern("foo*", "foo"));
        assert!(PubSub::match_pattern("foo*", "foobar"));
        assert!(PubSub::match_pattern("*bar", "bar"));
        assert!(PubSub::match_pattern("*bar", "foobar"));
        assert!(PubSub::match_pattern("f*r", "foobar"));
    }

    #[test]
    fn test_pattern_matching_question_mark() {
        assert!(PubSub::match_pattern("?", "a"));
        assert!(!PubSub::match_pattern("?", ""));
        assert!(!PubSub::match_pattern("?", "ab"));
        assert!(PubSub::match_pattern("a?c", "abc"));
        assert!(!PubSub::match_pattern("a?c", "ac"));
    }

    #[test]
    fn test_pattern_matching_character_class() {
        assert!(PubSub::match_pattern("[abc]", "a"));
        assert!(PubSub::match_pattern("[abc]", "b"));
        assert!(PubSub::match_pattern("[abc]", "c"));
        assert!(!PubSub::match_pattern("[abc]", "d"));
    }

    #[test]
    fn test_pattern_matching_negated_class() {
        assert!(!PubSub::match_pattern("[^abc]", "a"));
        assert!(PubSub::match_pattern("[^abc]", "d"));
    }

    #[test]
    fn test_pattern_matching_escape() {
        assert!(PubSub::match_pattern("\\*", "*"));
        assert!(!PubSub::match_pattern("\\*", "a"));
        assert!(PubSub::match_pattern("\\?", "?"));
        assert!(!PubSub::match_pattern("\\?", "a"));
    }

    #[test]
    fn test_pattern_matching_exact() {
        assert!(PubSub::match_pattern("hello", "hello"));
        assert!(!PubSub::match_pattern("hello", "hello!"));
        assert!(!PubSub::match_pattern("hello", "hell"));
    }

    // ==================== Mixed Channel and Pattern Tests ====================

    #[test]
    fn test_mixed_subscriptions() {
        let pubsub = PubSub::new();

        // Subscribe to both channel and pattern
        pubsub.subscribe(1, &["news.sports".to_string()]);
        pubsub.psubscribe(1, &["news.*".to_string()]);

        // Publish to news.sports - should match both
        let receivers = pubsub.publish("news.sports", b"hello");
        assert_eq!(receivers, 2);

        let messages = pubsub.get_messages(1);
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_subscription_count() {
        let pubsub = PubSub::new();

        pubsub.subscribe(1, &["ch1".to_string(), "ch2".to_string()]);
        pubsub.psubscribe(1, &["pat1".to_string()]);

        assert_eq!(pubsub.subscription_count(1), 3);
        assert!(pubsub.has_subscriptions(1));
    }

    // ==================== Utility Tests ====================

    #[test]
    fn test_numsub() {
        let pubsub = PubSub::new();

        pubsub.subscribe(1, &["news".to_string()]);
        pubsub.subscribe(2, &["news".to_string()]);
        pubsub.subscribe(3, &["sports".to_string()]);

        assert_eq!(pubsub.numsub("news"), 2);
        assert_eq!(pubsub.numsub("sports"), 1);
        assert_eq!(pubsub.numsub("unknown"), 0);
    }

    #[test]
    fn test_numpat() {
        let pubsub = PubSub::new();

        pubsub.psubscribe(1, &["news.*".to_string()]);
        pubsub.psubscribe(2, &["news.*".to_string()]);
        pubsub.psubscribe(2, &["sports.*".to_string()]);

        assert_eq!(pubsub.numpat(), 3);
    }

    #[test]
    fn test_channels_list() {
        let pubsub = PubSub::new();

        pubsub.subscribe(1, &[
            "news".to_string(),
            "sports".to_string(),
            "weather".to_string(),
        ]);

        let all_channels = pubsub.channels_list(None);
        assert_eq!(all_channels.len(), 3);

        // Filter with pattern
        let news_channels = pubsub.channels_list(Some("news"));
        assert_eq!(news_channels.len(), 1);
    }

    #[test]
    fn test_client_disconnected() {
        let pubsub = PubSub::new();

        pubsub.subscribe(1, &["news".to_string()]);
        pubsub.psubscribe(1, &["sports.*".to_string()]);

        // Publish messages before disconnect
        pubsub.publish("news", b"hello");

        // Disconnect client
        pubsub.client_disconnected(1);

        // Verify cleanup
        assert!(!pubsub.has_subscriptions(1));
        assert_eq!(pubsub.get_messages(1).len(), 0);
        assert_eq!(pubsub.numsub("news"), 0);
    }

    #[test]
    fn test_message_format() {
        let pubsub = PubSub::new();

        pubsub.subscribe(1, &["news".to_string()]);
        pubsub.publish("news", b"hello world");

        let messages = pubsub.get_messages(1);
        assert_eq!(messages.len(), 1);

        // Verify message structure
        if let RespValue::Array(Some(arr)) = &messages[0] {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0].as_str(), Some("message"));
            assert_eq!(arr[1].as_str(), Some("news"));
            assert_eq!(arr[2].as_bytes(), Some(b"hello world".as_slice()));
        } else {
            panic!("Expected array message");
        }
    }

    #[test]
    fn test_default() {
        let pubsub: PubSub = Default::default();
        assert_eq!(pubsub.numpat(), 0);
    }
}
