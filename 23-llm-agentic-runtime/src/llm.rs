//! LLM client abstraction for the agent reasoning loop.
//!
//! The agent runtime is model-agnostic: it talks to whatever implements
//! [`LlmClient`]. Two implementations ship here:
//!
//! - [`AnthropicClient`] — a real client calling the Anthropic Messages API
//!   (`POST /v1/messages`) over HTTPS with `reqwest`. Rust has no official
//!   Anthropic SDK, so this is a thin raw-HTTP wrapper.
//! - [`MockLlmClient`] — a deterministic client that replays scripted
//!   responses, so the agent loop can be unit-tested with no network or API key.

use crate::{Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Mutex;

/// A single chat message exchanged with the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Either `"user"` or `"assistant"`.
    pub role: String,
    /// Message text.
    pub content: String,
}

impl ChatMessage {
    /// Construct a user-role message.
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".to_string(), content: content.into() }
    }

    /// Construct an assistant-role message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".to_string(), content: content.into() }
    }
}

/// A pluggable LLM backend. Implement this to drive the agent with any model.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Generate a completion for the given system prompt and message history,
    /// returning the assistant's text.
    async fn complete(&self, system: &str, messages: &[ChatMessage]) -> Result<String>;
}

/// Real client for the Anthropic Messages API.
///
/// Defaults to `claude-opus-4-8`. The API key is read from the
/// `ANTHROPIC_API_KEY` environment variable via [`AnthropicClient::from_env`].
pub struct AnthropicClient {
    api_key: String,
    model: String,
    max_tokens: u32,
    base_url: String,
    http: reqwest::Client,
}

impl AnthropicClient {
    /// Build a client, reading the API key from `ANTHROPIC_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| Error::LlmError("ANTHROPIC_API_KEY is not set".to_string()))?;
        Ok(Self::new(api_key))
    }

    /// Build a client with an explicit API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "claude-opus-4-8".to_string(),
            max_tokens: 1024,
            base_url: "https://api.anthropic.com".to_string(),
            http: reqwest::Client::new(),
        }
    }

    /// Override the model id (default `claude-opus-4-8`).
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the per-response token cap (default 1024).
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Override the API base URL (useful for testing against a local mock server).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

#[async_trait]
impl LlmClient for AnthropicClient {
    async fn complete(&self, system: &str, messages: &[ChatMessage]) -> Result<String> {
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
        });

        let resp = self
            .http
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::LlmError(format!("request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::LlmError(format!("API returned {status}: {text}")));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| Error::LlmError(format!("could not decode response: {e}")))?;

        // The Messages API returns `content` as an array of blocks; concatenate
        // the text of every `type == "text"` block.
        let text = json
            .get("content")
            .and_then(|c| c.as_array())
            .map(|blocks| {
                blocks
                    .iter()
                    .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                    .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default();

        if text.is_empty() {
            return Err(Error::LlmError("response contained no text content".to_string()));
        }
        Ok(text)
    }
}

/// Deterministic client that replays scripted responses in order.
///
/// Used by tests and offline runs to exercise the agent loop without a network
/// call or API key. Once the scripted responses are exhausted it returns a
/// `finish` step so a loop can never hang.
pub struct MockLlmClient {
    responses: Mutex<VecDeque<String>>,
}

impl MockLlmClient {
    /// Build a mock that returns the given responses in order.
    pub fn new(responses: Vec<String>) -> Self {
        Self { responses: Mutex::new(responses.into_iter().collect()) }
    }

    /// Build a mock that finishes immediately with the given answer.
    pub fn finishing(answer: impl Into<String>) -> Self {
        let step = serde_json::json!({
            "thought": "I can answer this directly.",
            "action": "finish",
            "action_input": { "answer": answer.into() },
        });
        Self::new(vec![step.to_string()])
    }
}

#[async_trait]
impl LlmClient for MockLlmClient {
    async fn complete(&self, _system: &str, _messages: &[ChatMessage]) -> Result<String> {
        let mut queue = self.responses.lock().expect("mock mutex poisoned");
        Ok(queue.pop_front().unwrap_or_else(|| {
            serde_json::json!({
                "thought": "No further scripted steps; finishing.",
                "action": "finish",
                "action_input": { "answer": "done" },
            })
            .to_string()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_replays_in_order() {
        let client = MockLlmClient::new(vec!["a".to_string(), "b".to_string()]);
        assert_eq!(client.complete("s", &[]).await.unwrap(), "a");
        assert_eq!(client.complete("s", &[]).await.unwrap(), "b");
        // Exhausted → returns a finish step, never errors.
        assert!(client.complete("s", &[]).await.unwrap().contains("finish"));
    }

    #[tokio::test]
    async fn finishing_mock_emits_answer() {
        let client = MockLlmClient::finishing("42");
        let out = client.complete("s", &[ChatMessage::user("q")]).await.unwrap();
        assert!(out.contains("\"finish\""));
        assert!(out.contains("42"));
    }
}
