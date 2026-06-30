//! Safety guardrails for agent execution.

use crate::{Error, Result};
use regex::Regex;
use std::collections::HashMap;

/// Result of validation.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed.
    pub passed: bool,
    /// Reason for failure (if any).
    pub reason: Option<String>,
}

impl ValidationResult {
    /// Create a passed result.
    pub fn pass() -> Self {
        Self {
            passed: true,
            reason: None,
        }
    }

    /// Create a failed result.
    pub fn fail(reason: impl Into<String>) -> Self {
        Self {
            passed: false,
            reason: Some(reason.into()),
        }
    }
}

/// Safety guardrails for agent execution.
pub struct Guardrails {
    /// Blocked regex patterns.
    blocked_patterns: Vec<Regex>,
    /// Allowed tools.
    allowed_tools: Option<Vec<String>>,
    /// Blocked tools.
    blocked_tools: Vec<String>,
    /// Maximum input length.
    max_input_length: Option<usize>,
    /// Maximum output length.
    max_output_length: Option<usize>,
    /// Custom validators.
    custom_validators: Vec<Box<dyn Fn(&str) -> bool + Send + Sync>>,
}

impl Guardrails {
    /// Create new guardrails with defaults.
    pub fn new() -> Self {
        Self {
            blocked_patterns: Vec::new(),
            allowed_tools: None,
            blocked_tools: Vec::new(),
            max_input_length: Some(100000),
            max_output_length: Some(100000),
            custom_validators: Vec::new(),
        }
    }

    /// Add a blocked pattern.
    pub fn block_pattern(&mut self, pattern: &str) -> Result<()> {
        let regex = Regex::new(pattern).map_err(|e| {
            Error::ValidationFailed(format!("Invalid regex: {}", e))
        })?;
        self.blocked_patterns.push(regex);
        Ok(())
    }

    /// Set allowed tools (whitelist).
    pub fn set_allowed_tools(&mut self, tools: Vec<String>) {
        self.allowed_tools = Some(tools);
    }

    /// Block specific tools.
    pub fn block_tool(&mut self, tool: impl Into<String>) {
        self.blocked_tools.push(tool.into());
    }

    /// Set maximum input length.
    pub fn set_max_input_length(&mut self, length: usize) {
        self.max_input_length = Some(length);
    }

    /// Set maximum output length.
    pub fn set_max_output_length(&mut self, length: usize) {
        self.max_output_length = Some(length);
    }

    /// Add custom validator.
    pub fn add_validator<F>(&mut self, validator: F)
    where
        F: Fn(&str) -> bool + Send + Sync + 'static,
    {
        self.custom_validators.push(Box::new(validator));
    }

    /// Validate input text.
    pub fn validate_input(&self, text: &str) -> ValidationResult {
        // Check length
        if let Some(max) = self.max_input_length {
            if text.len() > max {
                return ValidationResult::fail(format!(
                    "Input too long: {} > {}",
                    text.len(),
                    max
                ));
            }
        }

        // Check blocked patterns
        for pattern in &self.blocked_patterns {
            if pattern.is_match(text) {
                return ValidationResult::fail(format!(
                    "Blocked pattern detected: {}",
                    pattern.as_str()
                ));
            }
        }

        // Run custom validators
        for validator in &self.custom_validators {
            if !validator(text) {
                return ValidationResult::fail("Custom validation failed");
            }
        }

        ValidationResult::pass()
    }

    /// Validate output text.
    pub fn validate_output(&self, text: &str) -> ValidationResult {
        // Check length
        if let Some(max) = self.max_output_length {
            if text.len() > max {
                return ValidationResult::fail(format!(
                    "Output too long: {} > {}",
                    text.len(),
                    max
                ));
            }
        }

        // Check blocked patterns
        for pattern in &self.blocked_patterns {
            if pattern.is_match(text) {
                return ValidationResult::fail(format!(
                    "Blocked pattern in output: {}",
                    pattern.as_str()
                ));
            }
        }

        ValidationResult::pass()
    }

    /// Validate action before execution.
    pub fn validate_action(
        &self,
        action: &str,
        _action_input: &HashMap<String, serde_json::Value>,
    ) -> ValidationResult {
        // Check blocked tools
        if self.blocked_tools.contains(&action.to_string()) {
            return ValidationResult::fail(format!("Tool is blocked: {}", action));
        }

        // Check allowed tools whitelist
        if let Some(allowed) = &self.allowed_tools {
            if !allowed.contains(&action.to_string()) {
                return ValidationResult::fail(format!("Tool not allowed: {}", action));
            }
        }

        ValidationResult::pass()
    }
}

impl Default for Guardrails {
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-built guardrail patterns.
pub struct CommonGuardrails;

impl CommonGuardrails {
    /// Get patterns to block PII.
    pub fn pii_patterns() -> Vec<&'static str> {
        vec![
            r"\b\d{3}-\d{2}-\d{4}\b",           // SSN
            r"\b\d{16}\b",                       // Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", // Email
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",   // Phone
        ]
    }

    /// Get patterns to block harmful code.
    pub fn harmful_code_patterns() -> Vec<&'static str> {
        vec![
            r"os\.system",
            r"subprocess\.(run|call|Popen)",
            r"eval\s*\(",
            r"exec\s*\(",
            r"__import__",
            r#"open\s*\([^)]*,\s*['"]w"#,        // File writes
            r"shutil\.(rmtree|remove)",
        ]
    }

    /// Create guardrails with PII protection.
    pub fn with_pii_protection() -> Guardrails {
        let mut guardrails = Guardrails::new();
        for pattern in Self::pii_patterns() {
            guardrails.block_pattern(pattern).ok();
        }
        guardrails
    }

    /// Create guardrails with code safety.
    pub fn with_code_safety() -> Guardrails {
        let mut guardrails = Guardrails::new();
        for pattern in Self::harmful_code_patterns() {
            guardrails.block_pattern(pattern).ok();
        }
        guardrails
    }

    /// Create comprehensive guardrails.
    pub fn comprehensive() -> Guardrails {
        let mut guardrails = Guardrails::new();

        for pattern in Self::pii_patterns() {
            guardrails.block_pattern(pattern).ok();
        }

        for pattern in Self::harmful_code_patterns() {
            guardrails.block_pattern(pattern).ok();
        }

        guardrails
    }
}

/// SQL injection prevention.
pub struct SqlGuard;

impl SqlGuard {
    /// Check if SQL is safe (read-only).
    pub fn is_safe_query(query: &str) -> bool {
        let dangerous = [
            "DROP", "DELETE", "TRUNCATE", "ALTER",
            "CREATE", "INSERT", "UPDATE", "GRANT",
            "REVOKE", "--", ";",
        ];

        let query_upper = query.to_uppercase();

        // Only allow SELECT
        if !query_upper.trim().starts_with("SELECT") {
            return false;
        }

        // Check for dangerous patterns
        !dangerous.iter().any(|d| query_upper.contains(d))
    }

    /// Sanitize SQL parameter.
    pub fn sanitize_param(param: &str) -> String {
        param
            .replace('\'', "''")
            .replace('\\', "\\\\")
            .replace('\0', "")
    }
}

/// Content moderation.
pub struct ContentModeration {
    blocked_words: Vec<String>,
    min_length: Option<usize>,
    max_length: Option<usize>,
}

impl ContentModeration {
    /// Create new content moderation.
    pub fn new() -> Self {
        Self {
            blocked_words: Vec::new(),
            min_length: None,
            max_length: None,
        }
    }

    /// Add blocked word.
    pub fn block_word(&mut self, word: impl Into<String>) {
        self.blocked_words.push(word.into().to_lowercase());
    }

    /// Set minimum length.
    pub fn set_min_length(&mut self, length: usize) {
        self.min_length = Some(length);
    }

    /// Set maximum length.
    pub fn set_max_length(&mut self, length: usize) {
        self.max_length = Some(length);
    }

    /// Check content.
    pub fn check(&self, content: &str) -> ValidationResult {
        // Check length
        if let Some(min) = self.min_length {
            if content.len() < min {
                return ValidationResult::fail("Content too short");
            }
        }

        if let Some(max) = self.max_length {
            if content.len() > max {
                return ValidationResult::fail("Content too long");
            }
        }

        // Check blocked words
        let content_lower = content.to_lowercase();
        for word in &self.blocked_words {
            if content_lower.contains(word) {
                return ValidationResult::fail(format!("Blocked word detected: {}", word));
            }
        }

        ValidationResult::pass()
    }
}

impl Default for ContentModeration {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guardrails_input() {
        let mut guardrails = Guardrails::new();
        guardrails.block_pattern(r"password").unwrap();

        let result = guardrails.validate_input("my password is 1234");
        assert!(!result.passed);

        let result = guardrails.validate_input("this is fine");
        assert!(result.passed);
    }

    #[test]
    fn test_guardrails_length() {
        let mut guardrails = Guardrails::new();
        guardrails.set_max_input_length(10);

        let result = guardrails.validate_input("short");
        assert!(result.passed);

        let result = guardrails.validate_input("this is too long");
        assert!(!result.passed);
    }

    #[test]
    fn test_tool_validation() {
        let mut guardrails = Guardrails::new();
        guardrails.block_tool("dangerous_tool");
        guardrails.set_allowed_tools(vec!["safe_tool".to_string()]);

        let result = guardrails.validate_action("safe_tool", &HashMap::new());
        assert!(result.passed);

        let result = guardrails.validate_action("dangerous_tool", &HashMap::new());
        assert!(!result.passed);

        let result = guardrails.validate_action("unknown_tool", &HashMap::new());
        assert!(!result.passed);
    }

    #[test]
    fn test_pii_protection() {
        let guardrails = CommonGuardrails::with_pii_protection();

        let result = guardrails.validate_input("SSN: 123-45-6789");
        assert!(!result.passed);

        let result = guardrails.validate_input("test@example.com");
        assert!(!result.passed);

        let result = guardrails.validate_input("This is normal text");
        assert!(result.passed);
    }

    #[test]
    fn test_sql_guard() {
        assert!(SqlGuard::is_safe_query("SELECT * FROM users"));
        assert!(!SqlGuard::is_safe_query("DROP TABLE users"));
        assert!(!SqlGuard::is_safe_query("SELECT * FROM users; DROP TABLE users"));
        assert!(!SqlGuard::is_safe_query("DELETE FROM users WHERE 1=1"));
    }

    #[test]
    fn test_content_moderation() {
        let mut moderation = ContentModeration::new();
        moderation.block_word("spam");
        moderation.set_min_length(5);
        moderation.set_max_length(100);

        let result = moderation.check("This is valid content");
        assert!(result.passed);

        let result = moderation.check("spam message");
        assert!(!result.passed);

        let result = moderation.check("hi");
        assert!(!result.passed);
    }
}
