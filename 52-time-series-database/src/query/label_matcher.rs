//! Label matchers for flexible series selection
//!
//! Provides Prometheus-style label matching:
//! - Exact match: {label="value"}
//! - Not equal: {label!="value"}
//! - Regex match: {label=~"regex"}
//! - Regex not match: {label!~"regex"}

use regex::Regex;
use std::collections::HashMap;

use crate::types::Tags;

/// Label matcher operation
#[derive(Debug, Clone)]
pub enum MatchOp {
    /// Exact equality: =
    Equal,
    /// Not equal: !=
    NotEqual,
    /// Regex match: =~
    RegexMatch,
    /// Regex not match: !~
    RegexNotMatch,
}

/// A single label matcher
#[derive(Debug, Clone)]
pub struct LabelMatcher {
    /// Label name
    pub name: String,
    /// Match operation
    pub op: MatchOp,
    /// Value to match against
    pub value: String,
    /// Compiled regex (for regex operations)
    regex: Option<Regex>,
}

impl LabelMatcher {
    /// Create an equality matcher
    pub fn eq(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            op: MatchOp::Equal,
            value: value.into(),
            regex: None,
        }
    }

    /// Create a not-equal matcher
    pub fn neq(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            op: MatchOp::NotEqual,
            value: value.into(),
            regex: None,
        }
    }

    /// Create a regex match matcher
    pub fn regex(name: impl Into<String>, pattern: impl Into<String>) -> Result<Self, regex::Error> {
        let pattern = pattern.into();
        let regex = Regex::new(&pattern)?;
        Ok(Self {
            name: name.into(),
            op: MatchOp::RegexMatch,
            value: pattern,
            regex: Some(regex),
        })
    }

    /// Create a regex not-match matcher
    pub fn not_regex(name: impl Into<String>, pattern: impl Into<String>) -> Result<Self, regex::Error> {
        let pattern = pattern.into();
        let regex = Regex::new(&pattern)?;
        Ok(Self {
            name: name.into(),
            op: MatchOp::RegexNotMatch,
            value: pattern,
            regex: Some(regex),
        })
    }

    /// Check if tags match this matcher
    pub fn matches(&self, tags: &Tags) -> bool {
        let label_value = tags.get(&self.name).map(|s| s.as_str()).unwrap_or("");

        match self.op {
            MatchOp::Equal => label_value == self.value,
            MatchOp::NotEqual => label_value != self.value,
            MatchOp::RegexMatch => {
                self.regex.as_ref().map_or(false, |r| r.is_match(label_value))
            }
            MatchOp::RegexNotMatch => {
                self.regex.as_ref().map_or(true, |r| !r.is_match(label_value))
            }
        }
    }
}

/// A set of label matchers (all must match)
#[derive(Debug, Clone, Default)]
pub struct LabelMatchers {
    matchers: Vec<LabelMatcher>,
}

impl LabelMatchers {
    /// Create a new empty matcher set
    pub fn new() -> Self {
        Self {
            matchers: Vec::new(),
        }
    }

    /// Add a matcher
    pub fn add(&mut self, matcher: LabelMatcher) {
        self.matchers.push(matcher);
    }

    /// Add a matcher (builder pattern)
    pub fn with(mut self, matcher: LabelMatcher) -> Self {
        self.matchers.push(matcher);
        self
    }

    /// Add an equality matcher
    pub fn eq(self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.with(LabelMatcher::eq(name, value))
    }

    /// Add a not-equal matcher
    pub fn neq(self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.with(LabelMatcher::neq(name, value))
    }

    /// Add a regex matcher
    pub fn regex(self, name: impl Into<String>, pattern: impl Into<String>) -> Result<Self, regex::Error> {
        Ok(self.with(LabelMatcher::regex(name, pattern)?))
    }

    /// Check if tags match all matchers
    pub fn matches(&self, tags: &Tags) -> bool {
        self.matchers.iter().all(|m| m.matches(tags))
    }

    /// Check if matcher set is empty
    pub fn is_empty(&self) -> bool {
        self.matchers.is_empty()
    }

    /// Get number of matchers
    pub fn len(&self) -> usize {
        self.matchers.len()
    }

    /// Get matchers
    pub fn matchers(&self) -> &[LabelMatcher] {
        &self.matchers
    }
}

/// Builder for creating label matchers from a string expression
pub struct LabelMatcherBuilder;

impl LabelMatcherBuilder {
    /// Parse a label matcher expression
    /// Format: "name op value" where op is =, !=, =~, or !~
    pub fn parse(expr: &str) -> Result<LabelMatcher, String> {
        let expr = expr.trim();

        // Try each operator in order of length (longest first)
        if let Some((name, value)) = Self::try_split(expr, "!~") {
            return LabelMatcher::not_regex(name, value)
                .map_err(|e| format!("Invalid regex: {}", e));
        }
        if let Some((name, value)) = Self::try_split(expr, "=~") {
            return LabelMatcher::regex(name, value)
                .map_err(|e| format!("Invalid regex: {}", e));
        }
        if let Some((name, value)) = Self::try_split(expr, "!=") {
            return Ok(LabelMatcher::neq(name, value));
        }
        if let Some((name, value)) = Self::try_split(expr, "=") {
            return Ok(LabelMatcher::eq(name, value));
        }

        Err(format!("Invalid label matcher expression: {}", expr))
    }

    /// Parse multiple matchers from a comma-separated string
    /// Format: "name1=value1, name2!=value2, ..."
    pub fn parse_many(expr: &str) -> Result<LabelMatchers, String> {
        let mut matchers = LabelMatchers::new();

        for part in expr.split(',') {
            let part = part.trim();
            if !part.is_empty() {
                matchers.add(Self::parse(part)?);
            }
        }

        Ok(matchers)
    }

    fn try_split(expr: &str, op: &str) -> Option<(String, String)> {
        expr.find(op).map(|pos| {
            let name = expr[..pos].trim().to_string();
            let value = expr[pos + op.len()..].trim();
            // Remove surrounding quotes if present
            let value = value.trim_matches('"').trim_matches('\'').to_string();
            (name, value)
        })
    }
}

/// Series selector combining metric name and label matchers
#[derive(Debug, Clone)]
pub struct SeriesSelector {
    /// Metric name (or regex pattern)
    pub metric_name: Option<String>,
    /// Whether metric_name is a regex
    pub metric_regex: Option<Regex>,
    /// Label matchers
    pub matchers: LabelMatchers,
}

impl SeriesSelector {
    /// Create a selector for a specific metric
    pub fn metric(name: impl Into<String>) -> Self {
        Self {
            metric_name: Some(name.into()),
            metric_regex: None,
            matchers: LabelMatchers::new(),
        }
    }

    /// Create a selector with a metric regex
    pub fn metric_pattern(pattern: &str) -> Result<Self, regex::Error> {
        Ok(Self {
            metric_name: Some(pattern.to_string()),
            metric_regex: Some(Regex::new(pattern)?),
            matchers: LabelMatchers::new(),
        })
    }

    /// Create a selector with only label matchers
    pub fn labels(matchers: LabelMatchers) -> Self {
        Self {
            metric_name: None,
            metric_regex: None,
            matchers,
        }
    }

    /// Add label matchers
    pub fn with_matchers(mut self, matchers: LabelMatchers) -> Self {
        self.matchers = matchers;
        self
    }

    /// Add a single label matcher
    pub fn with_matcher(mut self, matcher: LabelMatcher) -> Self {
        self.matchers.add(matcher);
        self
    }

    /// Check if a metric name and tags match this selector
    pub fn matches(&self, metric_name: &str, tags: &Tags) -> bool {
        // Check metric name
        if let Some(ref regex) = self.metric_regex {
            if !regex.is_match(metric_name) {
                return false;
            }
        } else if let Some(ref name) = self.metric_name {
            if name != metric_name {
                return false;
            }
        }

        // Check label matchers
        self.matchers.matches(tags)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tags(pairs: &[(&str, &str)]) -> Tags {
        let mut tags = Tags::new();
        for (k, v) in pairs {
            tags.insert(k.to_string(), v.to_string());
        }
        tags
    }

    #[test]
    fn test_equality_matcher() {
        let matcher = LabelMatcher::eq("host", "server1");

        assert!(matcher.matches(&make_tags(&[("host", "server1")])));
        assert!(!matcher.matches(&make_tags(&[("host", "server2")])));
        assert!(!matcher.matches(&make_tags(&[])));
    }

    #[test]
    fn test_not_equal_matcher() {
        let matcher = LabelMatcher::neq("host", "server1");

        assert!(!matcher.matches(&make_tags(&[("host", "server1")])));
        assert!(matcher.matches(&make_tags(&[("host", "server2")])));
        assert!(matcher.matches(&make_tags(&[])));
    }

    #[test]
    fn test_regex_matcher() {
        let matcher = LabelMatcher::regex("host", "server[0-9]+").unwrap();

        assert!(matcher.matches(&make_tags(&[("host", "server1")])));
        assert!(matcher.matches(&make_tags(&[("host", "server123")])));
        assert!(!matcher.matches(&make_tags(&[("host", "web1")])));
        assert!(!matcher.matches(&make_tags(&[])));
    }

    #[test]
    fn test_not_regex_matcher() {
        let matcher = LabelMatcher::not_regex("host", "server[0-9]+").unwrap();

        assert!(!matcher.matches(&make_tags(&[("host", "server1")])));
        assert!(matcher.matches(&make_tags(&[("host", "web1")])));
        assert!(matcher.matches(&make_tags(&[])));
    }

    #[test]
    fn test_label_matchers_all() {
        let matchers = LabelMatchers::new()
            .eq("host", "server1")
            .eq("region", "us-east");

        assert!(matchers.matches(&make_tags(&[
            ("host", "server1"),
            ("region", "us-east"),
        ])));

        // Missing one label
        assert!(!matchers.matches(&make_tags(&[("host", "server1")])));

        // Wrong value
        assert!(!matchers.matches(&make_tags(&[
            ("host", "server1"),
            ("region", "us-west"),
        ])));
    }

    #[test]
    fn test_label_matcher_builder_parse() {
        let matcher = LabelMatcherBuilder::parse("host = server1").unwrap();
        assert!(matches!(matcher.op, MatchOp::Equal));
        assert_eq!(matcher.name, "host");
        assert_eq!(matcher.value, "server1");

        let matcher = LabelMatcherBuilder::parse("host != server1").unwrap();
        assert!(matches!(matcher.op, MatchOp::NotEqual));

        let matcher = LabelMatcherBuilder::parse(r#"host =~ "server[0-9]+""#).unwrap();
        assert!(matches!(matcher.op, MatchOp::RegexMatch));
        assert_eq!(matcher.value, "server[0-9]+");

        let matcher = LabelMatcherBuilder::parse("host !~ server").unwrap();
        assert!(matches!(matcher.op, MatchOp::RegexNotMatch));
    }

    #[test]
    fn test_parse_many() {
        let matchers = LabelMatcherBuilder::parse_many(
            r#"host = server1, region != us-west, env =~ "prod.*""#
        ).unwrap();

        assert_eq!(matchers.len(), 3);
    }

    #[test]
    fn test_series_selector_metric() {
        let selector = SeriesSelector::metric("cpu_usage");

        assert!(selector.matches("cpu_usage", &Tags::new()));
        assert!(!selector.matches("memory_usage", &Tags::new()));
    }

    #[test]
    fn test_series_selector_metric_pattern() {
        let selector = SeriesSelector::metric_pattern("cpu_.*").unwrap();

        assert!(selector.matches("cpu_usage", &Tags::new()));
        assert!(selector.matches("cpu_system", &Tags::new()));
        assert!(!selector.matches("memory_usage", &Tags::new()));
    }

    #[test]
    fn test_series_selector_with_labels() {
        let selector = SeriesSelector::metric("cpu_usage")
            .with_matchers(LabelMatchers::new().eq("host", "server1"));

        assert!(selector.matches("cpu_usage", &make_tags(&[("host", "server1")])));
        assert!(!selector.matches("cpu_usage", &make_tags(&[("host", "server2")])));
        assert!(!selector.matches("memory_usage", &make_tags(&[("host", "server1")])));
    }

    #[test]
    fn test_empty_matchers() {
        let matchers = LabelMatchers::new();
        assert!(matchers.is_empty());
        assert!(matchers.matches(&Tags::new()));
        assert!(matchers.matches(&make_tags(&[("any", "value")])));
    }

    #[test]
    fn test_invalid_regex() {
        let result = LabelMatcher::regex("host", "[invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_invalid() {
        let result = LabelMatcherBuilder::parse("invalid_no_operator");
        assert!(result.is_err());
    }
}
