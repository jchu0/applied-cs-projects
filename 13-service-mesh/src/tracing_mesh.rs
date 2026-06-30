//! Distributed tracing support.

use std::collections::HashMap;
use std::time::SystemTime;

/// Span context for distributed tracing.
#[derive(Debug, Clone, Copy)]
pub struct SpanContext {
    /// Trace ID.
    pub trace_id: u128,
    /// Span ID.
    pub span_id: u64,
    /// Trace flags.
    pub flags: u8,
}

/// A span representing a single operation.
#[derive(Debug, Clone)]
pub struct Span {
    /// Trace ID.
    pub trace_id: u128,
    /// Span ID.
    pub span_id: u64,
    /// Parent span ID.
    pub parent_span_id: Option<u64>,
    /// Span name/operation.
    pub name: String,
    /// Service name.
    pub service_name: String,
    /// Start time.
    pub start_time: SystemTime,
    /// End time.
    pub end_time: Option<SystemTime>,
    /// Tags.
    pub tags: HashMap<String, String>,
    /// Logs/events.
    pub logs: Vec<SpanLog>,
}

impl Span {
    /// Add a tag to the span.
    pub fn set_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.tags.insert(key.into(), value.into());
    }

    /// Log an event.
    pub fn log(&mut self, message: impl Into<String>) {
        self.logs.push(SpanLog {
            timestamp: SystemTime::now(),
            message: message.into(),
        });
    }

    /// Finish the span.
    pub fn finish(&mut self) {
        self.end_time = Some(SystemTime::now());
    }

    /// Get span context.
    pub fn context(&self) -> SpanContext {
        SpanContext {
            trace_id: self.trace_id,
            span_id: self.span_id,
            flags: 1, // Sampled
        }
    }
}

/// A log entry in a span.
#[derive(Debug, Clone)]
pub struct SpanLog {
    /// Timestamp.
    pub timestamp: SystemTime,
    /// Log message.
    pub message: String,
}

/// Sampler for deciding whether to trace.
pub enum Sampler {
    /// Always sample.
    AlwaysOn,
    /// Never sample.
    AlwaysOff,
    /// Probabilistic sampling.
    Probabilistic { rate: f64 },
}

impl Sampler {
    /// Check if request should be sampled.
    pub fn should_sample(&self) -> bool {
        match self {
            Sampler::AlwaysOn => true,
            Sampler::AlwaysOff => false,
            Sampler::Probabilistic { rate } => rand::random::<f64>() < *rate,
        }
    }
}

/// Tracer for creating and managing spans.
pub struct Tracer {
    /// Service name.
    service_name: String,
    /// Collector endpoint.
    collector_endpoint: String,
    /// Sampler.
    sampler: Sampler,
}

impl Tracer {
    /// Create a new tracer.
    pub fn new(service_name: String, collector_endpoint: String) -> Self {
        Self {
            service_name,
            collector_endpoint,
            sampler: Sampler::AlwaysOn,
        }
    }

    /// Set sampler.
    pub fn with_sampler(mut self, sampler: Sampler) -> Self {
        self.sampler = sampler;
        self
    }

    /// Start a new span.
    pub fn start_span(&self, name: &str, parent: Option<&SpanContext>) -> Span {
        let trace_id = parent.map(|p| p.trace_id).unwrap_or_else(rand::random);
        let span_id: u64 = rand::random();
        let parent_span_id = parent.map(|p| p.span_id);

        Span {
            trace_id,
            span_id,
            parent_span_id,
            name: name.to_string(),
            service_name: self.service_name.clone(),
            start_time: SystemTime::now(),
            end_time: None,
            tags: HashMap::new(),
            logs: Vec::new(),
        }
    }

    /// Extract span context from headers (W3C Trace Context format).
    pub fn extract_context(&self, headers: &HashMap<String, String>) -> Option<SpanContext> {
        if let Some(traceparent) = headers.get("traceparent") {
            let parts: Vec<&str> = traceparent.split('-').collect();
            if parts.len() >= 4 {
                let trace_id = u128::from_str_radix(parts[1], 16).ok()?;
                let span_id = u64::from_str_radix(parts[2], 16).ok()?;
                let flags = u8::from_str_radix(parts[3], 16).ok()?;

                return Some(SpanContext {
                    trace_id,
                    span_id,
                    flags,
                });
            }
        }
        None
    }

    /// Inject span context into headers.
    pub fn inject_context(&self, context: &SpanContext, headers: &mut HashMap<String, String>) {
        let traceparent = format!(
            "00-{:032x}-{:016x}-{:02x}",
            context.trace_id, context.span_id, context.flags
        );
        headers.insert("traceparent".to_string(), traceparent);
    }

    /// Check if should sample.
    pub fn should_sample(&self) -> bool {
        self.sampler.should_sample()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_creation() {
        let tracer = Tracer::new("test-service".to_string(), "localhost:14268".to_string());

        let mut span = tracer.start_span("test-operation", None);
        span.set_tag("key", "value");
        span.log("test event");
        span.finish();

        assert_eq!(span.name, "test-operation");
        assert!(span.end_time.is_some());
        assert_eq!(span.tags.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_context_injection() {
        let tracer = Tracer::new("test".to_string(), "localhost".to_string());
        let span = tracer.start_span("test", None);
        let context = span.context();

        let mut headers = HashMap::new();
        tracer.inject_context(&context, &mut headers);

        let extracted = tracer.extract_context(&headers).unwrap();
        assert_eq!(extracted.trace_id, context.trace_id);
        assert_eq!(extracted.span_id, context.span_id);
    }
}
