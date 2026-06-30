//! Tracing system for agent execution debugging and training data.

use crate::agent::{AgentContext, AgentResult, ThoughtActionObservation};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Trace entry for a single agent execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTrace {
    /// Unique trace ID.
    pub trace_id: String,
    /// Task being executed.
    pub task: String,
    /// Context configuration.
    pub context: TraceContext,
    /// Execution steps.
    pub steps: Vec<TraceStep>,
    /// Final result.
    pub result: Option<TraceResult>,
    /// Start time.
    pub start_time: DateTime<Utc>,
    /// End time.
    pub end_time: Option<DateTime<Utc>>,
}

/// Trace context (subset of AgentContext).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    /// Maximum steps.
    pub max_steps: usize,
    /// Timeout seconds.
    pub timeout: f64,
    /// Enabled tools.
    pub tools: Vec<String>,
}

/// Single step in trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStep {
    /// Step number.
    pub step: usize,
    /// Agent's thought.
    pub thought: String,
    /// Action taken.
    pub action: Option<String>,
    /// Action input.
    pub action_input: Option<HashMap<String, serde_json::Value>>,
    /// Observation.
    pub observation: Option<String>,
    /// Error if any.
    pub error: Option<String>,
    /// Latency in ms.
    pub latency_ms: f64,
}

/// Trace result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceResult {
    /// Success status.
    pub success: bool,
    /// Final answer.
    pub answer: Option<String>,
    /// Total tokens.
    pub total_tokens: usize,
    /// Total latency.
    pub total_latency_ms: f64,
    /// Error message.
    pub error: Option<String>,
}

/// Agent tracer for debugging and training data.
pub struct AgentTracer {
    /// Output directory.
    output_dir: PathBuf,
    /// Active traces.
    traces: HashMap<String, AgentTrace>,
    /// Enable file output.
    file_output: bool,
}

impl AgentTracer {
    /// Create a new tracer.
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        let output_dir = output_dir.into();

        // Create output directory if needed
        if !output_dir.exists() {
            fs::create_dir_all(&output_dir).ok();
        }

        Self {
            output_dir,
            traces: HashMap::new(),
            file_output: true,
        }
    }

    /// Disable file output.
    pub fn without_file_output(mut self) -> Self {
        self.file_output = false;
        self
    }

    /// Start a new trace.
    pub fn start_trace(&mut self, task: &str, context: &AgentContext) -> String {
        let trace_id = format!(
            "{}_{}",
            Utc::now().format("%Y%m%d_%H%M%S"),
            uuid::Uuid::new_v4().to_string().split('-').next().unwrap_or("0000")
        );

        let trace = AgentTrace {
            trace_id: trace_id.clone(),
            task: task.to_string(),
            context: TraceContext {
                max_steps: context.max_steps,
                timeout: context.timeout_seconds,
                tools: context.tools_enabled.clone(),
            },
            steps: Vec::new(),
            result: None,
            start_time: Utc::now(),
            end_time: None,
        };

        self.traces.insert(trace_id.clone(), trace);
        trace_id
    }

    /// Log a step in the trace.
    pub fn log_step(&mut self, trace_id: &str, step: &ThoughtActionObservation) {
        if let Some(trace) = self.traces.get_mut(trace_id) {
            trace.steps.push(TraceStep {
                step: step.step,
                thought: step.thought.clone(),
                action: step.action.clone(),
                action_input: step.action_input.clone(),
                observation: step.observation.clone(),
                error: step.error.clone(),
                latency_ms: step.latency_ms,
            });
        }
    }

    /// End trace and save.
    pub fn end_trace(&mut self, trace_id: &str, result: &AgentResult) {
        let should_save = self.file_output;
        let trace_to_save = if let Some(trace) = self.traces.get_mut(trace_id) {
            trace.result = Some(TraceResult {
                success: result.success,
                answer: result.answer.clone(),
                total_tokens: result.total_tokens,
                total_latency_ms: result.total_latency_ms,
                error: result.error.clone(),
            });
            trace.end_time = Some(Utc::now());

            // Clone trace for saving to avoid borrow issues
            if should_save {
                Some(trace.clone())
            } else {
                None
            }
        } else {
            None
        };

        // Save to file after mutable borrow is released
        if let Some(trace) = trace_to_save {
            self.save_trace(&trace);
        }
    }

    /// Save trace to file.
    fn save_trace(&self, trace: &AgentTrace) {
        let filename = format!("{}.json", trace.trace_id);
        let filepath = self.output_dir.join(filename);

        if let Ok(json) = serde_json::to_string_pretty(trace) {
            fs::write(filepath, json).ok();
        }
    }

    /// Get trace by ID.
    pub fn get_trace(&self, trace_id: &str) -> Option<&AgentTrace> {
        self.traces.get(trace_id)
    }

    /// Get all traces.
    pub fn all_traces(&self) -> impl Iterator<Item = &AgentTrace> {
        self.traces.values()
    }

    /// Export traces as training data.
    pub fn export_training_data(&self) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for trace in self.traces.values() {
            // Only export successful traces
            if let Some(result) = &trace.result {
                if !result.success {
                    continue;
                }
            } else {
                continue;
            }

            // Convert to conversation format
            let mut messages = vec![
                Message {
                    role: "user".to_string(),
                    content: trace.task.clone(),
                }
            ];

            for step in &trace.steps {
                // Assistant turn
                let mut content = format!("Thought: {}\n", step.thought);
                if let Some(action) = &step.action {
                    content.push_str(&format!("Action: {}\n", action));
                    if let Some(input) = &step.action_input {
                        content.push_str(&format!(
                            "Action Input: {}",
                            serde_json::to_string(input).unwrap_or_default()
                        ));
                    }
                }

                messages.push(Message {
                    role: "assistant".to_string(),
                    content,
                });

                // Observation turn
                if let Some(obs) = &step.observation {
                    messages.push(Message {
                        role: "user".to_string(),
                        content: format!("Observation: {}", obs),
                    });
                }
            }

            examples.push(TrainingExample { messages });
        }

        examples
    }

    /// Load traces from directory.
    pub fn load_traces(&mut self) -> usize {
        let mut count = 0;

        if let Ok(entries) = fs::read_dir(&self.output_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "json").unwrap_or(false) {
                    if let Ok(content) = fs::read_to_string(&path) {
                        if let Ok(trace) = serde_json::from_str::<AgentTrace>(&content) {
                            self.traces.insert(trace.trace_id.clone(), trace);
                            count += 1;
                        }
                    }
                }
            }
        }

        count
    }

    /// Clear all traces.
    pub fn clear(&mut self) {
        self.traces.clear();
    }

    /// Get trace statistics.
    pub fn statistics(&self) -> TraceStatistics {
        let total = self.traces.len();
        let successful = self
            .traces
            .values()
            .filter(|t| t.result.as_ref().map(|r| r.success).unwrap_or(false))
            .count();

        let total_steps: usize = self.traces.values().map(|t| t.steps.len()).sum();
        let avg_steps = if total > 0 {
            total_steps as f64 / total as f64
        } else {
            0.0
        };

        let total_latency: f64 = self
            .traces
            .values()
            .filter_map(|t| t.result.as_ref())
            .map(|r| r.total_latency_ms)
            .sum();

        let avg_latency = if total > 0 {
            total_latency / total as f64
        } else {
            0.0
        };

        TraceStatistics {
            total_traces: total,
            successful_traces: successful,
            failed_traces: total - successful,
            avg_steps,
            avg_latency_ms: avg_latency,
        }
    }
}

/// Training example in conversation format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    /// Conversation messages.
    pub messages: Vec<Message>,
}

/// Single message in conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role (user/assistant).
    pub role: String,
    /// Message content.
    pub content: String,
}

/// Statistics about traces.
#[derive(Debug, Clone)]
pub struct TraceStatistics {
    /// Total number of traces.
    pub total_traces: usize,
    /// Successful traces.
    pub successful_traces: usize,
    /// Failed traces.
    pub failed_traces: usize,
    /// Average steps per trace.
    pub avg_steps: f64,
    /// Average latency in ms.
    pub avg_latency_ms: f64,
}

/// Span for detailed timing.
pub struct TracingSpan {
    /// Span name.
    pub name: String,
    /// Start time.
    pub start: std::time::Instant,
    /// Child spans.
    pub children: Vec<TracingSpan>,
    /// Duration (set on end).
    pub duration_ms: Option<f64>,
}

impl TracingSpan {
    /// Create a new span.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start: std::time::Instant::now(),
            children: Vec::new(),
            duration_ms: None,
        }
    }

    /// End the span.
    pub fn end(&mut self) {
        self.duration_ms = Some(self.start.elapsed().as_secs_f64() * 1000.0);
    }

    /// Add child span.
    pub fn add_child(&mut self, span: TracingSpan) {
        self.children.push(span);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_tracer() {
        let temp_dir = TempDir::new().unwrap();
        let mut tracer = AgentTracer::new(temp_dir.path()).without_file_output();

        let context = AgentContext::new("Test task").with_max_steps(10);

        let trace_id = tracer.start_trace("Test task", &context);
        assert!(!trace_id.is_empty());

        let step = ThoughtActionObservation::new(0, "Test thought".to_string());
        tracer.log_step(&trace_id, &step);

        let result = AgentResult::success(
            "Test answer".to_string(),
            vec![step],
            100.0,
        );
        tracer.end_trace(&trace_id, &result);

        let trace = tracer.get_trace(&trace_id).unwrap();
        assert_eq!(trace.steps.len(), 1);
        assert!(trace.result.is_some());
    }

    #[test]
    fn test_training_export() {
        let temp_dir = TempDir::new().unwrap();
        let mut tracer = AgentTracer::new(temp_dir.path()).without_file_output();

        let context = AgentContext::new("Test").with_max_steps(5);
        let trace_id = tracer.start_trace("Test", &context);

        let step = ThoughtActionObservation::new(0, "Thought".to_string())
            .with_action("action".to_string(), HashMap::new())
            .with_observation("Result".to_string());

        tracer.log_step(&trace_id, &step);

        let result = AgentResult::success("Answer".to_string(), vec![step], 50.0);
        tracer.end_trace(&trace_id, &result);

        let training_data = tracer.export_training_data();
        assert_eq!(training_data.len(), 1);
        assert!(!training_data[0].messages.is_empty());
    }

    #[test]
    fn test_statistics() {
        let temp_dir = TempDir::new().unwrap();
        let mut tracer = AgentTracer::new(temp_dir.path()).without_file_output();

        // Add successful trace
        let context = AgentContext::new("Task 1").with_max_steps(5);
        let trace_id = tracer.start_trace("Task 1", &context);
        let result = AgentResult::success("OK".to_string(), vec![], 100.0);
        tracer.end_trace(&trace_id, &result);

        // Add failed trace
        let trace_id = tracer.start_trace("Task 2", &context);
        let result = AgentResult::failure("Error".to_string(), vec![], 50.0);
        tracer.end_trace(&trace_id, &result);

        let stats = tracer.statistics();
        assert_eq!(stats.total_traces, 2);
        assert_eq!(stats.successful_traces, 1);
        assert_eq!(stats.failed_traces, 1);
    }

    #[test]
    fn test_tracing_span() {
        let mut span = TracingSpan::new("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        span.end();

        assert!(span.duration_ms.unwrap() >= 10.0);
    }
}
