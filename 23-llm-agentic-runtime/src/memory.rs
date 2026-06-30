//! Memory system for agent context.

use crate::agent::ThoughtActionObservation;
use std::collections::VecDeque;

/// Memory system for agent context.
pub struct AgentMemory {
    /// Short-term memory (recent steps).
    short_term: VecDeque<ThoughtActionObservation>,
    /// Maximum short-term capacity.
    short_term_limit: usize,
    /// Long-term memory storage (simplified as Vec).
    long_term: Vec<MemoryEntry>,
}

/// Entry in long-term memory.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    /// Step information.
    pub step: ThoughtActionObservation,
    /// Importance score.
    pub importance: f32,
    /// Embedding (placeholder).
    pub embedding: Option<Vec<f32>>,
}

impl AgentMemory {
    /// Create new agent memory.
    pub fn new(short_term_limit: usize) -> Self {
        Self {
            short_term: VecDeque::with_capacity(short_term_limit),
            short_term_limit,
            long_term: Vec::new(),
        }
    }

    /// Add a step to memory.
    pub fn add_step(&mut self, step: ThoughtActionObservation) {
        self.short_term.push_back(step.clone());

        // Move to long-term if exceeding limit
        if self.short_term.len() > self.short_term_limit {
            if let Some(overflow) = self.short_term.pop_front() {
                self.store_long_term(overflow);
            }
        }
    }

    /// Store step in long-term memory.
    fn store_long_term(&mut self, step: ThoughtActionObservation) {
        // Calculate importance (simplified)
        let importance = self.calculate_importance(&step);

        let entry = MemoryEntry {
            step,
            importance,
            embedding: None, // Would be computed by embedding model
        };

        self.long_term.push(entry);
    }

    /// Calculate importance score for a step.
    fn calculate_importance(&self, step: &ThoughtActionObservation) -> f32 {
        let mut score: f32 = 0.5; // Base score

        // Higher importance for errors
        if step.error.is_some() {
            score += 0.3;
        }

        // Higher importance for completion
        if step.action.as_deref() == Some("finish") {
            score += 0.2;
        }

        // Lower importance for routine steps
        if step.action.as_deref() == Some("think") {
            score -= 0.1;
        }

        score.clamp(0.0, 1.0)
    }

    /// Get recent steps as context string.
    pub fn get_recent_context(&self) -> String {
        let mut parts = Vec::new();

        for step in self.short_term.iter().rev().take(5).rev() {
            let mut part = format!("Step {}:\n", step.step);
            part.push_str(&format!("  Thought: {}\n", step.thought));

            if let Some(action) = &step.action {
                part.push_str(&format!("  Action: {}\n", action));
            }

            if let Some(obs) = &step.observation {
                let truncated = if obs.len() > 200 {
                    format!("{}...", &obs[..200])
                } else {
                    obs.clone()
                };
                part.push_str(&format!("  Observation: {}\n", truncated));
            }

            parts.push(part);
        }

        parts.join("\n")
    }

    /// Search long-term memory (simplified).
    pub fn search_long_term(&self, query: &str, k: usize) -> Vec<&MemoryEntry> {
        // Simplified search based on keyword matching
        // In production, would use vector similarity
        let query_lower = query.to_lowercase();

        let mut scored: Vec<(&MemoryEntry, f32)> = self
            .long_term
            .iter()
            .map(|entry| {
                let text = format!(
                    "{} {}",
                    entry.step.thought,
                    entry.step.observation.as_deref().unwrap_or("")
                )
                .to_lowercase();

                let score = if text.contains(&query_lower) {
                    entry.importance + 0.5
                } else {
                    entry.importance
                };

                (entry, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().take(k).map(|(e, _)| e).collect()
    }

    /// Get all short-term steps.
    pub fn short_term_steps(&self) -> impl Iterator<Item = &ThoughtActionObservation> {
        self.short_term.iter()
    }

    /// Get all long-term entries.
    pub fn long_term_entries(&self) -> impl Iterator<Item = &MemoryEntry> {
        self.long_term.iter()
    }

    /// Get short-term memory size.
    pub fn short_term_size(&self) -> usize {
        self.short_term.len()
    }

    /// Get long-term memory size.
    pub fn long_term_size(&self) -> usize {
        self.long_term.len()
    }

    /// Clear all memory.
    pub fn clear(&mut self) {
        self.short_term.clear();
        self.long_term.clear();
    }

    /// Clear only short-term memory.
    pub fn clear_short_term(&mut self) {
        self.short_term.clear();
    }

    /// Summarize memory state.
    pub fn summarize(&self) -> MemorySummary {
        let total_steps = self.short_term.len() + self.long_term.len();
        let error_count = self
            .short_term
            .iter()
            .chain(self.long_term.iter().map(|e| &e.step))
            .filter(|s| s.error.is_some())
            .count();

        let avg_latency = if total_steps > 0 {
            let total_latency: f64 = self
                .short_term
                .iter()
                .chain(self.long_term.iter().map(|e| &e.step))
                .map(|s| s.latency_ms)
                .sum();
            total_latency / total_steps as f64
        } else {
            0.0
        };

        MemorySummary {
            short_term_size: self.short_term.len(),
            long_term_size: self.long_term.len(),
            total_steps,
            error_count,
            avg_latency_ms: avg_latency,
        }
    }
}

/// Summary of memory state.
#[derive(Debug, Clone)]
pub struct MemorySummary {
    /// Short-term memory size.
    pub short_term_size: usize,
    /// Long-term memory size.
    pub long_term_size: usize,
    /// Total steps recorded.
    pub total_steps: usize,
    /// Number of errors.
    pub error_count: usize,
    /// Average latency.
    pub avg_latency_ms: f64,
}

/// Working memory for current task context.
pub struct WorkingMemory {
    /// Current task.
    pub task: String,
    /// Current goal.
    pub goal: Option<String>,
    /// Scratchpad for intermediate results.
    pub scratchpad: Vec<String>,
    /// Variables.
    pub variables: std::collections::HashMap<String, serde_json::Value>,
}

impl WorkingMemory {
    /// Create new working memory.
    pub fn new(task: impl Into<String>) -> Self {
        Self {
            task: task.into(),
            goal: None,
            scratchpad: Vec::new(),
            variables: std::collections::HashMap::new(),
        }
    }

    /// Set current goal.
    pub fn set_goal(&mut self, goal: impl Into<String>) {
        self.goal = Some(goal.into());
    }

    /// Add to scratchpad.
    pub fn note(&mut self, note: impl Into<String>) {
        self.scratchpad.push(note.into());
    }

    /// Set variable.
    pub fn set_var(&mut self, name: impl Into<String>, value: serde_json::Value) {
        self.variables.insert(name.into(), value);
    }

    /// Get variable.
    pub fn get_var(&self, name: &str) -> Option<&serde_json::Value> {
        self.variables.get(name)
    }

    /// Clear scratchpad.
    pub fn clear_scratchpad(&mut self) {
        self.scratchpad.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_memory() {
        let mut memory = AgentMemory::new(5);

        // Add steps
        for i in 0..7 {
            let step = ThoughtActionObservation::new(i, format!("Thought {}", i));
            memory.add_step(step);
        }

        // First 2 should be in long-term
        assert_eq!(memory.short_term_size(), 5);
        assert_eq!(memory.long_term_size(), 2);
    }

    #[test]
    fn test_memory_context() {
        let mut memory = AgentMemory::new(10);

        let step = ThoughtActionObservation::new(0, "Test thought".to_string())
            .with_action("test".to_string(), std::collections::HashMap::new())
            .with_observation("Test result".to_string());

        memory.add_step(step);

        let context = memory.get_recent_context();
        assert!(context.contains("Test thought"));
        assert!(context.contains("Test result"));
    }

    #[test]
    fn test_memory_search() {
        let mut memory = AgentMemory::new(2);

        for i in 0..5 {
            let step = ThoughtActionObservation::new(i, format!("Search term {}", i))
                .with_observation(format!("Result for term {}", i));
            memory.add_step(step);
        }

        let results = memory.search_long_term("term 0", 2);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_working_memory() {
        let mut wm = WorkingMemory::new("Test task");
        wm.set_goal("Complete the test");
        wm.note("Important note");
        wm.set_var("count", serde_json::json!(42));

        assert_eq!(wm.task, "Test task");
        assert_eq!(wm.goal, Some("Complete the test".to_string()));
        assert_eq!(wm.scratchpad.len(), 1);
        assert_eq!(wm.get_var("count"), Some(&serde_json::json!(42)));
    }

    #[test]
    fn test_memory_summary() {
        let mut memory = AgentMemory::new(5);

        for i in 0..3 {
            let mut step = ThoughtActionObservation::new(i, format!("Thought {}", i));
            step.latency_ms = 100.0;
            if i == 1 {
                step.error = Some("Test error".to_string());
            }
            memory.add_step(step);
        }

        let summary = memory.summarize();
        assert_eq!(summary.total_steps, 3);
        assert_eq!(summary.error_count, 1);
        assert!((summary.avg_latency_ms - 100.0).abs() < 0.1);
    }
}
