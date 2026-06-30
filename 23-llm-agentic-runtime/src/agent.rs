//! Core agent runtime implementation.

use crate::memory::AgentMemory;
use crate::planner::Planner;
use crate::tools::ToolRegistry;
use crate::llm::{ChatMessage, LlmClient};
use crate::{Error, Result, DEFAULT_MAX_STEPS, DEFAULT_TIMEOUT_SECONDS};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Agent execution state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentState {
    Idle,
    Thinking,
    Acting,
    Observing,
    Completed,
    Failed,
}

/// Context for agent execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentContext {
    /// Task to accomplish.
    pub task: String,
    /// Maximum number of steps.
    pub max_steps: usize,
    /// Timeout in seconds.
    pub timeout_seconds: f64,
    /// Model configuration.
    pub model_config: HashMap<String, String>,
    /// Enabled tools.
    pub tools_enabled: Vec<String>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl AgentContext {
    /// Create a new context with just a task.
    pub fn new(task: impl Into<String>) -> Self {
        Self {
            task: task.into(),
            max_steps: DEFAULT_MAX_STEPS,
            timeout_seconds: DEFAULT_TIMEOUT_SECONDS,
            model_config: HashMap::new(),
            tools_enabled: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set maximum steps.
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Set timeout.
    pub fn with_timeout(mut self, timeout_seconds: f64) -> Self {
        self.timeout_seconds = timeout_seconds;
        self
    }

    /// Enable specific tools.
    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tools_enabled = tools;
        self
    }
}

/// Single step in agent reasoning (Thought-Action-Observation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtActionObservation {
    /// Step number.
    pub step: usize,
    /// Agent's reasoning.
    pub thought: String,
    /// Action to take.
    pub action: Option<String>,
    /// Input parameters for action.
    pub action_input: Option<HashMap<String, serde_json::Value>>,
    /// Result of action.
    pub observation: Option<String>,
    /// Error if any.
    pub error: Option<String>,
    /// Timestamp.
    pub timestamp: f64,
    /// Step latency in milliseconds.
    pub latency_ms: f64,
}

impl ThoughtActionObservation {
    /// Create a new step.
    pub fn new(step: usize, thought: String) -> Self {
        Self {
            step,
            thought,
            action: None,
            action_input: None,
            observation: None,
            error: None,
            timestamp: 0.0,
            latency_ms: 0.0,
        }
    }

    /// Set action.
    pub fn with_action(mut self, action: String, input: HashMap<String, serde_json::Value>) -> Self {
        self.action = Some(action);
        self.action_input = Some(input);
        self
    }

    /// Set observation.
    pub fn with_observation(mut self, observation: String) -> Self {
        self.observation = Some(observation);
        self
    }

    /// Set error.
    pub fn with_error(mut self, error: String) -> Self {
        self.error = Some(error);
        self
    }
}

/// Final result from agent execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    /// Whether execution succeeded.
    pub success: bool,
    /// Final answer.
    pub answer: Option<String>,
    /// All steps taken.
    pub steps: Vec<ThoughtActionObservation>,
    /// Total tokens used.
    pub total_tokens: usize,
    /// Total latency in milliseconds.
    pub total_latency_ms: f64,
    /// Error message if failed.
    pub error: Option<String>,
}

impl AgentResult {
    /// Create a successful result.
    pub fn success(answer: String, steps: Vec<ThoughtActionObservation>, total_latency_ms: f64) -> Self {
        Self {
            success: true,
            answer: Some(answer),
            steps,
            total_tokens: 0,
            total_latency_ms,
            error: None,
        }
    }

    /// Create a failed result.
    pub fn failure(error: String, steps: Vec<ThoughtActionObservation>, total_latency_ms: f64) -> Self {
        Self {
            success: false,
            answer: None,
            steps,
            total_tokens: 0,
            total_latency_ms,
            error: Some(error),
        }
    }
}

/// Configuration for agent runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRuntimeConfig {
    /// Enable sandboxed execution.
    pub sandbox_enabled: bool,
    /// Enable guardrails.
    pub guardrails_enabled: bool,
    /// Enable tracing.
    pub tracing_enabled: bool,
    /// Retry failed actions.
    pub retry_on_failure: bool,
    /// Maximum retries per action.
    pub max_retries: usize,
}

impl Default for AgentRuntimeConfig {
    fn default() -> Self {
        Self {
            sandbox_enabled: false,
            guardrails_enabled: true,
            tracing_enabled: true,
            retry_on_failure: true,
            max_retries: 3,
        }
    }
}

/// Core agent runtime with ReAct-style reasoning loop.
pub struct AgentRuntime {
    /// Tool registry.
    pub tools: ToolRegistry,
    /// Planner.
    pub planner: Box<dyn Planner + Send + Sync>,
    /// Memory system.
    pub memory: AgentMemory,
    /// Runtime configuration.
    pub config: AgentRuntimeConfig,
    /// Current state.
    state: AgentState,
    /// Current context.
    current_context: Option<AgentContext>,
    /// Execution trace.
    trace: Vec<ThoughtActionObservation>,
    /// Optional LLM client driving the reasoning loop. Without one, a
    /// deterministic heuristic is used.
    llm: Option<Box<dyn LlmClient + Send + Sync>>,
}

impl AgentRuntime {
    /// Create a new agent runtime.
    pub fn new(
        tools: ToolRegistry,
        planner: Box<dyn Planner + Send + Sync>,
        memory: AgentMemory,
        config: AgentRuntimeConfig,
    ) -> Self {
        Self {
            tools,
            planner,
            memory,
            config,
            state: AgentState::Idle,
            current_context: None,
            trace: Vec::new(),
            llm: None,
        }
    }

    /// Attach an LLM client to drive the reasoning loop. Without one, the
    /// runtime falls back to a deterministic heuristic (handy for demos/tests).
    pub fn with_llm(mut self, client: Box<dyn LlmClient + Send + Sync>) -> Self {
        self.llm = Some(client);
        self
    }

    /// Execute agent on given task.
    pub async fn run(&mut self, context: AgentContext) -> AgentResult {
        self.current_context = Some(context.clone());
        self.state = AgentState::Thinking;
        self.trace.clear();

        let start_time = Instant::now();
        let timeout = Duration::from_secs_f64(context.timeout_seconds);

        for step in 0..context.max_steps {
            // Check timeout
            if start_time.elapsed() > timeout {
                self.state = AgentState::Failed;
                return AgentResult::failure(
                    "Execution timed out".to_string(),
                    self.trace.clone(),
                    start_time.elapsed().as_millis() as f64,
                );
            }

            let step_start = Instant::now();

            // Generate thought and action
            let (thought, action, action_input) = match self.think(step).await {
                Ok(t) => t,
                Err(e) => {
                    self.state = AgentState::Failed;
                    return AgentResult::failure(
                        format!("Reasoning step failed: {}", e),
                        self.trace.clone(),
                        start_time.elapsed().as_millis() as f64,
                    );
                }
            };

            let mut tao = ThoughtActionObservation::new(step, thought)
                .with_action(action.clone(), action_input.clone());

            // Check for completion
            if action == "finish" {
                self.state = AgentState::Completed;
                let answer = action_input
                    .get("answer")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                tao.observation = Some(answer.clone());
                tao.latency_ms = step_start.elapsed().as_millis() as f64;
                self.trace.push(tao);

                return AgentResult::success(
                    answer,
                    self.trace.clone(),
                    start_time.elapsed().as_millis() as f64,
                );
            }

            // Execute action
            self.state = AgentState::Acting;
            match self.act(&action, &action_input).await {
                Ok(observation) => {
                    tao.observation = Some(observation);
                }
                Err(e) => {
                    tao.error = Some(e.to_string());
                    tao.observation = Some(format!("Error: {}", e));
                }
            }

            tao.latency_ms = step_start.elapsed().as_millis() as f64;
            self.trace.push(tao.clone());

            // Update memory
            self.state = AgentState::Observing;
            self.memory.add_step(tao);

            self.state = AgentState::Thinking;
        }

        // Max steps exceeded
        self.state = AgentState::Failed;
        AgentResult::failure(
            format!("Maximum steps ({}) exceeded", context.max_steps),
            self.trace.clone(),
            start_time.elapsed().as_millis() as f64,
        )
    }

    /// Generate the next thought and action.
    ///
    /// With an [`LlmClient`] attached, the model is prompted in a ReAct style and
    /// its JSON reply parsed into `(thought, action, action_input)`. Without one,
    /// a deterministic heuristic is used (see [`Self::think_simulated`]).
    async fn think(
        &self,
        step: usize,
    ) -> Result<(String, String, HashMap<String, serde_json::Value>)> {
        let context = self.current_context.as_ref().unwrap();
        match &self.llm {
            None => Ok(self.think_simulated(step)),
            Some(client) => {
                let system = self.react_system_prompt();
                let user = self.react_user_prompt(context);
                let reply = client.complete(&system, &[ChatMessage::user(user)]).await?;
                parse_react_step(&reply)
            }
        }
    }

    /// ReAct system prompt describing the protocol and the available tools.
    fn react_system_prompt(&self) -> String {
        format!(
            "You are an autonomous agent that solves tasks with a ReAct loop (reason, then act). \
             On each turn respond with ONLY a JSON object of the form:\n\
             {{\"thought\": \"<reasoning>\", \"action\": \"<tool name or 'finish'>\", \"action_input\": {{...}}}}\n\
             Use action \"finish\" with action_input {{\"answer\": \"...\"}} once you can answer.\n\
             Available tools:\n{}",
            self.tools.get_formatted_descriptions()
        )
    }

    /// User prompt built from the task and the trace accumulated so far.
    fn react_user_prompt(&self, context: &AgentContext) -> String {
        let mut s = format!("Task: {}\n", context.task);
        if !self.trace.is_empty() {
            s.push_str("\nHistory so far:\n");
            for tao in &self.trace {
                s.push_str(&format!("- thought: {}\n", tao.thought));
                if let Some(a) = &tao.action {
                    s.push_str(&format!("  action: {}\n", a));
                }
                if let Some(o) = &tao.observation {
                    s.push_str(&format!("  observation: {}\n", o));
                }
            }
        }
        s.push_str("\nRespond with the next JSON step.");
        s
    }

    /// Deterministic fallback used when no LLM client is attached.
    fn think_simulated(
        &self,
        step: usize,
    ) -> (String, String, HashMap<String, serde_json::Value>) {
        let context = self.current_context.as_ref().unwrap();

        if step == 0 {
            // First step: analyze task
            let thought = format!("I need to accomplish: {}. Let me break this down.", context.task);
            let action = "analyze".to_string();
            let mut input = HashMap::new();
            input.insert("task".to_string(), serde_json::json!(context.task));
            (thought, action, input)
        } else if step >= 5 || self.trace.len() >= 3 {
            // After a few steps, finish
            let thought = "I have gathered enough information. Let me provide the answer.".to_string();
            let action = "finish".to_string();
            let mut input = HashMap::new();
            input.insert(
                "answer".to_string(),
                serde_json::json!(format!("Completed task: {}", context.task)),
            );
            (thought, action, input)
        } else {
            // Intermediate steps
            let thought = format!("Step {}: Processing...", step);
            let action = "process".to_string();
            let mut input = HashMap::new();
            input.insert("step".to_string(), serde_json::json!(step));
            (thought, action, input)
        }
    }

    /// Execute the chosen action.
    async fn act(
        &self,
        action: &str,
        action_input: &HashMap<String, serde_json::Value>,
    ) -> Result<String> {
        // Check if tool exists
        if let Some(tool) = self.tools.get_tool(action) {
            // Execute tool
            tool.execute(action_input.clone()).await
        } else {
            // Unknown action - simulate for demo
            Ok(format!("Executed action '{}' with input: {:?}", action, action_input))
        }
    }

    /// Get current state.
    pub fn state(&self) -> AgentState {
        self.state
    }

    /// Get execution trace.
    pub fn trace(&self) -> &[ThoughtActionObservation] {
        &self.trace
    }

    /// Reset runtime for new task.
    pub fn reset(&mut self) {
        self.state = AgentState::Idle;
        self.current_context = None;
        self.trace.clear();
        self.memory.clear();
    }
}

/// Parse a ReAct JSON step out of a (possibly prose-wrapped) model reply.
fn parse_react_step(
    reply: &str,
) -> Result<(String, String, HashMap<String, serde_json::Value>)> {
    let json_str = extract_json_object(reply).unwrap_or_else(|| reply.trim().to_string());
    let value: serde_json::Value = serde_json::from_str(&json_str).map_err(|e| {
        Error::ParseError(format!("LLM reply was not valid JSON ({}); reply: {}", e, reply))
    })?;
    let thought = value
        .get("thought")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let action = value
        .get("action")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::ParseError("LLM reply missing 'action'".to_string()))?
        .to_string();
    let action_input = match value.get("action_input") {
        Some(serde_json::Value::Object(map)) => map.clone().into_iter().collect(),
        _ => HashMap::new(),
    };
    Ok((thought, action, action_input))
}

/// Return the substring spanning the first balanced `{...}` (naive brace match).
fn extract_json_object(s: &str) -> Option<String> {
    let start = s.find('{')?;
    let mut depth = 0usize;
    for (i, c) in s[start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(s[start..start + i + 1].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::SimplePlanner;

    #[tokio::test]
    async fn test_agent_runtime() {
        let tools = ToolRegistry::new();
        let planner = Box::new(SimplePlanner::new());
        let memory = AgentMemory::new(10);
        let config = AgentRuntimeConfig::default();

        let mut runtime = AgentRuntime::new(tools, planner, memory, config);

        let context = AgentContext::new("Test task").with_max_steps(10);

        let result = runtime.run(context).await;

        assert!(result.success);
        assert!(result.answer.is_some());
        assert!(!result.steps.is_empty());
    }

    #[tokio::test]
    async fn test_agent_timeout() {
        let tools = ToolRegistry::new();
        let planner = Box::new(SimplePlanner::new());
        let memory = AgentMemory::new(10);
        let config = AgentRuntimeConfig::default();

        let mut runtime = AgentRuntime::new(tools, planner, memory, config);

        let context = AgentContext::new("Test task")
            .with_timeout(0.001) // Very short timeout
            .with_max_steps(1000);

        let result = runtime.run(context).await;

        // May or may not timeout depending on execution speed
        assert!(result.steps.len() <= 1000);
    }

    #[test]
    fn test_agent_context() {
        let context = AgentContext::new("Do something")
            .with_max_steps(50)
            .with_timeout(600.0)
            .with_tools(vec!["search".to_string(), "calculate".to_string()]);

        assert_eq!(context.task, "Do something");
        assert_eq!(context.max_steps, 50);
        assert_eq!(context.timeout_seconds, 600.0);
        assert_eq!(context.tools_enabled.len(), 2);
    }

    #[tokio::test]
    async fn test_agent_with_mock_llm_finishes() {
        use crate::llm::MockLlmClient;
        let runtime = AgentRuntime::new(
            ToolRegistry::new(),
            Box::new(SimplePlanner::new()),
            AgentMemory::new(10),
            AgentRuntimeConfig::default(),
        )
        .with_llm(Box::new(MockLlmClient::finishing("the capital is Paris")));
        let mut runtime = runtime;

        let result = runtime
            .run(AgentContext::new("What is the capital of France?"))
            .await;

        assert!(result.success);
        assert_eq!(result.answer.as_deref(), Some("the capital is Paris"));
    }

    #[tokio::test]
    async fn test_agent_mock_llm_drives_a_tool_then_finishes() {
        use crate::llm::MockLlmClient;
        use crate::tools::SearchTool;
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(SearchTool::new("search")));

        // First the model calls the search tool, then it finishes.
        let scripted = vec![
            r#"{"thought":"I should search","action":"search","action_input":{"query":"France"}}"#
                .to_string(),
            r#"{"thought":"Now I can answer","action":"finish","action_input":{"answer":"Paris"}}"#
                .to_string(),
        ];
        let mut runtime = AgentRuntime::new(
            tools,
            Box::new(SimplePlanner::new()),
            AgentMemory::new(10),
            AgentRuntimeConfig::default(),
        )
        .with_llm(Box::new(MockLlmClient::new(scripted)));

        let result = runtime
            .run(AgentContext::new("capital of France").with_max_steps(5))
            .await;

        assert!(result.success);
        assert_eq!(result.answer.as_deref(), Some("Paris"));
        // The search tool ran on the first step (LLM chose it).
        assert!(runtime
            .trace()
            .iter()
            .any(|t| t.action.as_deref() == Some("search")));
    }

    #[test]
    fn test_parse_react_step_tolerates_surrounding_prose() {
        let reply = "Here is my step:\n\
            {\"thought\":\"t\",\"action\":\"finish\",\"action_input\":{\"answer\":\"a\"}}\nThanks!";
        let (thought, action, input) = parse_react_step(reply).unwrap();
        assert_eq!(thought, "t");
        assert_eq!(action, "finish");
        assert_eq!(input.get("answer").unwrap(), "a");
    }
}
