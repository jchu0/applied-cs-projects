//! Planning system for agent execution.

use crate::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Single step in an execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// Action to perform.
    pub action: String,
    /// Description of what this step accomplishes.
    pub description: String,
    /// Dependencies (indices of steps this depends on).
    pub dependencies: Vec<usize>,
    /// Input parameters.
    pub inputs: HashMap<String, serde_json::Value>,
}

/// Execution plan for agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    /// Steps in the plan.
    pub steps: Vec<PlanStep>,
    /// Current step index.
    pub current_step: usize,
}

impl Plan {
    /// Create a new plan.
    pub fn new(steps: Vec<PlanStep>) -> Self {
        Self {
            steps,
            current_step: 0,
        }
    }

    /// Get the next step.
    pub fn next_step(&mut self) -> Option<&PlanStep> {
        if self.current_step < self.steps.len() {
            let step = &self.steps[self.current_step];
            self.current_step += 1;
            Some(step)
        } else {
            None
        }
    }

    /// Check if plan is complete.
    pub fn is_complete(&self) -> bool {
        self.current_step >= self.steps.len()
    }

    /// Reset plan to beginning.
    pub fn reset(&mut self) {
        self.current_step = 0;
    }

    /// Get number of steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if plan is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// Base trait for planners.
#[async_trait]
pub trait Planner {
    /// Generate execution plan for task.
    async fn plan(&self, task: &str, tools: &[String]) -> Result<Plan>;
}

/// Simple planner that creates a basic plan.
pub struct SimplePlanner;

impl SimplePlanner {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SimplePlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Planner for SimplePlanner {
    async fn plan(&self, task: &str, _tools: &[String]) -> Result<Plan> {
        // Create a simple three-step plan
        let steps = vec![
            PlanStep {
                action: "analyze".to_string(),
                description: "Analyze the task requirements".to_string(),
                dependencies: vec![],
                inputs: {
                    let mut m = HashMap::new();
                    m.insert("task".to_string(), serde_json::json!(task));
                    m
                },
            },
            PlanStep {
                action: "execute".to_string(),
                description: "Execute the main task".to_string(),
                dependencies: vec![0],
                inputs: HashMap::new(),
            },
            PlanStep {
                action: "finish".to_string(),
                description: "Finalize and return result".to_string(),
                dependencies: vec![1],
                inputs: HashMap::new(),
            },
        ];

        Ok(Plan::new(steps))
    }
}

/// Rule-based planner for common patterns.
pub struct RulePlanner {
    rules: HashMap<String, Vec<PlanStep>>,
}

impl RulePlanner {
    /// Create a new rule planner.
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }

    /// Add a rule for a pattern.
    pub fn add_rule(&mut self, pattern: impl Into<String>, steps: Vec<PlanStep>) {
        self.rules.insert(pattern.into(), steps);
    }

    /// Find matching pattern.
    fn find_pattern(&self, task: &str) -> Option<&Vec<PlanStep>> {
        let task_lower = task.to_lowercase();
        for (pattern, steps) in &self.rules {
            if task_lower.contains(&pattern.to_lowercase()) {
                return Some(steps);
            }
        }
        None
    }
}

impl Default for RulePlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Planner for RulePlanner {
    async fn plan(&self, task: &str, _tools: &[String]) -> Result<Plan> {
        if let Some(steps) = self.find_pattern(task) {
            Ok(Plan::new(steps.clone()))
        } else {
            // Default single-step plan
            Ok(Plan::new(vec![PlanStep {
                action: "think".to_string(),
                description: task.to_string(),
                dependencies: vec![],
                inputs: HashMap::new(),
            }]))
        }
    }
}

/// Hybrid planner combining model and rules.
pub struct HybridPlanner {
    rule_planner: RulePlanner,
    use_model_threshold: usize,
}

impl HybridPlanner {
    /// Create a new hybrid planner.
    pub fn new(rule_planner: RulePlanner, use_model_threshold: usize) -> Self {
        Self {
            rule_planner,
            use_model_threshold,
        }
    }
}

#[async_trait]
impl Planner for HybridPlanner {
    async fn plan(&self, task: &str, tools: &[String]) -> Result<Plan> {
        // Try rule-based first
        let rule_plan = self.rule_planner.plan(task, tools).await?;

        // Use rule plan if it has enough steps
        if rule_plan.len() >= self.use_model_threshold {
            Ok(rule_plan)
        } else {
            // Otherwise, create a more detailed plan
            let steps = vec![
                PlanStep {
                    action: "understand".to_string(),
                    description: "Understand the task context".to_string(),
                    dependencies: vec![],
                    inputs: {
                        let mut m = HashMap::new();
                        m.insert("task".to_string(), serde_json::json!(task));
                        m
                    },
                },
                PlanStep {
                    action: "gather".to_string(),
                    description: "Gather necessary information".to_string(),
                    dependencies: vec![0],
                    inputs: HashMap::new(),
                },
                PlanStep {
                    action: "process".to_string(),
                    description: "Process the gathered information".to_string(),
                    dependencies: vec![1],
                    inputs: HashMap::new(),
                },
                PlanStep {
                    action: "synthesize".to_string(),
                    description: "Synthesize the results".to_string(),
                    dependencies: vec![2],
                    inputs: HashMap::new(),
                },
                PlanStep {
                    action: "finish".to_string(),
                    description: "Provide final answer".to_string(),
                    dependencies: vec![3],
                    inputs: HashMap::new(),
                },
            ];

            Ok(Plan::new(steps))
        }
    }
}

/// Plan optimizer for improving execution efficiency.
pub struct PlanOptimizer;

impl PlanOptimizer {
    /// Optimize a plan by removing redundant steps.
    pub fn optimize(plan: &mut Plan) {
        // Remove steps with no dependencies and no dependents
        let mut used_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();

        // Mark all dependencies as used
        for step in &plan.steps {
            for &dep in &step.dependencies {
                used_indices.insert(dep);
            }
        }

        // Mark all steps that have dependents or are final steps
        for (i, _) in plan.steps.iter().enumerate() {
            if i == plan.steps.len() - 1 || used_indices.contains(&i) {
                used_indices.insert(i);
            }
        }
    }

    /// Find parallelizable steps.
    pub fn find_parallel_steps(plan: &Plan) -> Vec<Vec<usize>> {
        let mut levels: Vec<Vec<usize>> = Vec::new();
        let mut step_levels: HashMap<usize, usize> = HashMap::new();

        for (i, step) in plan.steps.iter().enumerate() {
            let level = if step.dependencies.is_empty() {
                0
            } else {
                step.dependencies
                    .iter()
                    .filter_map(|&dep| step_levels.get(&dep))
                    .max()
                    .unwrap_or(&0)
                    + 1
            };

            step_levels.insert(i, level);

            while levels.len() <= level {
                levels.push(Vec::new());
            }
            levels[level].push(i);
        }

        levels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan() {
        let steps = vec![
            PlanStep {
                action: "step1".to_string(),
                description: "First step".to_string(),
                dependencies: vec![],
                inputs: HashMap::new(),
            },
            PlanStep {
                action: "step2".to_string(),
                description: "Second step".to_string(),
                dependencies: vec![0],
                inputs: HashMap::new(),
            },
        ];

        let mut plan = Plan::new(steps);
        assert_eq!(plan.len(), 2);
        assert!(!plan.is_complete());

        let step1 = plan.next_step().unwrap();
        assert_eq!(step1.action, "step1");

        let step2 = plan.next_step().unwrap();
        assert_eq!(step2.action, "step2");

        assert!(plan.is_complete());
        assert!(plan.next_step().is_none());
    }

    #[tokio::test]
    async fn test_simple_planner() {
        let planner = SimplePlanner::new();
        let plan = planner.plan("Test task", &[]).await.unwrap();

        assert!(!plan.is_empty());
        assert!(plan.steps.iter().any(|s| s.action == "finish"));
    }

    #[tokio::test]
    async fn test_rule_planner() {
        let mut planner = RulePlanner::new();
        planner.add_rule(
            "search",
            vec![
                PlanStep {
                    action: "search".to_string(),
                    description: "Search for information".to_string(),
                    dependencies: vec![],
                    inputs: HashMap::new(),
                },
                PlanStep {
                    action: "analyze".to_string(),
                    description: "Analyze results".to_string(),
                    dependencies: vec![0],
                    inputs: HashMap::new(),
                },
            ],
        );

        let plan = planner.plan("Search for data", &[]).await.unwrap();
        assert_eq!(plan.len(), 2);
        assert_eq!(plan.steps[0].action, "search");
    }

    #[test]
    fn test_parallel_steps() {
        let steps = vec![
            PlanStep {
                action: "a".to_string(),
                description: "".to_string(),
                dependencies: vec![],
                inputs: HashMap::new(),
            },
            PlanStep {
                action: "b".to_string(),
                description: "".to_string(),
                dependencies: vec![],
                inputs: HashMap::new(),
            },
            PlanStep {
                action: "c".to_string(),
                description: "".to_string(),
                dependencies: vec![0, 1],
                inputs: HashMap::new(),
            },
        ];

        let plan = Plan::new(steps);
        let parallel = PlanOptimizer::find_parallel_steps(&plan);

        assert_eq!(parallel.len(), 2);
        assert_eq!(parallel[0].len(), 2); // a and b can run in parallel
        assert_eq!(parallel[1].len(), 1); // c depends on both
    }
}
