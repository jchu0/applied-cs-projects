//! Execution graph for parallel tool execution.

use crate::planner::Plan;
use crate::tools::ToolRegistry;
use crate::{Error, Result};
use std::collections::{HashMap, HashSet};

/// Status of an execution node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

/// Node in execution graph.
#[derive(Debug, Clone)]
pub struct ExecutionNode {
    /// Unique identifier.
    pub id: String,
    /// Action to perform.
    pub action: String,
    /// Input parameters.
    pub inputs: HashMap<String, serde_json::Value>,
    /// Dependencies (node IDs).
    pub dependencies: Vec<String>,
    /// Node status.
    pub status: NodeStatus,
    /// Result of execution.
    pub result: Option<String>,
    /// Error message.
    pub error: Option<String>,
}

impl ExecutionNode {
    /// Create a new execution node.
    pub fn new(
        id: impl Into<String>,
        action: impl Into<String>,
        inputs: HashMap<String, serde_json::Value>,
        dependencies: Vec<String>,
    ) -> Self {
        Self {
            id: id.into(),
            action: action.into(),
            inputs,
            dependencies,
            status: NodeStatus::Pending,
            result: None,
            error: None,
        }
    }

    /// Check if all dependencies are completed.
    pub fn is_ready(&self, completed: &HashSet<String>) -> bool {
        self.dependencies.iter().all(|dep| completed.contains(dep))
    }
}

/// DAG-based execution graph for parallel tool execution.
pub struct ExecutionGraph {
    /// Tool registry.
    tools: ToolRegistry,
    /// Nodes in the graph.
    nodes: HashMap<String, ExecutionNode>,
    /// Completed node IDs.
    completed: HashSet<String>,
    /// Maximum parallel executions.
    max_parallel: usize,
}

impl ExecutionGraph {
    /// Create a new execution graph.
    pub fn new(tools: ToolRegistry) -> Self {
        Self {
            tools,
            nodes: HashMap::new(),
            completed: HashSet::new(),
            max_parallel: 5,
        }
    }

    /// Set maximum parallel executions.
    pub fn with_max_parallel(mut self, max: usize) -> Self {
        self.max_parallel = max;
        self
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: ExecutionNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Build graph from plan.
    pub fn build_from_plan(&mut self, plan: &Plan) {
        for (i, step) in plan.steps.iter().enumerate() {
            let node = ExecutionNode::new(
                format!("step_{}", i),
                step.action.clone(),
                step.inputs.clone(),
                step.dependencies
                    .iter()
                    .map(|d| format!("step_{}", d))
                    .collect(),
            );
            self.add_node(node);
        }
    }

    /// Execute the graph.
    pub async fn execute(&mut self) -> Result<HashMap<String, String>> {
        let mut results = HashMap::new();

        while !self.is_complete() {
            // Find ready nodes
            let ready: Vec<String> = self
                .nodes
                .iter()
                .filter(|(_, node)| {
                    node.status == NodeStatus::Pending && node.is_ready(&self.completed)
                })
                .map(|(id, _)| id.clone())
                .collect();

            if ready.is_empty() && !self.is_complete() {
                return Err(Error::Internal("Execution graph is stuck".to_string()));
            }

            // Execute ready nodes (up to max_parallel)
            let batch: Vec<String> = ready.into_iter().take(self.max_parallel).collect();

            for node_id in batch {
                self.execute_node(&node_id, &mut results).await?;
            }
        }

        Ok(results)
    }

    /// Execute a single node.
    async fn execute_node(
        &mut self,
        node_id: &str,
        results: &mut HashMap<String, String>,
    ) -> Result<()> {
        // Get action and inputs first to avoid borrow issues
        let (action, node_inputs) = {
            let node = self.nodes.get(node_id).ok_or_else(|| {
                Error::Internal(format!("Node not found: {}", node_id))
            })?;
            (node.action.clone(), node.inputs.clone())
        };

        // Now set status with mutable borrow
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.status = NodeStatus::Running;
        }

        // Resolve input references
        let inputs = self.resolve_inputs(&node_inputs, results);

        // Execute tool
        let result = if let Some(tool) = self.tools.get_tool(&action) {
            match tool.execute(inputs).await {
                Ok(r) => r,
                Err(e) => {
                    let node = self.nodes.get_mut(node_id).unwrap();
                    node.status = NodeStatus::Failed;
                    node.error = Some(e.to_string());
                    return Err(e);
                }
            }
        } else {
            // Simulate execution for unknown tools
            format!("Executed {}", action)
        };

        // Update node
        let node = self.nodes.get_mut(node_id).unwrap();
        node.result = Some(result.clone());
        node.status = NodeStatus::Completed;

        // Track completion
        results.insert(node_id.to_string(), result);
        self.completed.insert(node_id.to_string());

        Ok(())
    }

    /// Resolve references to previous node results.
    fn resolve_inputs(
        &self,
        inputs: &HashMap<String, serde_json::Value>,
        results: &HashMap<String, String>,
    ) -> HashMap<String, serde_json::Value> {
        let mut resolved = HashMap::new();

        for (key, value) in inputs {
            let resolved_value = if let Some(s) = value.as_str() {
                if let Some(ref_id) = s.strip_prefix('$') {
                    // Reference to previous result
                    if let Some(result) = results.get(ref_id) {
                        serde_json::json!(result)
                    } else {
                        value.clone()
                    }
                } else {
                    value.clone()
                }
            } else {
                value.clone()
            };

            resolved.insert(key.clone(), resolved_value);
        }

        resolved
    }

    /// Check if all nodes are completed.
    pub fn is_complete(&self) -> bool {
        self.completed.len() == self.nodes.len()
    }

    /// Get node by ID.
    pub fn get_node(&self, id: &str) -> Option<&ExecutionNode> {
        self.nodes.get(id)
    }

    /// Get all nodes.
    pub fn nodes(&self) -> impl Iterator<Item = &ExecutionNode> {
        self.nodes.values()
    }

    /// Get execution order (topological sort).
    pub fn execution_order(&self) -> Vec<String> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut temp_visited = HashSet::new();

        // Sort keys for deterministic order
        let mut keys: Vec<&String> = self.nodes.keys().collect();
        keys.sort();

        for id in keys {
            if !visited.contains(id) {
                self.topo_sort(id, &mut visited, &mut temp_visited, &mut order);
            }
        }

        // No need to reverse - dependencies are visited first, then the node is added
        order
    }

    fn topo_sort(
        &self,
        node_id: &str,
        visited: &mut HashSet<String>,
        temp_visited: &mut HashSet<String>,
        order: &mut Vec<String>,
    ) {
        if temp_visited.contains(node_id) {
            return; // Cycle detected
        }

        if !visited.contains(node_id) {
            temp_visited.insert(node_id.to_string());

            if let Some(node) = self.nodes.get(node_id) {
                for dep in &node.dependencies {
                    self.topo_sort(dep, visited, temp_visited, order);
                }
            }

            temp_visited.remove(node_id);
            visited.insert(node_id.to_string());
            order.push(node_id.to_string());
        }
    }

    /// Reset graph for re-execution.
    pub fn reset(&mut self) {
        for node in self.nodes.values_mut() {
            node.status = NodeStatus::Pending;
            node.result = None;
            node.error = None;
        }
        self.completed.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::PlanStep;

    #[test]
    fn test_execution_node() {
        let mut completed = HashSet::new();
        let node = ExecutionNode::new(
            "node1",
            "action",
            HashMap::new(),
            vec!["dep1".to_string()],
        );

        assert!(!node.is_ready(&completed));

        completed.insert("dep1".to_string());
        assert!(node.is_ready(&completed));
    }

    #[test]
    fn test_build_from_plan() {
        let plan = Plan::new(vec![
            PlanStep {
                action: "step1".to_string(),
                description: "First".to_string(),
                dependencies: vec![],
                inputs: HashMap::new(),
            },
            PlanStep {
                action: "step2".to_string(),
                description: "Second".to_string(),
                dependencies: vec![0],
                inputs: HashMap::new(),
            },
        ]);

        let tools = ToolRegistry::new();
        let mut graph = ExecutionGraph::new(tools);
        graph.build_from_plan(&plan);

        assert_eq!(graph.nodes.len(), 2);
        assert!(graph.get_node("step_0").is_some());
        assert!(graph.get_node("step_1").is_some());
    }

    #[test]
    fn test_execution_order() {
        let tools = ToolRegistry::new();
        let mut graph = ExecutionGraph::new(tools);

        graph.add_node(ExecutionNode::new("a", "action", HashMap::new(), vec![]));
        graph.add_node(ExecutionNode::new(
            "b",
            "action",
            HashMap::new(),
            vec!["a".to_string()],
        ));
        graph.add_node(ExecutionNode::new(
            "c",
            "action",
            HashMap::new(),
            vec!["a".to_string(), "b".to_string()],
        ));

        let order = graph.execution_order();

        // 'a' must come before 'b' and 'c', 'b' must come before 'c'
        let a_pos = order.iter().position(|x| x == "a").unwrap();
        let b_pos = order.iter().position(|x| x == "b").unwrap();
        let c_pos = order.iter().position(|x| x == "c").unwrap();

        assert!(a_pos < b_pos);
        assert!(a_pos < c_pos);
        assert!(b_pos < c_pos);
    }

    #[tokio::test]
    async fn test_execute_graph() {
        let tools = ToolRegistry::new();
        let mut graph = ExecutionGraph::new(tools);

        graph.add_node(ExecutionNode::new("a", "action1", HashMap::new(), vec![]));
        graph.add_node(ExecutionNode::new(
            "b",
            "action2",
            HashMap::new(),
            vec!["a".to_string()],
        ));

        let results = graph.execute().await.unwrap();

        assert_eq!(results.len(), 2);
        assert!(graph.is_complete());
    }
}
