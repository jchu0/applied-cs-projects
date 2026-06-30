//! Tool system for agent actions.

use crate::{Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// JSON Schema for tool parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    /// Schema type (usually "object").
    #[serde(rename = "type")]
    pub schema_type: String,
    /// Parameter properties.
    pub properties: HashMap<String, ParameterSchema>,
    /// Required parameters.
    pub required: Vec<String>,
}

/// Schema for a single parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSchema {
    /// Parameter type.
    #[serde(rename = "type")]
    pub param_type: String,
    /// Parameter description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Tool metadata for model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMetadata {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// Parameter schema.
    pub parameters: ToolSchema,
}

/// Base trait for all tools.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Get tool name.
    fn name(&self) -> &str;

    /// Get tool description.
    fn description(&self) -> &str;

    /// Get parameter schema.
    fn parameters(&self) -> ToolSchema;

    /// Execute the tool.
    async fn execute(&self, inputs: HashMap<String, serde_json::Value>) -> Result<String>;

    /// Get tool metadata.
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters(),
        }
    }
}

/// Function-based tool.
pub struct FunctionTool {
    name: String,
    description: String,
    parameters: ToolSchema,
    handler: Box<dyn Fn(HashMap<String, serde_json::Value>) -> Result<String> + Send + Sync>,
}

impl FunctionTool {
    /// Create a new function tool.
    pub fn new<F>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: ToolSchema,
        handler: F,
    ) -> Self
    where
        F: Fn(HashMap<String, serde_json::Value>) -> Result<String> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            handler: Box::new(handler),
        }
    }
}

#[async_trait]
impl Tool for FunctionTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> ToolSchema {
        self.parameters.clone()
    }

    async fn execute(&self, inputs: HashMap<String, serde_json::Value>) -> Result<String> {
        (self.handler)(inputs)
    }
}

/// Calculator tool for mathematical operations.
pub struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Perform mathematical calculations"
    }

    fn parameters(&self) -> ToolSchema {
        let mut properties = HashMap::new();
        properties.insert(
            "expression".to_string(),
            ParameterSchema {
                param_type: "string".to_string(),
                description: Some("Mathematical expression to evaluate".to_string()),
            },
        );

        ToolSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["expression".to_string()],
        }
    }

    async fn execute(&self, inputs: HashMap<String, serde_json::Value>) -> Result<String> {
        let expression = inputs
            .get("expression")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::ValidationFailed("Missing expression".to_string()))?;

        // Simple evaluation (in production, use a proper math parser)
        let result = self.evaluate_simple(expression)?;
        Ok(format!("{}", result))
    }
}

impl CalculatorTool {
    fn evaluate_simple(&self, expr: &str) -> Result<f64> {
        // Very simple evaluation for demo purposes
        let expr = expr.trim();

        // Try to parse as a simple number
        if let Ok(num) = expr.parse::<f64>() {
            return Ok(num);
        }

        // Try simple operations
        for op in ['+', '-', '*', '/'] {
            if let Some(pos) = expr.rfind(op) {
                if pos > 0 {
                    let left = self.evaluate_simple(&expr[..pos])?;
                    let right = self.evaluate_simple(&expr[pos + 1..])?;
                    return match op {
                        '+' => Ok(left + right),
                        '-' => Ok(left - right),
                        '*' => Ok(left * right),
                        '/' => {
                            if right == 0.0 {
                                Err(Error::ToolExecutionFailed("Division by zero".to_string()))
                            } else {
                                Ok(left / right)
                            }
                        }
                        _ => unreachable!(),
                    };
                }
            }
        }

        Err(Error::ToolExecutionFailed(format!(
            "Cannot evaluate: {}",
            expr
        )))
    }
}

/// Search tool stub.
pub struct SearchTool {
    name: String,
}

impl SearchTool {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[async_trait]
impl Tool for SearchTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        "Search for information"
    }

    fn parameters(&self) -> ToolSchema {
        let mut properties = HashMap::new();
        properties.insert(
            "query".to_string(),
            ParameterSchema {
                param_type: "string".to_string(),
                description: Some("Search query".to_string()),
            },
        );

        ToolSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["query".to_string()],
        }
    }

    async fn execute(&self, inputs: HashMap<String, serde_json::Value>) -> Result<String> {
        let query = inputs
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::ValidationFailed("Missing query".to_string()))?;

        // No real search backend is wired in. Return an explicit placeholder
        // rather than fabricated results, so callers are not misled.
        Ok(format!(
            "[SearchTool] No search backend is configured; cannot retrieve results for '{}'. \
             Plug in a real implementation (e.g. an HTTP search API) to enable retrieval.",
            query
        ))
    }
}

/// Registry for managing tools.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool.
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let name = tool.name().to_string();
        self.tools.insert(name, tool);
    }

    /// Get a tool by name.
    pub fn get_tool(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    /// Get all tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Get OpenAI-compatible tool schemas.
    pub fn get_tool_schemas(&self) -> Vec<serde_json::Value> {
        self.tools
            .values()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name(),
                        "description": tool.description(),
                        "parameters": tool.parameters(),
                    }
                })
            })
            .collect()
    }

    /// Get formatted descriptions for prompt.
    pub fn get_formatted_descriptions(&self) -> String {
        self.tools
            .values()
            .map(|tool| {
                let params: Vec<String> = tool
                    .parameters()
                    .properties
                    .keys()
                    .map(|k| k.to_string())
                    .collect();
                format!("- {}({}): {}", tool.name(), params.join(", "), tool.description())
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Get number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_calculator_tool() {
        let tool = CalculatorTool;

        let mut inputs = HashMap::new();
        inputs.insert("expression".to_string(), serde_json::json!("2+3"));

        let result = tool.execute(inputs).await.unwrap();
        assert_eq!(result, "5");
    }

    #[tokio::test]
    async fn test_search_tool() {
        let tool = SearchTool::new("search");

        let mut inputs = HashMap::new();
        inputs.insert("query".to_string(), serde_json::json!("rust programming"));

        let result = tool.execute(inputs).await.unwrap();
        assert!(result.contains("rust programming"));
    }

    #[test]
    fn test_tool_registry() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(CalculatorTool));
        registry.register(Box::new(SearchTool::new("search")));

        assert_eq!(registry.len(), 2);
        assert!(registry.get_tool("calculator").is_some());
        assert!(registry.get_tool("search").is_some());
        assert!(registry.get_tool("unknown").is_none());
    }

    #[test]
    fn test_tool_schemas() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(CalculatorTool));

        let schemas = registry.get_tool_schemas();
        assert_eq!(schemas.len(), 1);

        let schema = &schemas[0];
        assert_eq!(schema["type"], "function");
        assert_eq!(schema["function"]["name"], "calculator");
    }
}
