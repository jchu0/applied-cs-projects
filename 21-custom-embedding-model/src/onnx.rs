//! ONNX export and runtime integration.
//!
//! Provides:
//! - ONNX model export from embedding models
//! - ONNX Runtime session management
//! - Inference optimization and quantization
//! - Dynamic batching for inference

use crate::{Error, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// ONNX data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxDataType {
    Float32,
    Float16,
    Int64,
    Int32,
    Int8,
    Uint8,
    Bool,
    String,
}

impl OnnxDataType {
    /// Get size in bytes.
    pub fn size(&self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float16 => 2,
            Self::Int64 => 8,
            Self::Int32 => 4,
            Self::Int8 | Self::Uint8 | Self::Bool => 1,
            Self::String => 0, // Variable
        }
    }

    /// Get ONNX type constant.
    pub fn onnx_type(&self) -> i32 {
        match self {
            Self::Float32 => 1,
            Self::Float16 => 10,
            Self::Int64 => 7,
            Self::Int32 => 6,
            Self::Int8 => 3,
            Self::Uint8 => 2,
            Self::Bool => 9,
            Self::String => 8,
        }
    }
}

/// ONNX tensor description.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name.
    pub name: String,
    /// Data type.
    pub dtype: OnnxDataType,
    /// Shape (negative values indicate dynamic dimensions).
    pub shape: Vec<i64>,
}

impl TensorInfo {
    /// Create new tensor info.
    pub fn new(name: impl Into<String>, dtype: OnnxDataType, shape: Vec<i64>) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape,
        }
    }

    /// Check if tensor has dynamic dimensions.
    pub fn is_dynamic(&self) -> bool {
        self.shape.iter().any(|&d| d < 0)
    }

    /// Get number of elements (None if dynamic).
    pub fn num_elements(&self) -> Option<usize> {
        if self.is_dynamic() {
            None
        } else {
            Some(self.shape.iter().map(|&d| d as usize).product())
        }
    }
}

/// ONNX export configuration.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// ONNX opset version.
    pub opset_version: i32,
    /// Enable dynamic axes.
    pub dynamic_axes: HashMap<String, Vec<usize>>,
    /// Input names.
    pub input_names: Vec<String>,
    /// Output names.
    pub output_names: Vec<String>,
    /// Enable optimization.
    pub optimize: bool,
    /// External data threshold (bytes).
    pub external_data_threshold: Option<usize>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        let mut dynamic_axes = HashMap::new();
        dynamic_axes.insert("input_ids".to_string(), vec![0, 1]); // batch, seq
        dynamic_axes.insert("attention_mask".to_string(), vec![0, 1]);
        dynamic_axes.insert("embeddings".to_string(), vec![0]);

        Self {
            opset_version: 17,
            dynamic_axes,
            input_names: vec!["input_ids".to_string(), "attention_mask".to_string()],
            output_names: vec!["embeddings".to_string()],
            optimize: true,
            external_data_threshold: Some(1024 * 1024 * 1024), // 1GB
        }
    }
}

/// ONNX Runtime execution providers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionProvider {
    /// CPU execution.
    Cpu,
    /// CUDA GPU execution.
    Cuda { device_id: i32 },
    /// TensorRT execution.
    TensorRT { device_id: i32 },
    /// CoreML (Apple).
    CoreML,
    /// DirectML (Windows).
    DirectML { device_id: i32 },
    /// OpenVINO.
    OpenVINO,
    /// NNAPI (Android).
    Nnapi,
}

impl ExecutionProvider {
    /// Get provider name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "CPUExecutionProvider",
            Self::Cuda { .. } => "CUDAExecutionProvider",
            Self::TensorRT { .. } => "TensorrtExecutionProvider",
            Self::CoreML => "CoreMLExecutionProvider",
            Self::DirectML { .. } => "DmlExecutionProvider",
            Self::OpenVINO => "OpenVINOExecutionProvider",
            Self::Nnapi => "NnapiExecutionProvider",
        }
    }
}

/// ONNX Runtime session configuration.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Execution providers in priority order.
    pub execution_providers: Vec<ExecutionProvider>,
    /// Number of intra-op threads.
    pub intra_op_num_threads: i32,
    /// Number of inter-op threads.
    pub inter_op_num_threads: i32,
    /// Enable memory pattern optimization.
    pub enable_mem_pattern: bool,
    /// Enable CPU memory arena.
    pub enable_cpu_mem_arena: bool,
    /// Graph optimization level.
    pub graph_optimization_level: GraphOptimizationLevel,
    /// Log severity level.
    pub log_severity_level: i32,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            execution_providers: vec![ExecutionProvider::Cpu],
            intra_op_num_threads: 0, // Use default
            inter_op_num_threads: 0,
            enable_mem_pattern: true,
            enable_cpu_mem_arena: true,
            graph_optimization_level: GraphOptimizationLevel::All,
            log_severity_level: 2, // Warning
        }
    }
}

/// Graph optimization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphOptimizationLevel {
    /// No optimization.
    Disabled,
    /// Basic optimizations.
    Basic,
    /// Extended optimizations.
    Extended,
    /// All optimizations.
    All,
}

impl GraphOptimizationLevel {
    /// Get ORT level value.
    pub fn as_ort_level(&self) -> i32 {
        match self {
            Self::Disabled => 0,
            Self::Basic => 1,
            Self::Extended => 2,
            Self::All => 99,
        }
    }
}

/// ONNX Runtime inference session.
pub struct OnnxSession {
    /// Model path.
    model_path: PathBuf,
    /// Session configuration.
    config: SessionConfig,
    /// Input tensor infos.
    inputs: Vec<TensorInfo>,
    /// Output tensor infos.
    outputs: Vec<TensorInfo>,
    /// Statistics.
    stats: SessionStats,
    /// Session initialized.
    initialized: bool,
}

/// Session statistics.
#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    /// Number of inference runs.
    pub num_runs: usize,
    /// Total inference time.
    pub total_inference_time: Duration,
    /// Average latency.
    pub avg_latency_ms: f64,
    /// P50 latency.
    pub p50_latency_ms: f64,
    /// P99 latency.
    pub p99_latency_ms: f64,
    /// Latency samples for percentile calculation.
    latency_samples: Vec<f64>,
}

impl SessionStats {
    /// Update with new latency sample.
    pub fn add_sample(&mut self, latency_ms: f64) {
        self.num_runs += 1;
        self.total_inference_time += Duration::from_secs_f64(latency_ms / 1000.0);
        self.latency_samples.push(latency_ms);

        // Update average
        self.avg_latency_ms = self.latency_samples.iter().sum::<f64>() / self.num_runs as f64;

        // Update percentiles (keep sorted)
        self.latency_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = self.latency_samples.len();
        self.p50_latency_ms = self.latency_samples[n / 2];
        self.p99_latency_ms = self.latency_samples[(n * 99) / 100];
    }
}

impl OnnxSession {
    /// Create new session from model file.
    pub fn new(model_path: impl AsRef<Path>, config: SessionConfig) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();

        // Define default inputs/outputs for embedding models
        let inputs = vec![
            TensorInfo::new("input_ids", OnnxDataType::Int64, vec![-1, -1]),
            TensorInfo::new("attention_mask", OnnxDataType::Int64, vec![-1, -1]),
        ];

        let outputs = vec![
            TensorInfo::new("embeddings", OnnxDataType::Float32, vec![-1, 768]),
        ];

        Ok(Self {
            model_path,
            config,
            inputs,
            outputs,
            stats: SessionStats::default(),
            initialized: true,
        })
    }

    /// Run inference.
    pub fn run(&mut self, inputs: &[OnnxTensor]) -> Result<Vec<OnnxTensor>> {
        if !self.initialized {
            return Err(Error::InvalidConfig("Session not initialized".into()));
        }

        let start = Instant::now();

        // Validate inputs
        if inputs.len() != self.inputs.len() {
            return Err(Error::InvalidConfig(format!(
                "Expected {} inputs, got {}",
                self.inputs.len(),
                inputs.len()
            )));
        }

        // Simulate inference
        let batch_size = inputs[0].shape[0] as usize;
        let output_dim = 768;

        let output_data: Vec<f32> = (0..batch_size * output_dim)
            .map(|i| (i as f32).sin() * 0.1)
            .collect();

        let output = OnnxTensor {
            name: "embeddings".to_string(),
            dtype: OnnxDataType::Float32,
            shape: vec![batch_size as i64, output_dim as i64],
            data: TensorData::Float32(output_data),
        };

        // Update stats
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.stats.add_sample(latency_ms);

        Ok(vec![output])
    }

    /// Get input tensor infos.
    pub fn inputs(&self) -> &[TensorInfo] {
        &self.inputs
    }

    /// Get output tensor infos.
    pub fn outputs(&self) -> &[TensorInfo] {
        &self.outputs
    }

    /// Get statistics.
    pub fn stats(&self) -> &SessionStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = SessionStats::default();
    }

    /// Get model path.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Get active execution providers.
    pub fn execution_providers(&self) -> &[ExecutionProvider] {
        &self.config.execution_providers
    }
}

/// ONNX tensor data.
#[derive(Debug, Clone)]
pub enum TensorData {
    Float32(Vec<f32>),
    Float16(Vec<u16>), // Stored as raw bits
    Int64(Vec<i64>),
    Int32(Vec<i32>),
    Int8(Vec<i8>),
    Uint8(Vec<u8>),
    Bool(Vec<bool>),
}

/// ONNX tensor.
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    /// Tensor name.
    pub name: String,
    /// Data type.
    pub dtype: OnnxDataType,
    /// Shape.
    pub shape: Vec<i64>,
    /// Data.
    pub data: TensorData,
}

impl OnnxTensor {
    /// Create new float32 tensor.
    pub fn new_f32(name: impl Into<String>, shape: Vec<i64>, data: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            dtype: OnnxDataType::Float32,
            shape,
            data: TensorData::Float32(data),
        }
    }

    /// Create new int64 tensor.
    pub fn new_i64(name: impl Into<String>, shape: Vec<i64>, data: Vec<i64>) -> Self {
        Self {
            name: name.into(),
            dtype: OnnxDataType::Int64,
            shape,
            data: TensorData::Int64(data),
        }
    }

    /// Get number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    /// Get data as f32 slice.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.data {
            TensorData::Float32(data) => Some(data),
            _ => None,
        }
    }

    /// Get data as i64 slice.
    pub fn as_i64(&self) -> Option<&[i64]> {
        match &self.data {
            TensorData::Int64(data) => Some(data),
            _ => None,
        }
    }
}

/// ONNX model exporter.
pub struct OnnxExporter {
    /// Export configuration.
    config: ExportConfig,
}

impl OnnxExporter {
    /// Create new exporter.
    pub fn new(config: ExportConfig) -> Self {
        Self { config }
    }

    /// Export model to ONNX format.
    pub fn export(
        &self,
        model_params: &HashMap<String, Vec<f32>>,
        output_path: impl AsRef<Path>,
    ) -> Result<ExportResult> {
        let output_path = output_path.as_ref();
        let start = Instant::now();

        // Calculate model size
        let total_params: usize = model_params.values().map(|v| v.len()).sum();
        let model_size_bytes = total_params * std::mem::size_of::<f32>();

        // Build ONNX graph (simulated)
        let graph = self.build_graph(model_params)?;

        // Apply optimizations
        let optimized_graph = if self.config.optimize {
            self.optimize_graph(graph)?
        } else {
            graph
        };

        // Serialize to file (simulated)
        let export_time = start.elapsed();

        Ok(ExportResult {
            output_path: output_path.to_path_buf(),
            model_size_bytes,
            num_parameters: total_params,
            opset_version: self.config.opset_version,
            export_time,
            num_nodes: optimized_graph.nodes.len(),
        })
    }

    /// Build ONNX graph from model.
    fn build_graph(&self, model_params: &HashMap<String, Vec<f32>>) -> Result<OnnxGraph> {
        let mut nodes = Vec::new();
        let mut initializers = Vec::new();

        // Add input nodes
        for input_name in &self.config.input_names {
            nodes.push(OnnxNode {
                name: input_name.clone(),
                op_type: "Input".to_string(),
                inputs: vec![],
                outputs: vec![input_name.clone()],
                attributes: HashMap::new(),
            });
        }

        // Add parameter initializers
        for (name, values) in model_params {
            initializers.push(OnnxInitializer {
                name: name.clone(),
                dtype: OnnxDataType::Float32,
                dims: vec![values.len() as i64],
                data: values.clone(),
            });
        }

        // Add computation nodes (simplified)
        nodes.push(OnnxNode {
            name: "embedding_layer".to_string(),
            op_type: "Gather".to_string(),
            inputs: vec!["embedding_weights".to_string(), "input_ids".to_string()],
            outputs: vec!["token_embeddings".to_string()],
            attributes: HashMap::new(),
        });

        nodes.push(OnnxNode {
            name: "mean_pooling".to_string(),
            op_type: "ReduceMean".to_string(),
            inputs: vec!["token_embeddings".to_string(), "attention_mask".to_string()],
            outputs: vec!["embeddings".to_string()],
            attributes: {
                let mut attrs = HashMap::new();
                attrs.insert("axes".to_string(), OnnxAttribute::IntList(vec![1]));
                attrs
            },
        });

        Ok(OnnxGraph {
            nodes,
            initializers,
            inputs: self.config.input_names.clone(),
            outputs: self.config.output_names.clone(),
        })
    }

    /// Apply graph optimizations.
    fn optimize_graph(&self, mut graph: OnnxGraph) -> Result<OnnxGraph> {
        // Apply constant folding
        // Apply common subexpression elimination
        // Apply operator fusion

        // In a real implementation, this would use ONNX Runtime's graph optimizers
        Ok(graph)
    }
}

/// ONNX graph representation.
#[derive(Debug)]
struct OnnxGraph {
    nodes: Vec<OnnxNode>,
    initializers: Vec<OnnxInitializer>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

/// ONNX node.
#[derive(Debug)]
struct OnnxNode {
    name: String,
    op_type: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attributes: HashMap<String, OnnxAttribute>,
}

/// ONNX attribute.
#[derive(Debug)]
enum OnnxAttribute {
    Int(i64),
    Float(f32),
    String(String),
    IntList(Vec<i64>),
    FloatList(Vec<f32>),
}

/// ONNX initializer (weight/bias).
#[derive(Debug)]
struct OnnxInitializer {
    name: String,
    dtype: OnnxDataType,
    dims: Vec<i64>,
    data: Vec<f32>,
}

/// Export result.
#[derive(Debug)]
pub struct ExportResult {
    /// Output file path.
    pub output_path: PathBuf,
    /// Model size in bytes.
    pub model_size_bytes: usize,
    /// Number of parameters.
    pub num_parameters: usize,
    /// ONNX opset version.
    pub opset_version: i32,
    /// Export time.
    pub export_time: Duration,
    /// Number of nodes in graph.
    pub num_nodes: usize,
}

/// Quantization configuration for ONNX models.
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Quantization mode.
    pub mode: QuantizationMode,
    /// Weight type after quantization.
    pub weight_type: OnnxDataType,
    /// Activation type after quantization.
    pub activation_type: OnnxDataType,
    /// Per-channel quantization.
    pub per_channel: bool,
    /// Nodes to exclude from quantization.
    pub exclude_nodes: Vec<String>,
    /// Calibration method.
    pub calibration_method: CalibrationMethod,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            mode: QuantizationMode::Dynamic,
            weight_type: OnnxDataType::Int8,
            activation_type: OnnxDataType::Int8,
            per_channel: false,
            exclude_nodes: vec![],
            calibration_method: CalibrationMethod::MinMax,
        }
    }
}

/// Quantization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    /// Dynamic quantization.
    Dynamic,
    /// Static quantization with calibration.
    Static,
    /// Quantization-aware training.
    Qat,
}

/// Calibration method for static quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationMethod {
    /// Min-max calibration.
    MinMax,
    /// Entropy calibration.
    Entropy,
    /// Percentile calibration.
    Percentile { percentile: u8 },
}

/// ONNX model quantizer.
pub struct OnnxQuantizer {
    /// Quantization config.
    config: QuantizationConfig,
}

impl OnnxQuantizer {
    /// Create new quantizer.
    pub fn new(config: QuantizationConfig) -> Self {
        Self { config }
    }

    /// Quantize model.
    pub fn quantize(
        &self,
        input_path: impl AsRef<Path>,
        output_path: impl AsRef<Path>,
    ) -> Result<QuantizationResult> {
        let input_path = input_path.as_ref();
        let output_path = output_path.as_ref();

        // In a real implementation, this would use ONNX Runtime quantization tools
        let original_size = 1000000; // Simulated
        let quantized_size = match self.config.weight_type {
            OnnxDataType::Int8 | OnnxDataType::Uint8 => original_size / 4,
            OnnxDataType::Float16 => original_size / 2,
            _ => original_size,
        };

        Ok(QuantizationResult {
            output_path: output_path.to_path_buf(),
            original_size_bytes: original_size,
            quantized_size_bytes: quantized_size,
            compression_ratio: original_size as f32 / quantized_size as f32,
            mode: self.config.mode,
        })
    }

    /// Calibrate model with sample data.
    pub fn calibrate(&self, _model_path: impl AsRef<Path>, _calibration_data: &[OnnxTensor]) -> Result<()> {
        // In a real implementation, this would run inference on calibration data
        // and collect activation statistics for quantization
        Ok(())
    }
}

/// Quantization result.
#[derive(Debug)]
pub struct QuantizationResult {
    /// Output file path.
    pub output_path: PathBuf,
    /// Original model size.
    pub original_size_bytes: usize,
    /// Quantized model size.
    pub quantized_size_bytes: usize,
    /// Compression ratio.
    pub compression_ratio: f32,
    /// Quantization mode used.
    pub mode: QuantizationMode,
}

/// Dynamic batcher for inference.
pub struct DynamicBatcher {
    /// Maximum batch size.
    max_batch_size: usize,
    /// Maximum wait time.
    max_wait_time: Duration,
    /// Pending requests.
    pending: Vec<BatchRequest>,
    /// Statistics.
    stats: BatcherStats,
}

/// Batch request.
#[derive(Debug)]
struct BatchRequest {
    inputs: Vec<OnnxTensor>,
    timestamp: Instant,
}

/// Batcher statistics.
#[derive(Debug, Clone, Default)]
pub struct BatcherStats {
    /// Total requests processed.
    pub total_requests: usize,
    /// Total batches processed.
    pub total_batches: usize,
    /// Average batch size.
    pub avg_batch_size: f32,
    /// Average wait time.
    pub avg_wait_time_ms: f64,
}

impl DynamicBatcher {
    /// Create new batcher.
    pub fn new(max_batch_size: usize, max_wait_time: Duration) -> Self {
        Self {
            max_batch_size,
            max_wait_time,
            pending: Vec::new(),
            stats: BatcherStats::default(),
        }
    }

    /// Add request to batch.
    pub fn add(&mut self, inputs: Vec<OnnxTensor>) -> Option<Vec<Vec<OnnxTensor>>> {
        self.pending.push(BatchRequest {
            inputs,
            timestamp: Instant::now(),
        });

        // Check if we should flush
        if self.should_flush() {
            Some(self.flush())
        } else {
            None
        }
    }

    /// Check if batch should be flushed.
    fn should_flush(&self) -> bool {
        if self.pending.len() >= self.max_batch_size {
            return true;
        }

        if let Some(oldest) = self.pending.first() {
            if oldest.timestamp.elapsed() >= self.max_wait_time {
                return true;
            }
        }

        false
    }

    /// Flush pending requests into batches.
    pub fn flush(&mut self) -> Vec<Vec<OnnxTensor>> {
        let requests: Vec<_> = self.pending.drain(..).collect();

        if requests.is_empty() {
            return vec![];
        }

        // Update stats
        let wait_times: Vec<f64> = requests
            .iter()
            .map(|r| r.timestamp.elapsed().as_secs_f64() * 1000.0)
            .collect();

        let avg_wait = wait_times.iter().sum::<f64>() / wait_times.len() as f64;

        self.stats.total_requests += requests.len();
        self.stats.total_batches += 1;
        self.stats.avg_batch_size = self.stats.total_requests as f32 / self.stats.total_batches as f32;
        self.stats.avg_wait_time_ms =
            (self.stats.avg_wait_time_ms * (self.stats.total_batches - 1) as f64 + avg_wait)
            / self.stats.total_batches as f64;

        // Group into batches
        let mut batches = vec![];
        for chunk in requests.chunks(self.max_batch_size) {
            let batch: Vec<Vec<OnnxTensor>> = chunk.iter().map(|r| r.inputs.clone()).collect();
            batches.push(batch.into_iter().flatten().collect());
        }

        batches
    }

    /// Get statistics.
    pub fn stats(&self) -> &BatcherStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_data_type() {
        assert_eq!(OnnxDataType::Float32.size(), 4);
        assert_eq!(OnnxDataType::Float16.size(), 2);
        assert_eq!(OnnxDataType::Int8.size(), 1);
    }

    #[test]
    fn test_tensor_info() {
        let info = TensorInfo::new("input", OnnxDataType::Float32, vec![1, 768]);
        assert!(!info.is_dynamic());
        assert_eq!(info.num_elements(), Some(768));

        let dynamic = TensorInfo::new("input", OnnxDataType::Float32, vec![-1, 768]);
        assert!(dynamic.is_dynamic());
        assert_eq!(dynamic.num_elements(), None);
    }

    #[test]
    fn test_export_config_default() {
        let config = ExportConfig::default();
        assert_eq!(config.opset_version, 17);
        assert!(config.optimize);
    }

    #[test]
    fn test_execution_provider() {
        let cpu = ExecutionProvider::Cpu;
        assert_eq!(cpu.name(), "CPUExecutionProvider");

        let cuda = ExecutionProvider::Cuda { device_id: 0 };
        assert_eq!(cuda.name(), "CUDAExecutionProvider");
    }

    #[test]
    fn test_session_config_default() {
        let config = SessionConfig::default();
        assert_eq!(config.execution_providers, vec![ExecutionProvider::Cpu]);
        assert_eq!(config.graph_optimization_level, GraphOptimizationLevel::All);
    }

    #[test]
    fn test_onnx_session_creation() {
        let config = SessionConfig::default();
        let session = OnnxSession::new("/tmp/model.onnx", config);
        assert!(session.is_ok());
    }

    #[test]
    fn test_onnx_session_inference() {
        let config = SessionConfig::default();
        let mut session = OnnxSession::new("/tmp/model.onnx", config).unwrap();

        let input_ids = OnnxTensor::new_i64("input_ids", vec![2, 10], vec![0i64; 20]);
        let attention_mask = OnnxTensor::new_i64("attention_mask", vec![2, 10], vec![1i64; 20]);

        let outputs = session.run(&[input_ids, attention_mask]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape, vec![2, 768]);
    }

    #[test]
    fn test_session_stats() {
        let config = SessionConfig::default();
        let mut session = OnnxSession::new("/tmp/model.onnx", config).unwrap();

        let input_ids = OnnxTensor::new_i64("input_ids", vec![1, 5], vec![0i64; 5]);
        let attention_mask = OnnxTensor::new_i64("attention_mask", vec![1, 5], vec![1i64; 5]);

        for _ in 0..10 {
            session.run(&[input_ids.clone(), attention_mask.clone()]).unwrap();
        }

        let stats = session.stats();
        assert_eq!(stats.num_runs, 10);
        assert!(stats.avg_latency_ms >= 0.0);
    }

    #[test]
    fn test_onnx_tensor() {
        let tensor = OnnxTensor::new_f32("test", vec![2, 3], vec![1.0; 6]);
        assert_eq!(tensor.num_elements(), 6);
        assert!(tensor.as_f32().is_some());
        assert!(tensor.as_i64().is_none());
    }

    #[test]
    fn test_onnx_exporter() {
        let config = ExportConfig::default();
        let exporter = OnnxExporter::new(config);

        let mut params = HashMap::new();
        params.insert("embedding_weights".to_string(), vec![0.1; 768 * 30000]);

        let result = exporter.export(&params, "/tmp/model.onnx");
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.model_size_bytes > 0);
        assert_eq!(result.opset_version, 17);
    }

    #[test]
    fn test_quantization_config() {
        let config = QuantizationConfig::default();
        assert_eq!(config.mode, QuantizationMode::Dynamic);
        assert_eq!(config.weight_type, OnnxDataType::Int8);
    }

    #[test]
    fn test_quantizer() {
        let config = QuantizationConfig::default();
        let quantizer = OnnxQuantizer::new(config);

        let result = quantizer.quantize("/tmp/model.onnx", "/tmp/model_quant.onnx");
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.compression_ratio > 1.0);
    }

    #[test]
    fn test_dynamic_batcher() {
        let mut batcher = DynamicBatcher::new(4, Duration::from_millis(100));

        let tensor = OnnxTensor::new_f32("input", vec![1, 10], vec![0.0; 10]);

        // Add requests but don't exceed batch size
        assert!(batcher.add(vec![tensor.clone()]).is_none());
        assert!(batcher.add(vec![tensor.clone()]).is_none());
        assert!(batcher.add(vec![tensor.clone()]).is_none());

        // Fourth request should trigger flush
        let batches = batcher.add(vec![tensor.clone()]);
        assert!(batches.is_some());

        let stats = batcher.stats();
        assert_eq!(stats.total_requests, 4);
        assert_eq!(stats.total_batches, 1);
    }

    #[test]
    fn test_batcher_flush() {
        let mut batcher = DynamicBatcher::new(10, Duration::from_millis(1));

        let tensor = OnnxTensor::new_f32("input", vec![1, 5], vec![0.0; 5]);
        batcher.add(vec![tensor.clone()]);
        batcher.add(vec![tensor.clone()]);

        // Manual flush
        let batches = batcher.flush();
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_graph_optimization_level() {
        assert_eq!(GraphOptimizationLevel::Disabled.as_ort_level(), 0);
        assert_eq!(GraphOptimizationLevel::Basic.as_ort_level(), 1);
        assert_eq!(GraphOptimizationLevel::All.as_ort_level(), 99);
    }
}
