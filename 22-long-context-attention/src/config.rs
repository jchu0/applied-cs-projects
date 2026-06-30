//! Attention configuration types.

use serde::{Deserialize, Serialize};

/// Attention algorithm type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionType {
    /// Standard O(n^2) attention.
    Standard,
    /// Memory-efficient FlashAttention.
    Flash,
    /// Sliding window attention.
    SlidingWindow,
    /// Block sparse attention (Longformer-style).
    BlockSparse,
    /// Linear attention.
    Linear,
    /// Auto-select based on sequence length.
    Auto,
}

impl Default for AttentionType {
    fn default() -> Self {
        Self::Auto
    }
}

/// Data type for computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    Float16,
    Float32,
    BFloat16,
}

impl Default for DataType {
    fn default() -> Self {
        Self::Float32
    }
}

/// Configuration for attention computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Number of KV heads (for GQA/MQA), None means same as num_heads.
    pub num_kv_heads: Option<usize>,
    /// Maximum sequence length.
    pub max_seq_len: usize,

    /// Algorithm selection.
    pub attention_type: AttentionType,

    /// Sliding window size.
    pub window_size: usize,

    /// Block size for block sparse attention.
    pub block_size: usize,
    /// Number of global tokens (attend to/from all).
    pub num_global_tokens: usize,
    /// Number of random blocks for long-range.
    pub num_random_blocks: usize,

    /// Use ALiBi positional bias.
    pub use_alibi: bool,
    /// Use RoPE positional encoding.
    pub use_rope: bool,
    /// Causal masking.
    pub causal: bool,
    /// Dropout rate.
    pub dropout: f32,

    /// Data type.
    pub dtype: DataType,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            num_kv_heads: None,
            max_seq_len: 8192,
            attention_type: AttentionType::Auto,
            window_size: 4096,
            block_size: 64,
            num_global_tokens: 256,
            num_random_blocks: 3,
            use_alibi: false,
            use_rope: true,
            causal: true,
            dropout: 0.0,
            dtype: DataType::Float32,
        }
    }
}

impl AttentionConfig {
    /// Create a new configuration with specified heads and dimensions.
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            head_dim,
            ..Default::default()
        }
    }

    /// Set the attention type.
    pub fn with_attention_type(mut self, attention_type: AttentionType) -> Self {
        self.attention_type = attention_type;
        self
    }

    /// Set the window size.
    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Set causal masking.
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Set the maximum sequence length.
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    /// Set the number of KV heads for grouped query attention.
    pub fn with_num_kv_heads(mut self, num_kv_heads: usize) -> Self {
        self.num_kv_heads = Some(num_kv_heads);
        self
    }

    /// Enable RoPE positional encoding.
    pub fn with_rope(mut self, use_rope: bool) -> Self {
        self.use_rope = use_rope;
        self
    }

    /// Enable ALiBi positional bias.
    pub fn with_alibi(mut self, use_alibi: bool) -> Self {
        self.use_alibi = use_alibi;
        self
    }

    /// Get the effective number of KV heads.
    pub fn effective_num_kv_heads(&self) -> usize {
        self.num_kv_heads.unwrap_or(self.num_heads)
    }

    /// Calculate the scale factor for attention scores.
    pub fn scale(&self) -> f32 {
        (self.head_dim as f32).powf(-0.5)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> crate::Result<()> {
        if self.num_heads == 0 {
            return Err(crate::Error::InvalidConfig(
                "num_heads must be > 0".to_string(),
            ));
        }

        if self.head_dim == 0 {
            return Err(crate::Error::InvalidConfig(
                "head_dim must be > 0".to_string(),
            ));
        }

        if let Some(kv_heads) = self.num_kv_heads {
            if self.num_heads % kv_heads != 0 {
                return Err(crate::Error::InvalidConfig(
                    "num_heads must be divisible by num_kv_heads".to_string(),
                ));
            }
        }

        if self.window_size == 0 {
            return Err(crate::Error::InvalidConfig(
                "window_size must be > 0".to_string(),
            ));
        }

        if self.block_size == 0 {
            return Err(crate::Error::InvalidConfig(
                "block_size must be > 0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Tensor shape for attention computations.
#[derive(Debug, Clone, Copy)]
pub struct TensorShape {
    pub batch: usize,
    pub seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl TensorShape {
    /// Create a new tensor shape.
    pub fn new(batch: usize, seq_len: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            batch,
            seq_len,
            num_heads,
            head_dim,
        }
    }

    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        self.batch * self.seq_len * self.num_heads * self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AttentionConfig::default();
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert!(config.causal);
    }

    #[test]
    fn test_config_builder() {
        let config = AttentionConfig::new(16, 128)
            .with_attention_type(AttentionType::Flash)
            .with_window_size(2048)
            .with_causal(false);

        assert_eq!(config.num_heads, 16);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.attention_type, AttentionType::Flash);
        assert_eq!(config.window_size, 2048);
        assert!(!config.causal);
    }

    #[test]
    fn test_config_validation() {
        let config = AttentionConfig::new(8, 64);
        assert!(config.validate().is_ok());

        let bad_config = AttentionConfig::new(0, 64);
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_gqa_config() {
        let config = AttentionConfig::new(32, 128).with_num_kv_heads(8);
        assert_eq!(config.effective_num_kv_heads(), 8);
        assert!(config.validate().is_ok());
    }
}
