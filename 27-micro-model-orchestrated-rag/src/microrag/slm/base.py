"""Base SLM class for all specialized models."""

import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Union
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    logging as transformers_logging
)
from sentence_transformers import SentenceTransformer, CrossEncoder
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress excessive warnings from transformers
transformers_logging.set_verbosity_error()

# Set default cache directory for models
CACHE_DIR = os.environ.get('TRANSFORMERS_CACHE', './model_cache')


class BaseSLM(ABC):
    """Base class for all SLM components."""

    def __init__(self, model_name: str = None):
        """Initialize SLM.

        Args:
            model_name: Name of the base model
        """
        self.model_name = model_name
        self._loaded = False
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()

    def _get_device(self) -> torch.device:
        """Get the best available device (GPU if available, else CPU)."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @abstractmethod
    async def process(self, **kwargs) -> Any:
        """Process input and return output.

        Args:
            **kwargs: Input arguments

        Returns:
            Processed output
        """
        pass

    def load(self):
        """Load model weights."""
        if not self._loaded:
            try:
                self._load_model()
                self._loaded = True
                logger.info(f"Loaded model {self.model_name} on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {str(e)}")
                raise

    def _load_model(self):
        """Override this method in subclasses to load specific model types."""
        pass

    def unload(self):
        """Unload model weights to free memory."""
        if self._loaded:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._loaded = False
            logger.info(f"Unloaded model {self.model_name}")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    async def __call__(self, **kwargs) -> Any:
        """Allow calling SLM as function."""
        if not self._loaded:
            self.load()
        return await self.process(**kwargs)

    def tokenize(self, text: Union[str, list], **kwargs) -> dict:
        """Tokenize text using the model's tokenizer.

        Args:
            text: Text or list of texts to tokenize
            **kwargs: Additional tokenizer arguments

        Returns:
            Tokenized inputs
        """
        if not self.tokenizer:
            raise RuntimeError(f"Tokenizer not loaded for {self.model_name}")

        default_kwargs = {
            'padding': True,
            'truncation': True,
            'max_length': 512,
            'return_tensors': 'pt'
        }
        default_kwargs.update(kwargs)

        return self.tokenizer(text, **default_kwargs)

    def generate_text(self, prompt: str, max_length: int = 200, **kwargs) -> str:
        """Generate text using a causal language model.

        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            **kwargs: Additional generation arguments

        Returns:
            Generated text
        """
        if not isinstance(self.model, type(AutoModelForCausalLM)):
            raise RuntimeError(f"Model {self.model_name} is not a text generation model")

        inputs = self.tokenize(prompt).to(self.device)

        default_kwargs = {
            'max_length': max_length,
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.95,
            'pad_token_id': self.tokenizer.eos_token_id
        }
        default_kwargs.update(kwargs)

        with torch.no_grad():
            outputs = self.model.generate(inputs['input_ids'], **default_kwargs)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class EmbeddingModelMixin:
    """Mixin for embedding models."""

    def _load_embedding_model(self, model_name: str):
        """Load a sentence transformer model for embeddings."""
        try:
            self.model = SentenceTransformer(model_name, cache_folder=CACHE_DIR)
            self.model.to(self.device)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {str(e)}")
            # Fallback to a smaller model
            fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
            logger.info(f"Attempting fallback model: {fallback_model}")
            self.model = SentenceTransformer(fallback_model, cache_folder=CACHE_DIR)
            self.model.to(self.device)


class CrossEncoderMixin:
    """Mixin for cross-encoder models."""

    def _load_cross_encoder(self, model_name: str):
        """Load a cross-encoder model for reranking."""
        try:
            self.model = CrossEncoder(model_name, max_length=512)
            logger.info(f"Loaded cross-encoder model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder {model_name}: {str(e)}")
            # Fallback to a smaller model
            fallback_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            logger.info(f"Attempting fallback model: {fallback_model}")
            self.model = CrossEncoder(fallback_model, max_length=512)


class GenerativeModelMixin:
    """Mixin for generative language models."""

    def _load_generative_model(self, model_name: str, load_in_8bit: bool = False):
        """Load a generative language model."""
        try:
            # For smaller models, we can use standard loading
            model_kwargs = {
                'cache_dir': CACHE_DIR,
                'torch_dtype': torch.float16 if self.device.type != 'cpu' else torch.float32,
                'low_cpu_mem_usage': True
            }

            if load_in_8bit and self.device.type == 'cuda':
                model_kwargs['load_in_8bit'] = True

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=CACHE_DIR,
                trust_remote_code=True
            )

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                **model_kwargs
            ).to(self.device)

            self.model.eval()
            logger.info(f"Loaded generative model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to load generative model {model_name}: {str(e)}")
            # Fallback to a smaller model
            fallback_model = "microsoft/phi-2"
            logger.info(f"Attempting fallback model: {fallback_model}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                fallback_model,
                cache_dir=CACHE_DIR,
                trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                fallback_model,
                cache_dir=CACHE_DIR,
                torch_dtype=torch.float16 if self.device.type != 'cpu' else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).to(self.device)

            self.model.eval()


class MockSLM(BaseSLM):
    """Mock SLM for testing."""

    def __init__(self):
        super().__init__("mock")
        self._loaded = True

    async def process(self, **kwargs) -> Any:
        return {"mock": True, "inputs": list(kwargs.keys())}