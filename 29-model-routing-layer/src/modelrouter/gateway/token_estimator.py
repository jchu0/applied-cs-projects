"""Token estimation for requests."""

from typing import Any

from ..schemas import InferenceRequest, ModelPricing


class TokenEstimator:
    """Estimates token count for requests."""

    def __init__(self, tokenizers: dict[str, Any] = None):
        """Initialize estimator.

        Args:
            tokenizers: Model to tokenizer mapping
        """
        self.tokenizers = tokenizers or {}

    def estimate(
        self,
        prompt: str,
        max_tokens: int,
        model: str = "default"
    ) -> int:
        """Estimate total token count.

        Args:
            prompt: Input prompt
            max_tokens: Maximum output tokens
            model: Model name

        Returns:
            Estimated total tokens
        """
        # Get tokenizer for model
        tokenizer = self.tokenizers.get(model, self.tokenizers.get("default"))

        if tokenizer:
            input_tokens = len(tokenizer.encode(prompt))
        else:
            # Simple estimation: ~4 chars per token
            input_tokens = len(prompt) // 4

        return input_tokens + max_tokens

    def estimate_input_tokens(
        self,
        prompt: str,
        model: str = "default"
    ) -> int:
        """Estimate input tokens only.

        Args:
            prompt: Input prompt
            model: Model name

        Returns:
            Estimated input tokens
        """
        tokenizer = self.tokenizers.get(model, self.tokenizers.get("default"))

        if tokenizer:
            return len(tokenizer.encode(prompt))
        else:
            return len(prompt) // 4

    def estimate_cost(
        self,
        request: InferenceRequest,
        pricing: ModelPricing
    ) -> float:
        """Estimate cost in dollars.

        Args:
            request: Inference request
            pricing: Model pricing config

        Returns:
            Estimated cost in dollars
        """
        input_tokens = self.estimate_input_tokens(request.prompt, request.model)
        output_tokens = request.max_tokens

        return (
            input_tokens * pricing.input_cost_per_1k / 1000 +
            output_tokens * pricing.output_cost_per_1k / 1000
        )


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, chars_per_token: int = 4):
        self.chars_per_token = chars_per_token

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return list(range(len(text) // self.chars_per_token))

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        return "x" * (len(tokens) * self.chars_per_token)
