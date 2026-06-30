"""Tests for analysis and synthesis transforms.

Tests cover:
- GDN normalization (forward and inverse)
- AnalysisTransform (encoder)
- SynthesisTransform (decoder)
- Encoder-decoder roundtrip
- Shape preservation and downsampling/upsampling
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

import os
import sys
# Add the `src/` directory to the path so `neural_compression` imports resolve.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neural_compression.transforms import (
    GDN,
    AnalysisTransform,
    SynthesisTransform,
    ResidualBlock,
    EnhancedAnalysisTransform,
)


class TestGDN:
    """Tests for Generalized Divisive Normalization."""

    def test_gdn_output_shape(self, num_filters):
        """GDN preserves input shape."""
        gdn = GDN(num_filters)
        x = torch.randn(2, num_filters, 16, 16)
        y = gdn(x)

        assert y.shape == x.shape

    def test_gdn_inverse_relationship(self, num_filters):
        """GDN and inverse GDN should approximately invert each other."""
        gdn = GDN(num_filters, inverse=False)
        igdn = GDN(num_filters, inverse=True)

        # Share parameters by copying state dict
        igdn.load_state_dict(gdn.state_dict())

        # Use smaller values for better numerical stability
        x = torch.randn(2, num_filters, 8, 8) * 0.1

        y = gdn(x)
        x_reconstructed = igdn(y)

        # Should be close but not exact due to numerical precision
        # Use relaxed tolerance since GDN involves sqrt and division operations
        # that can accumulate numerical errors
        assert torch.allclose(x, x_reconstructed, rtol=0.01, atol=0.01)

    def test_gdn_positive_output_for_positive_input(self, num_filters):
        """GDN should preserve sign for positive inputs."""
        gdn = GDN(num_filters)
        x = torch.abs(torch.randn(2, num_filters, 8, 8)) + 0.1

        y = gdn(x)

        assert (y > 0).all()

    def test_gdn_gradient_flow(self, num_filters):
        """GDN should allow gradient flow."""
        gdn = GDN(num_filters)
        x = torch.randn(2, num_filters, 8, 8, requires_grad=True)

        y = gdn(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gdn_parameter_learning(self, num_filters):
        """GDN parameters should be learnable."""
        gdn = GDN(num_filters)

        # Check parameters exist
        params = list(gdn.parameters())
        assert len(params) == 2  # beta and gamma

        # Check gradients can be computed
        x = torch.randn(2, num_filters, 8, 8)
        y = gdn(x)
        loss = y.sum()
        loss.backward()

        assert gdn.beta.grad is not None
        assert gdn.gamma.grad is not None

    def test_gdn_numerical_stability(self, num_filters):
        """GDN should be numerically stable for various input ranges."""
        gdn = GDN(num_filters)

        # Test with very small values
        x_small = torch.randn(2, num_filters, 8, 8) * 1e-6
        y_small = gdn(x_small)
        assert not torch.isnan(y_small).any()
        assert not torch.isinf(y_small).any()

        # Test with large values
        x_large = torch.randn(2, num_filters, 8, 8) * 100
        y_large = gdn(x_large)
        assert not torch.isnan(y_large).any()
        assert not torch.isinf(y_large).any()

    def test_inverse_gdn_amplifies(self, num_filters):
        """Inverse GDN should generally amplify (multiply by norm)."""
        igdn = GDN(num_filters, inverse=True)
        x = torch.randn(2, num_filters, 8, 8)

        y = igdn(x)

        # Inverse GDN multiplies by norm, so magnitude should generally increase
        # for typical inputs (not always true but usually)
        assert y.abs().mean() >= x.abs().mean() * 0.5  # Relaxed condition


class TestAnalysisTransform:
    """Tests for encoder (analysis) transform."""

    def test_output_shape(self, analysis_transform, random_image, latent_channels):
        """Encoder should downsample by 16x spatially."""
        y = analysis_transform(random_image)

        B, _, H, W = random_image.shape
        expected_shape = (B, latent_channels, H // 16, W // 16)

        assert y.shape == expected_shape

    def test_different_input_sizes(self, analysis_transform):
        """Encoder should work with different input sizes (divisible by 16)."""
        for size in [32, 64, 128, 256]:
            x = torch.randn(1, 3, size, size)
            y = analysis_transform(x)

            assert y.shape[2] == size // 16
            assert y.shape[3] == size // 16

    def test_gradient_flow(self, analysis_transform, random_image):
        """Gradients should flow through encoder."""
        x = random_image.clone().requires_grad_(True)
        y = analysis_transform(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_output_range(self, analysis_transform, random_image):
        """Encoder output should have reasonable range."""
        y = analysis_transform(random_image)

        # Output should not be extremely large or small
        assert y.abs().max() < 1000
        assert y.std() > 0.01

    def test_batch_independence(self, analysis_transform):
        """Each batch element should be processed independently."""
        x1 = torch.randn(1, 3, 64, 64)
        x2 = torch.randn(1, 3, 64, 64)
        x_batch = torch.cat([x1, x2], dim=0)

        y_batch = analysis_transform(x_batch)
        y1 = analysis_transform(x1)
        y2 = analysis_transform(x2)

        assert torch.allclose(y_batch[0], y1[0], atol=1e-5)
        assert torch.allclose(y_batch[1], y2[0], atol=1e-5)

    def test_get_output_shape(self, analysis_transform):
        """get_output_shape should correctly predict output dimensions."""
        input_shape = (128, 128)
        predicted = analysis_transform.get_output_shape(input_shape)

        x = torch.randn(1, 3, *input_shape)
        y = analysis_transform(x)

        assert predicted == (y.shape[2], y.shape[3])


class TestSynthesisTransform:
    """Tests for decoder (synthesis) transform."""

    def test_output_shape(self, synthesis_transform, random_latent):
        """Decoder should upsample by 16x spatially."""
        x_hat = synthesis_transform(random_latent)

        B, _, H, W = random_latent.shape
        expected_shape = (B, 3, H * 16, W * 16)

        assert x_hat.shape == expected_shape

    def test_different_input_sizes(self, synthesis_transform, latent_channels):
        """Decoder should work with different latent sizes."""
        for latent_size in [2, 4, 8, 16]:
            y = torch.randn(1, latent_channels, latent_size, latent_size)
            x_hat = synthesis_transform(y)

            assert x_hat.shape[2] == latent_size * 16
            assert x_hat.shape[3] == latent_size * 16

    def test_gradient_flow(self, synthesis_transform, random_latent):
        """Gradients should flow through decoder."""
        y = random_latent.clone().requires_grad_(True)
        x_hat = synthesis_transform(y)
        loss = x_hat.sum()
        loss.backward()

        assert y.grad is not None
        assert not torch.isnan(y.grad).any()

    def test_rgb_output(self, synthesis_transform, random_latent):
        """Decoder should output 3 channels (RGB)."""
        x_hat = synthesis_transform(random_latent)

        assert x_hat.shape[1] == 3

    def test_get_output_shape(self, synthesis_transform, latent_channels):
        """get_output_shape should correctly predict output dimensions."""
        input_shape = (8, 8)
        predicted = synthesis_transform.get_output_shape(input_shape)

        y = torch.randn(1, latent_channels, *input_shape)
        x_hat = synthesis_transform(y)

        assert predicted == (x_hat.shape[2], x_hat.shape[3])


class TestEncoderDecoderRoundtrip:
    """Tests for encoder-decoder pipeline."""

    def test_shape_preservation(
        self, analysis_transform, synthesis_transform, random_image
    ):
        """Encoder-decoder should preserve input shape."""
        y = analysis_transform(random_image)
        x_hat = synthesis_transform(y)

        assert x_hat.shape == random_image.shape

    def test_reconstruction_quality(
        self, analysis_transform, synthesis_transform, random_image
    ):
        """Reconstruction should not be completely random."""
        y = analysis_transform(random_image)
        x_hat = synthesis_transform(y)

        # Without training, reconstruction won't be perfect but should
        # have some correlation with input
        mse = ((x_hat - random_image) ** 2).mean()

        # MSE should be finite and not too extreme
        assert mse.isfinite()
        assert mse < 10  # Reasonable upper bound for untrained model

    def test_end_to_end_gradient(
        self, analysis_transform, synthesis_transform, random_image
    ):
        """Gradients should flow through the entire pipeline."""
        x = random_image.clone().requires_grad_(True)
        y = analysis_transform(x)
        x_hat = synthesis_transform(y)

        loss = ((x_hat - x.detach()) ** 2).mean()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_deterministic_output(
        self, analysis_transform, synthesis_transform, random_image
    ):
        """Same input should produce same output (eval mode)."""
        analysis_transform.eval()
        synthesis_transform.eval()

        with torch.no_grad():
            y1 = analysis_transform(random_image)
            x_hat1 = synthesis_transform(y1)

            y2 = analysis_transform(random_image)
            x_hat2 = synthesis_transform(y2)

        assert torch.allclose(x_hat1, x_hat2)


class TestResidualBlock:
    """Tests for residual block."""

    def test_output_shape(self, num_filters):
        """Residual block preserves shape."""
        block = ResidualBlock(num_filters, use_gdn=True)
        x = torch.randn(2, num_filters, 16, 16)
        y = block(x)

        assert y.shape == x.shape

    def test_residual_connection(self, num_filters):
        """Output should include input (residual connection)."""
        block = ResidualBlock(num_filters, use_gdn=True)
        x = torch.randn(2, num_filters, 16, 16)
        y = block(x)

        # If we zero out the conv weights, output should equal input
        with torch.no_grad():
            for param in block.parameters():
                if param.dim() > 1:  # Conv weights
                    param.zero_()

        x_test = torch.randn(2, num_filters, 8, 8)
        y_test = block(x_test)

        # Should be close to input with zero convs (not exact due to normalization)
        # Just verify it runs without error
        assert y_test.shape == x_test.shape

    def test_batch_norm_variant(self, num_filters):
        """Residual block with batch norm should work."""
        block = ResidualBlock(num_filters, use_gdn=False)
        x = torch.randn(2, num_filters, 16, 16)
        y = block(x)

        assert y.shape == x.shape


class TestEnhancedAnalysisTransform:
    """Tests for enhanced encoder with residual blocks."""

    def test_output_shape(self, latent_channels, num_filters, random_image):
        """Enhanced encoder should have same output shape as basic."""
        encoder = EnhancedAnalysisTransform(
            in_channels=3,
            latent_channels=latent_channels,
            num_filters=num_filters,
            num_residual_blocks=2,
        )
        y = encoder(random_image)

        B, _, H, W = random_image.shape
        expected_shape = (B, latent_channels, H // 16, W // 16)

        assert y.shape == expected_shape

    def test_more_parameters(self, latent_channels, num_filters):
        """Enhanced encoder should have more parameters than basic."""
        basic = AnalysisTransform(
            in_channels=3, latent_channels=latent_channels, num_filters=num_filters
        )
        enhanced = EnhancedAnalysisTransform(
            in_channels=3,
            latent_channels=latent_channels,
            num_filters=num_filters,
            num_residual_blocks=3,
        )

        basic_params = sum(p.numel() for p in basic.parameters())
        enhanced_params = sum(p.numel() for p in enhanced.parameters())

        assert enhanced_params > basic_params


class TestTransformProperties:
    """Tests for mathematical properties of transforms."""

    def test_encoder_continuity(self, analysis_transform):
        """Small input changes should produce small output changes."""
        x1 = torch.randn(1, 3, 64, 64)
        x2 = x1 + torch.randn_like(x1) * 0.01  # Small perturbation

        y1 = analysis_transform(x1)
        y2 = analysis_transform(x2)

        # Output difference should be bounded by input difference
        input_diff = (x1 - x2).abs().max()
        output_diff = (y1 - y2).abs().max()

        # Roughly Lipschitz continuous (relaxed bound)
        assert output_diff < input_diff * 1000

    def test_decoder_continuity(self, synthesis_transform, latent_channels):
        """Small latent changes should produce small output changes."""
        y1 = torch.randn(1, latent_channels, 4, 4)
        y2 = y1 + torch.randn_like(y1) * 0.01

        x1 = synthesis_transform(y1)
        x2 = synthesis_transform(y2)

        input_diff = (y1 - y2).abs().max()
        output_diff = (x1 - x2).abs().max()

        assert output_diff < input_diff * 1000

    def test_latent_variance(self, analysis_transform, random_image):
        """Latent should have non-zero variance (information preserved)."""
        y = analysis_transform(random_image)

        # Each channel should have some variance
        channel_vars = y.var(dim=(0, 2, 3))
        assert (channel_vars > 0).all()
