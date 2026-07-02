"""Tests for data loading utilities."""

import pytest
import torch
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_compression.data import (
    RandomCropTransform,
    CenterCropTransform,
    ComposeTransform,
    RandomHorizontalFlip,
)


class TestRandomCropTransform:
    """Tests for RandomCropTransform."""

    def test_square_crop(self):
        """Test square crop."""
        transform = RandomCropTransform(32)
        image = torch.rand(3, 64, 64)
        cropped = transform(image)

        assert cropped.shape == (3, 32, 32)

    def test_rectangular_crop(self):
        """Test rectangular crop."""
        transform = RandomCropTransform((32, 48))
        image = torch.rand(3, 64, 64)
        cropped = transform(image)

        assert cropped.shape == (3, 32, 48)

    def test_crop_smaller_image(self):
        """Test crop when image is smaller than crop size."""
        transform = RandomCropTransform(64)
        image = torch.rand(3, 32, 32)
        # Should pad and then crop
        cropped = transform(image)

        assert cropped.shape == (3, 64, 64)

    def test_randomness(self):
        """Test that crops are random."""
        transform = RandomCropTransform(32)
        image = torch.arange(64 * 64).reshape(1, 64, 64).float()

        crops = [transform(image) for _ in range(10)]

        # Not all crops should be identical
        different = False
        for i in range(1, len(crops)):
            if not torch.equal(crops[0], crops[i]):
                different = True
                break

        assert different


class TestCenterCropTransform:
    """Tests for CenterCropTransform."""

    def test_center_crop(self):
        """Test center crop."""
        transform = CenterCropTransform(32)
        image = torch.rand(3, 64, 64)
        cropped = transform(image)

        assert cropped.shape == (3, 32, 32)

    def test_deterministic(self):
        """Test that center crop is deterministic."""
        transform = CenterCropTransform(32)
        image = torch.rand(3, 64, 64)

        crop1 = transform(image)
        crop2 = transform(image)

        assert torch.equal(crop1, crop2)


class TestComposeTransform:
    """Tests for ComposeTransform."""

    def test_compose_multiple(self):
        """Test composing multiple transforms."""
        transforms = ComposeTransform([
            RandomCropTransform(48),
            CenterCropTransform(32),
        ])

        image = torch.rand(3, 64, 64)
        result = transforms(image)

        assert result.shape == (3, 32, 32)

    def test_empty_compose(self):
        """Test empty composition."""
        transforms = ComposeTransform([])
        image = torch.rand(3, 64, 64)
        result = transforms(image)

        assert torch.equal(result, image)


class TestRandomHorizontalFlip:
    """Tests for RandomHorizontalFlip."""

    def test_flip_deterministic_p1(self):
        """Test flip with p=1 always flips."""
        transform = RandomHorizontalFlip(p=1.0)
        image = torch.arange(12).reshape(1, 3, 4).float()
        flipped = transform(image)

        # Should be horizontally flipped
        expected = torch.flip(image, dims=[2])
        assert torch.equal(flipped, expected)

    def test_no_flip_p0(self):
        """Test flip with p=0 never flips."""
        transform = RandomHorizontalFlip(p=0.0)
        image = torch.rand(3, 32, 32)
        result = transform(image)

        assert torch.equal(result, image)

    def test_stochastic(self):
        """Test that flip is stochastic with p=0.5."""
        transform = RandomHorizontalFlip(p=0.5)
        image = torch.rand(3, 32, 32)

        # Run many times, should get both flipped and non-flipped
        results = [torch.equal(transform(image.clone()), image) for _ in range(100)]

        # Should have mix of True and False
        assert True in results and False in results


class TestImageFolderDataset:
    """Tests for ImageFolderDataset."""

    def test_empty_directory_raises(self):
        """Test that empty directory raises error."""
        from neural_compression.data import ImageFolderDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                ImageFolderDataset(tmpdir)

    def test_with_images(self):
        """Test with actual images."""
        from neural_compression.data import ImageFolderDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy PNG files (just headers, won't actually load)
            # This test checks the file discovery logic
            try:
                from PIL import Image
                import numpy as np

                # Create real test images
                for i in range(3):
                    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
                    img.save(os.path.join(tmpdir, f"test_{i}.png"))

                dataset = ImageFolderDataset(tmpdir)
                assert len(dataset) == 3

                # Test loading
                img = dataset[0]
                assert img.shape[0] == 3  # RGB
                assert img.min() >= 0
                assert img.max() <= 1

            except ImportError:
                pytest.skip("PIL not available")


class TestKodakDataset:
    """Tests for KodakDataset integrity handling (no real network)."""

    def _make_kodak_png(self, path):
        from PIL import Image
        import numpy as np
        img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        img.save(path)

    def test_download_is_opt_in(self):
        """Default download=False: missing images raise, no network access."""
        from neural_compression.data import KodakDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                KodakDataset(tmpdir)  # download defaults to False

    def test_serves_cached_images(self):
        """A fully populated cache loads without downloading."""
        pytest.importorskip("PIL")
        from neural_compression.data import KodakDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 25):
                self._make_kodak_png(os.path.join(tmpdir, f"kodim{i:02d}.png"))
            ds = KodakDataset(tmpdir, download=False)
            assert len(ds) == 24
            img = ds[0]
            assert img.shape[0] == 3

    def test_download_validates_and_moves(self, monkeypatch):
        """A mocked download writes valid PNGs; files are validated and cached."""
        pytest.importorskip("PIL")
        import urllib.request
        from neural_compression.data import KodakDataset

        def fake_urlretrieve(url, dest):
            self._make_kodak_png(dest)

        monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

        with tempfile.TemporaryDirectory() as tmpdir:
            ds = KodakDataset(tmpdir, download=True)
            assert len(ds) == 24
            # No leftover partial files.
            assert not list(Path(tmpdir).glob("*.part"))
            assert all((Path(tmpdir) / f"kodim{i:02d}.png").exists() for i in range(1, 25))

    def test_corrupt_download_raises_and_cleans_up(self, monkeypatch):
        """A corrupt download raises a clear error and leaves no partial file."""
        pytest.importorskip("PIL")
        import urllib.request
        from neural_compression.data import KodakDataset

        def fake_urlretrieve(url, dest):
            # Write garbage that is not a decodable image.
            with open(dest, "wb") as f:
                f.write(b"not a real png")

        monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError):
                KodakDataset(tmpdir, download=True)
            # Cleaned up: no partial files and no bogus final image.
            assert not list(Path(tmpdir).glob("*.part"))
            assert not (Path(tmpdir) / "kodim01.png").exists()

    def test_empty_download_raises(self, monkeypatch):
        """A zero-byte download is rejected."""
        import urllib.request
        from neural_compression.data import KodakDataset

        def fake_urlretrieve(url, dest):
            open(dest, "wb").close()  # empty file

        monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError):
                KodakDataset(tmpdir, download=True)
