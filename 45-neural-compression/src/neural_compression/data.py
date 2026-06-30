"""Data loading utilities for neural compression.

This module implements:
- ImageFolderDataset: Load images from directory
- RandomCropDataset: Random crop augmentation
- Kodak dataset loader
"""

import os
import random
from typing import Tuple, Optional, List, Callable, Union
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ImageFolderDataset(Dataset):
    """Dataset for loading images from a folder.

    Supports PNG, JPEG, and other common image formats.

    Args:
        root: Root directory containing images
        transform: Optional transform to apply
        extensions: List of valid file extensions
        recursive: Whether to search subdirectories
    """

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ):
        self.root = Path(root)
        self.transform = transform

        if extensions is None:
            extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
        self.extensions = [ext.lower() for ext in extensions]

        # Find all image files
        self.image_paths: List[Path] = []
        if recursive:
            for ext in self.extensions:
                self.image_paths.extend(self.root.rglob(f"*{ext}"))
        else:
            for ext in self.extensions:
                self.image_paths.extend(self.root.glob(f"*{ext}"))

        self.image_paths = sorted(self.image_paths)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and return an image.

        Args:
            idx: Index

        Returns:
            Image tensor [C, H, W] in range [0, 1]
        """
        path = self.image_paths[idx]

        # Load image
        image = self._load_image(path)

        if self.transform is not None:
            image = self.transform(image)

        return image

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load image from path.

        Args:
            path: Path to image

        Returns:
            Image tensor [C, H, W]
        """
        try:
            from PIL import Image

            img = Image.open(path).convert("RGB")
            img = np.array(img, dtype=np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            return img
        except ImportError:
            # Fallback to cv2
            import cv2

            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            return img


class RandomCropTransform:
    """Random crop transform for images.

    Args:
        size: Crop size (height, width) or int for square
    """

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random crop.

        Args:
            image: Input tensor [C, H, W]

        Returns:
            Cropped tensor [C, crop_h, crop_w]
        """
        _, h, w = image.shape
        crop_h, crop_w = self.size

        if h < crop_h or w < crop_w:
            # Pad if necessary
            pad_h = max(0, crop_h - h)
            pad_w = max(0, crop_w - w)
            # Use constant padding for large padding amounts since reflect
            # requires padding < input dimension
            if pad_h >= h or pad_w >= w:
                image = torch.nn.functional.pad(
                    image, (0, pad_w, 0, pad_h), mode="constant", value=0
                )
            else:
                image = torch.nn.functional.pad(
                    image, (0, pad_w, 0, pad_h), mode="reflect"
                )
            _, h, w = image.shape

        # Random crop
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        return image[:, top : top + crop_h, left : left + crop_w]


class CenterCropTransform:
    """Center crop transform for images.

    Args:
        size: Crop size (height, width) or int for square
    """

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply center crop.

        Args:
            image: Input tensor [C, H, W]

        Returns:
            Cropped tensor [C, crop_h, crop_w]
        """
        _, h, w = image.shape
        crop_h, crop_w = self.size

        top = (h - crop_h) // 2
        left = (w - crop_w) // 2

        return image[:, top : top + crop_h, left : left + crop_w]


class ComposeTransform:
    """Compose multiple transforms.

    Args:
        transforms: List of transforms
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            image = t(image)
        return image


class RandomHorizontalFlip:
    """Random horizontal flip.

    Args:
        p: Probability of flip
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return torch.flip(image, dims=[2])
        return image


class KodakDataset(Dataset):
    """Kodak test dataset (24 images).

    Standard test set for image compression evaluation.

    Args:
        root: Root directory (will download if needed)
        download: Whether to download if not present
    """

    KODAK_URLS = [
        f"http://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png" for i in range(1, 25)
    ]

    def __init__(self, root: Union[str, Path], download: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.image_paths = [self.root / f"kodim{i:02d}.png" for i in range(1, 25)]

        if download:
            self._download()

        # Check all images exist
        missing = [p for p in self.image_paths if not p.exists()]
        if missing:
            raise ValueError(f"Missing Kodak images: {missing}")

    def _download(self):
        """Download Kodak dataset."""
        try:
            import urllib.request

            for url, path in zip(self.KODAK_URLS, self.image_paths):
                if not path.exists():
                    print(f"Downloading {url}...")
                    urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f"Warning: Could not download Kodak images: {e}")

    def __len__(self) -> int:
        return 24

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load Kodak image.

        Args:
            idx: Index (0-23)

        Returns:
            Image tensor [3, 512, 768]
        """
        try:
            from PIL import Image

            img = Image.open(self.image_paths[idx]).convert("RGB")
            img = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(img).permute(2, 0, 1)
        except ImportError:
            import cv2

            img = cv2.imread(str(self.image_paths[idx]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            return torch.from_numpy(img).permute(2, 0, 1)


def create_dataloaders(
    train_dir: str,
    val_dir: Optional[str] = None,
    batch_size: int = 8,
    crop_size: int = 256,
    num_workers: int = 4,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create train and validation dataloaders.

    Args:
        train_dir: Training images directory
        val_dir: Optional validation images directory
        batch_size: Batch size
        crop_size: Random crop size for training
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_transform = ComposeTransform(
        [
            RandomCropTransform(crop_size),
            RandomHorizontalFlip(0.5),
        ]
    )

    train_dataset = ImageFolderDataset(train_dir, transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_dir is not None:
        val_transform = CenterCropTransform(crop_size)
        val_dataset = ImageFolderDataset(val_dir, transform=val_transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader
