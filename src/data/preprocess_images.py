"""
preprocess_images.py — Resize and normalize OpenI PNG images for the CNN.

Transform pipeline:
  1. Load PNG (convert to RGB — handles grayscale CXRs)
  2. Resize to 224×224
  3. ToTensor → [0, 1] float32
  4. Normalize with ImageNet mean/std

Assertions on first batch:
  - tensor.shape == (batch_size, 3, 224, 224)
  - tensor.dtype == torch.float32
"""

import os
import logging
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class OpenIDataset(Dataset):
    """PyTorch dataset that loads OpenI PNG images and optional labels."""

    def __init__(self, studies_df: pd.DataFrame, label_cols: list[str] | None = None, transform=None):
        self.studies = studies_df.reset_index(drop=True)
        self.label_cols = label_cols
        self.transform = transform or TRANSFORM

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        img_path = self.studies.at[idx, "image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}. Using blank tensor.")
            image = Image.new("RGB", (256, 256), color=128)
            
        img_tensor = self.transform(image)
        
        if self.label_cols:
            labels = torch.tensor(self.studies.loc[idx, self.label_cols].values.astype(float), dtype=torch.float32)
            return img_tensor, labels
            
        return img_tensor


def verify_preprocessing(
    processed_dir: str = "data/processed",
    batch_size: int = 4,
) -> torch.Tensor:
    """Load studies, build a DataLoader, assert the first batch is correct."""
    studies = pd.read_csv(os.path.join(processed_dir, "studies.csv"))
    logger.info(f"Loaded {len(studies)} studies for preprocessing check")

    dataset = OpenIDataset(studies)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    first_batch = next(iter(loader))

    # ── Assertions ────────────────────────────────────────────────────────────
    assert first_batch.shape == (batch_size, 3, 224, 224), \
        f"Bad shape: {first_batch.shape}"
    assert first_batch.dtype == torch.float32, \
        f"Bad dtype: {first_batch.dtype}"

    logger.info(f"Preprocessing OK: shape={first_batch.shape}, dtype={first_batch.dtype}")
    return first_batch


if __name__ == "__main__":
    batch = verify_preprocessing()
    print(f"First batch stats: min={batch.min():.3f}, max={batch.max():.3f}, mean={batch.mean():.3f}")
