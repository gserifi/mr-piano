from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import AutoImageProcessor
import numpy as np
from PIL import Image


class SynthesiaDataset(Dataset):
    """
    Dataset for Synthesia images and keypoints.
    """

    def __init__(self, data_dir: Path, processor: AutoImageProcessor):
        """
        :param data_dir: Path to the directory containing images and keypoints
        :param processor: Image processor given by the backbone
        """

        self.processor = processor

        self.images = list(sorted(data_dir.glob("*.png")))
        self.keypoints = list(sorted(data_dir.glob("*.npy")))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        :param idx: Index of the sample

        :return: Tuple of (image, keypoints)
            - image: Tensor of shape (3, H, W)
            - keypoints: Tensor of shape (N, 2)
        """

        image = Image.open(self.images[idx]).convert("RGB")
        keypoints = np.load(self.keypoints[idx], allow_pickle=True).item()["keypoints2d"]

        image = self.processor(images=image, do_resize=False, return_tensors="pt")["pixel_values"].squeeze(0)
        keypoints = torch.from_numpy(keypoints)

        return image.to(torch.float32), keypoints.to(torch.float32)


if __name__ == "__main__":
    backbone = "facebook/dinov3-vits16-pretrain-lvd1689m"
    dataset = SynthesiaDataset(
        data_dir=Path.cwd().parent / "outputs",
        processor=AutoImageProcessor.from_pretrained(backbone),
    )

    image, keypoints = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Keypoints shape: {keypoints.shape}")
