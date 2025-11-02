from dataclasses import asdict
from pathlib import Path
import math
import json
import logging

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
import lightning as L
from lightning.fabric import Fabric
import tyro
from torchinfo import summary
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor

from perception.configs import TrainConfig, train_configs
from perception.dataset import SynthesiaDataset
from perception.models import model_registry
from perception.utils import keypoints_to_heatmaps


class Trainer:
    def __init__(self, cfg: TrainConfig, log_dir: Path, ckpt_dir: Path, log_writer: SummaryWriter):
        self.cfg = cfg
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.log_writer = log_writer

        # Create Datasets and Data Loaders
        processor = AutoImageProcessor.from_pretrained(cfg.backbone)
        dataset = SynthesiaDataset(
            data_dir=cfg.data_dir,
            processor=processor,
        )

        train_size = int((1 - cfg.val_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        logging.info(f"Dataset split: Train ({len(train_dataset)} samples), Val ({len(val_dataset)} samples)")

        train_vis_dataset = Subset(train_dataset, torch.randperm(len(train_dataset))[: cfg.n_val_vis])

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.n_workers,
            pin_memory=cfg.pin_mem,
        )

        self.train_vis_dataloader = DataLoader(
            train_vis_dataset,
            batch_size=cfg.n_val_vis,
            shuffle=False,
            num_workers=cfg.n_workers,
            pin_memory=cfg.pin_mem,
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.n_workers,
            pin_memory=cfg.pin_mem,
        )

        # Create Model and Optimizer
        self.model = model_registry[cfg.model][1](
            n_keypoints=cfg.n_keypoints,
            backbone=cfg.backbone,
        )

        logging.info(f"Using model: {cfg.model} with backbone: {cfg.backbone}")

        if cfg.freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            logging.info("Using frozen backbone.")

        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.print_model()

        # Setup Fabric
        self.fabric = Fabric(accelerator=cfg.accelerator, devices=cfg.devices)
        self.model, self.optim = self.fabric.setup(self.model, self.optim)
        self.train_dataloader, self.train_vis_dataloader, self.val_dataloader = self.fabric.setup_dataloaders(
            self.train_dataloader, self.train_vis_dataloader, self.val_dataloader
        )

        # Load Checkpoint
        self.start_epoch = 1
        if cfg.ckpt is not None:
            rest = self.fabric.load(cfg.ckpt, dict(model=self.model, optim=self.optim))
            self.start_epoch = rest["epoch"] + 1  # don't train twice

    def print_model(self):
        summary(self.model)

    def log_param_histograms(self, step: int):
        total_param_norm, n_param_norm = 0.0, 0
        total_grad_norm, n_grad_norm = 0.0, 0

        for name, param in self.model.named_parameters():
            w = param.detach()
            total_param_norm += (w**2).sum().item()
            n_param_norm += w.numel()

            self.log_writer.add_histogram(f"{name}/param", w.cpu(), step)

            if param.grad is not None:
                g = param.grad.detach()
                total_grad_norm += (g**2).sum().item()
                n_grad_norm += g.numel()

                self.log_writer.add_histogram(f"{name}/grad", g.cpu(), step)

        avg_param_norm = total_param_norm / n_param_norm
        avg_grad_norm = total_grad_norm / n_grad_norm

        self.log_writer.add_scalar("train/avg_param_norm", avg_param_norm, step)
        self.log_writer.add_scalar("train/avg_grad_norm", avg_grad_norm, step)

    def forward_loss(self, kpts: Tensor, pred_heatmaps: Tensor, epoch: int) -> tuple[Tensor, float]:
        t = epoch / self.cfg.n_epochs

        # linear decay
        heatmap_sigma = self.cfg.heatmap_sigma[0] * (1 - t) + self.cfg.heatmap_sigma[1] * t
        target_heatmaps = keypoints_to_heatmaps(kpts, self.cfg.input_size, heatmap_sigma)

        return F.mse_loss(pred_heatmaps, target_heatmaps), heatmap_sigma

    def train_epoch(self, epoch: int):
        it = tqdm(self.train_dataloader, unit="batch", desc=f"Epoch {epoch}/{self.cfg.n_epochs}")
        for i, (images, kpts) in enumerate(it):
            self.optim.zero_grad()

            pred_heatmaps, _ = self.model(images)
            loss, heatmap_sigma = self.forward_loss(kpts, pred_heatmaps, epoch)

            self.fabric.backward(loss)
            self.optim.step()

            step = (epoch - 1) * len(self.train_dataloader) + i
            self.log_writer.add_scalar("train/loss", loss.item(), step)
            it.set_postfix(dict(loss=loss.item()))
            self.log_writer.add_scalar("train/lr", self.optim.param_groups[0]["lr"], step)
            self.log_writer.add_scalar("train/heatmap_sigma", heatmap_sigma, step)
            self.log_writer.add_scalar("train/epoch", epoch, step)

            if step % self.cfg.param_hist_every_n_steps == 0:
                self.log_param_histograms(step)

    def plot_kpts(self, kpts: Tensor, pred_kpts: Tensor, pred_heatmaps: Tensor) -> plt.Figure:
        combined_heatmap = pred_heatmaps.sum(dim=0).cpu().numpy()

        plt.imshow(combined_heatmap, cmap="magma", interpolation="nearest")
        plt.scatter(kpts[:, 0].cpu(), kpts[:, 1].cpu(), c="green", s=3, label="Target Keypoints")
        plt.scatter(pred_kpts[:, 0].cpu(), pred_kpts[:, 1].cpu(), c="yellow", s=3, label="Predicted Keypoints")

        return plt.gcf()

    def val_epoch(self, epoch: int):
        self.model.eval()

        vis_indices_rng = torch.Generator()
        vis_indices_rng.manual_seed(0)
        vis_indices = torch.randperm(len(self.val_dataloader), generator=vis_indices_rng)[: self.cfg.n_val_vis]

        train_vis_images, train_vis_kpts = next(iter(self.train_vis_dataloader))

        with torch.no_grad():
            train_pred_heatmaps, train_pred_kpts = self.model(train_vis_images)

        for i in range(train_vis_images.shape[0]):
            target_kpts = train_vis_kpts[i]
            pred_kpts = train_pred_kpts[i]
            pred_heatmaps = train_pred_heatmaps[i]

            kpts_fig = self.plot_kpts(target_kpts, pred_kpts, pred_heatmaps)
            self.log_writer.add_figure(f"train/{i}.kpts", kpts_fig, epoch)
            plt.close(kpts_fig)

        total_loss = 0.0
        i_vis = 0
        with torch.no_grad():
            it = tqdm(self.val_dataloader, unit="batch", desc=f"Val Epoch {epoch}/{self.cfg.n_epochs}")
            for i, (images, kpts) in enumerate(it):
                val_pred_heatmaps, val_pred_kpts = self.model(images)
                total_loss += self.forward_loss(kpts, val_pred_heatmaps, epoch)[0].item()

                if i in vis_indices:
                    # Visualize first sample in selected batch
                    target_kpts = kpts[0]
                    pred_kpts = val_pred_kpts[0]
                    pred_heatmaps = val_pred_heatmaps[0]

                    kpts_fig = self.plot_kpts(target_kpts, pred_kpts, pred_heatmaps)
                    self.log_writer.add_figure(f"val/{i}.kpts", kpts_fig, epoch)
                    plt.close(kpts_fig)

                    i_vis += 1

        avg_loss = total_loss / len(self.val_dataloader)
        self.log_writer.add_scalar("val/loss", avg_loss, epoch)
        logging.info(f"Epoch {epoch}: Validation Loss = {avg_loss:.4f}")
        self.model.train()

    def save_epoch(self, epoch: int):
        epoch_str = str(epoch).zfill(int(math.log10(self.cfg.n_epochs)) + 1)
        ckpt_path = self.ckpt_dir / f"ckpt_{epoch_str}.pth"
        self.fabric.save(ckpt_path, dict(cfg=asdict(self.cfg), model=self.model, optim=self.optim, epoch=epoch))
        logging.info(f"Epoch {epoch}: Saved checkpoint to {ckpt_path}")

    def run(self):
        self.save_epoch(0)  # Initial checkpoint before training
        self.val_epoch(0)  # Initial validation before training

        for epoch in range(self.start_epoch, self.cfg.n_epochs + 1):  # Indexing starts at 1
            self.train_epoch(epoch)

            if epoch % self.cfg.save_every_n_epochs == 0 or epoch == self.cfg.n_epochs:
                self.save_epoch(epoch)

            if epoch % self.cfg.val_every_n_epochs == 0 or epoch == self.cfg.n_epochs:
                self.val_epoch(epoch)


def entrypoint():
    train_cfg = tyro.extras.overridable_config_cli(train_configs)

    if train_cfg.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # Set random seed for reproducibility
    L.seed_everything(train_cfg.seed)

    # Create logging locations
    train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    train_log_dir = train_cfg.output_dir / "logs"
    train_ckpt_dir = train_cfg.output_dir / "checkpoints"
    train_log_dir.mkdir(exist_ok=True)
    train_ckpt_dir.mkdir(exist_ok=True)

    # Get TensorBoard file name
    train_log_writer = SummaryWriter(log_dir=str(train_log_dir))
    log_name = train_log_writer.file_writer.event_writer._file_name.split("/")[-1].replace("events.out.tfevents", "")

    # Save config
    json.dump(
        asdict(train_cfg),
        open(train_cfg.output_dir / f"config{log_name}.txt", "w"),
        indent=4,
        default=lambda x: str(x.resolve()) if isinstance(x, Path) else x,
    )

    # Setup logging
    fh = logging.FileHandler(train_cfg.output_dir / f"train{log_name}.log")
    sh = logging.StreamHandler()

    formatter = logging.Formatter("PID %(process)d - %(filename)s:%(lineno)s | %(levelname)s | %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # Check for TF32 support
    if torch.cuda.is_available() and torch.cuda.is_tf32_supported() and train_cfg.allow_tf32:
        torch.set_float32_matmul_precision("high")
        logging.warning("Enabled TF32. Turn this off if there are precision issues.")

    trainer = Trainer(train_cfg, train_log_dir, train_ckpt_dir, train_log_writer)
    trainer.run()


if __name__ == "__main__":
    entrypoint()
