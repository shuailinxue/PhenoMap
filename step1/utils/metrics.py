
import os
from typing import List

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class LossCurveCallback(Callback):

    def __init__(self, save_path: str = "loss_curve.png") -> None:
        super().__init__()
        self.save_path = save_path
        self.train_losses: List[float] = []

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            loss = metrics["train_loss"].detach().cpu().item()
            self.train_losses.append(loss)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        parent_dir = os.path.dirname(self.save_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        plt.figure()
        plt.plot(self.train_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.save_path, dpi=300)
        plt.close()
        print(f"Saved loss curve: {os.path.abspath(self.save_path)}")
