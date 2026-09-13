
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from ..utils.metrics import LossCurveCallback


@dataclass
class TrainerConfig:

    max_epochs: int = 500
    accelerator: str = "gpu"
    devices: List[int] = field(default_factory=lambda: [0])
    log_every_n_steps: int = 1
    loss_curve_save_path: Optional[str] = "loss_curve.png"
    extra_callbacks: List[pl.Callback] = field(default_factory=list)


def run_training(
    model: pl.LightningModule,
    train_loader: DataLoader,
    config: TrainerConfig,
) -> pl.LightningModule:
    callbacks: List[pl.Callback] = list(config.extra_callbacks)

    if config.loss_curve_save_path is not None:
        callbacks.append(LossCurveCallback(save_path=config.loss_curve_save_path))

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        log_every_n_steps=config.log_every_n_steps,
        callbacks=callbacks,
    )

    trainer.fit(model, train_loader)
    return model


def run_prediction(
    model: pl.LightningModule,
    predict_loader: DataLoader,
    gene_names: List[str],
    scaler: MinMaxScaler,
    config: TrainerConfig,
) -> pd.DataFrame:
    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=config.devices,
    )

    raw_results: List[np.ndarray] = trainer.predict(model, predict_loader)

    if not raw_results:
        raise ValueError("Prediction returned no batches")

    final_predictions = np.concatenate(raw_results, axis=0)

    dataset = predict_loader.dataset
    all_cell_ids: List[str] = []
    for i in range(len(dataset)):
        all_cell_ids.extend(dataset.get_cell_ids(i))

    if final_predictions.shape[0] != len(all_cell_ids):
        raise ValueError(
            f"Prediction rows ({final_predictions.shape[0]}) do not match "
            f"the number of cell IDs ({len(all_cell_ids)})"
        )

    denorm_predictions = scaler.inverse_transform(final_predictions)

    return pd.DataFrame(
        denorm_predictions,
        index=all_cell_ids,
        columns=gene_names,
    )
