import os
import argparse
import warnings
import logging
import glob
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from step3.models import PhenoMapModel, KEEPVisionEncoder, KEEPTextEncoder, ScGPTEncoder
from step3.utils.config import load_config, validate_config
from step3.data import (
    PhenotypeSourceDataset,
    InferenceDataset,
    TargetAnnotationDataset,
    TargetAnnotationFolderDataset,
)
from step3.utils.metrics import (
    compute_classification_metrics,
    get_gmm_peaks_from_scores,
    get_latest_checkpoint,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("scGPT").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")


TIGER_LABEL_COLORS = np.array([
    [0, 0, 0],        # 0 background / unlabeled
    [230, 25, 75],    # 1 invasive tumor
    [60, 180, 75],    # 2 tumor-associated stroma
    [255, 225, 25],   # 3 in-situ tumor
    [0, 130, 200],    # 4 healthy glands
    [245, 130, 48],   # 5 necrosis not in-situ
    [145, 30, 180],   # 6 inflamed stroma
    [70, 240, 240],   # 7 rest
], dtype=np.uint8)


class JointSourceTargetDataset(Dataset):
    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        if len(self.target_dataset) == 0:
            raise ValueError("TargetAnnotationDataset is empty; check target_data paths and tissue_thresh")

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, idx):
        return {
            'source': self.source_dataset[idx],
            'target': self.target_dataset[idx % len(self.target_dataset)]
        }


class PhenoMapDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gmm_peak_neg = None
        self.gmm_peak_pos = None
        self.target_dataset = None

    def setup(self, stage=None):
        self.train_dataset = PhenotypeSourceDataset(self.config, is_train=True)
        print(f"Train: {len(self.train_dataset)} patches | No validation during training")

        target_cfg = self.config.get('target_data', {})
        if target_cfg.get('enabled', False):
            if target_cfg.get('he_folder') and target_cfg.get('mask_folder'):
                self.target_dataset = TargetAnnotationFolderDataset(self.config)
            else:
                self.target_dataset = TargetAnnotationDataset(self.config)
            print(f"Joint training enabled: source + target ({len(self.target_dataset)} target patches)")

        hazard_scores = self.train_dataset.df.iloc[:, 0].values
        self.gmm_peak_neg, self.gmm_peak_pos = get_gmm_peaks_from_scores(hazard_scores)

    def train_dataloader(self):
        train_dataset = self.train_dataset
        if self.target_dataset is not None:
            train_dataset = JointSourceTargetDataset(self.train_dataset, self.target_dataset)
        return DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True, drop_last=True, num_workers=self.config['training'].get('num_workers', 4), pin_memory=True
        )


class PhenoMapLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.phenomap_model = PhenoMapModel(config)

        self.vision_encoder = KEEPVisionEncoder()
        self.vision_encoder.requires_grad_(False)

        self.disable_text_similarity = config['model'].get('disable_text_similarity', False)
        self.text_encoder = None
        if not self.disable_text_similarity:
            self.text_encoder = KEEPTextEncoder()
            self.text_encoder.requires_grad_(False)

        self.gene_encoder = ScGPTEncoder(config)
        self.gene_encoder.requires_grad_(False)

        self.register_buffer('v_high', torch.zeros(1, config['model']['keep_text_dim']))
        self.register_buffer('v_low',  torch.zeros(1, config['model']['keep_text_dim']))
        self.global_peak_neg = 0.0
        self.global_peak_pos = 0.0


    def on_fit_start(self):
        if not self.disable_text_similarity:
            high_risk_text = self.config['model']['text_prompts']['high_risk']
            low_risk_text  = self.config['model']['text_prompts']['low_risk']
            self.text_encoder.eval()
            with torch.no_grad():
                self.v_high = self.text_encoder(high_risk_text).to(self.device)
                self.v_low  = self.text_encoder(low_risk_text).to(self.device)

        self.gene_encoder = self.gene_encoder.to(self.device)

        dm = self.trainer.datamodule
        self.global_peak_neg = dm.gmm_peak_neg
        self.global_peak_pos = dm.gmm_peak_pos


    def _get_thresholds(self):
        num_epochs  = self.config['training']['num_epochs']
        stage1_end  = int(num_epochs * 0.2)
        stage3_start = int(num_epochs * 0.8)
        epoch = self.current_epoch

        if epoch <= stage1_end:
            decay_ratio = 0.0
        elif epoch >= stage3_start:
            decay_ratio = 1.0
        else:
            decay_ratio = (epoch - stage1_end) / (stage3_start - stage1_end)

        thresh_neg = self.global_peak_neg * (1.0 - decay_ratio)
        thresh_pos = self.global_peak_pos * (1.0 - decay_ratio)
        return thresh_neg, thresh_pos


    def _encode(self, patches, coords, gene_ids=None, values=None):
        self.vision_encoder.eval()
        with torch.no_grad():
            f_vis = self.vision_encoder(patches).unsqueeze(1)  # [B,1,768]
            if gene_ids is not None and values is not None:
                self.gene_encoder.eval()
                e_gene = self.gene_encoder.extract_gene_features(gene_ids, values).unsqueeze(1)
            else:
                e_gene = None
        coords_exp = coords.unsqueeze(1)
        return f_vis, coords_exp, e_gene


    def _source_training_loss(self, batch):
        patches           = batch['patch']
        coords            = batch['coords']
        hazard_scores     = batch['hazard_score']
        gene_ids          = batch['gene_ids']
        values            = batch['values']

        source_filter = self.config.get('soft_label_tuning', {}).get('source_filter', 'none')
        if source_filter == 'gmm':
            thresh_neg, thresh_pos = self._get_thresholds()
            valid_mask = (hazard_scores <= thresh_neg) | (hazard_scores >= thresh_pos)
            if valid_mask.sum() == 0:
                return None  # skip fully-filtered batches

            patches           = patches[valid_mask]
            coords            = coords[valid_mask]
            hazard_scores     = hazard_scores[valid_mask]
            gene_ids          = gene_ids[valid_mask]
            values            = values[valid_mask]
        elif source_filter != 'none':
            raise ValueError(f"Unsupported soft_label_tuning.source_filter: {source_filter}")

        f_vis, coords_exp, e_gene = self._encode(patches, coords, gene_ids, values)

        logits, loss_total, loss_dict = self.phenomap_model(
            f_vis=f_vis,
            coords=coords_exp,
            e_scgpt=e_gene,
            v_high=self.v_high,
            v_low=self.v_low,
            hazard_scores=hazard_scores.unsqueeze(1),
            label_mode='hazard',
            use_align=True,
        )

        metrics = compute_classification_metrics(logits, hazard_scores)

        self.log('train_source_loss', loss_total, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_source_f1', metrics['f1'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_source_acc', metrics['acc'], on_step=False, on_epoch=True)
        self.log('train_source_loss_align', loss_dict['loss_align'], on_step=True, on_epoch=False)
        self.log('train_source_loss_ce', loss_dict['loss_ce'], on_step=True, on_epoch=False)

        return loss_total

    def _target_training_loss(self, batch):
        patches = batch['patch']
        coords = batch['coords']
        risk_targets = batch['risk_target']

        f_vis, coords_exp, _ = self._encode(patches, coords)

        logits, loss_total, loss_dict = self.phenomap_model(
            f_vis=f_vis,
            coords=coords_exp,
            e_scgpt=None,
            v_high=self.v_high,
            v_low=self.v_low,
            risk_targets=risk_targets.unsqueeze(1),
            label_mode='prob',
            use_align=False,
        )

        metrics = compute_classification_metrics(logits, risk_targets, threshold=0.5)
        self.log('train_target_loss', loss_total, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_target_f1', metrics['f1'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_target_acc', metrics['acc'], on_step=False, on_epoch=True)
        self.log('train_target_loss_ce', loss_dict['loss_ce'], on_step=True, on_epoch=False)

        return loss_total

    def training_step(self, batch, batch_idx):
        if 'source' in batch and 'target' in batch:
            source_loss = self._source_training_loss(batch['source'])
            target_loss = self._target_training_loss(batch['target'])
            target_weight = self.config.get('target_data', {}).get('loss_weight', 1.0)

            if source_loss is None:
                loss_total = target_weight * target_loss
            else:
                loss_total = source_loss + target_weight * target_loss
            self.log('train_loss', loss_total, on_step=True, on_epoch=True, prog_bar=True)
            return loss_total

        loss_total = self._source_training_loss(batch)
        if loss_total is not None:
            self.log('train_loss', loss_total, on_step=True, on_epoch=True, prog_bar=True)
        return loss_total


    def configure_optimizers(self):
        return torch.optim.Adam(
            self.phenomap_model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )



def run_inference(model: PhenoMapLightningModule, dataloader: DataLoader):
    model.eval()
    all_risk_scores = []

    with torch.no_grad():
        for batch in dataloader:
            patches = batch['patch'].to(model.device)
            coords  = batch['coords'].to(model.device)

            with torch.no_grad():
                f_vis = model.vision_encoder(patches).unsqueeze(1)  # [B,1,768]
            coords_exp = coords.unsqueeze(1)                         # [B,1,2]

            logits = model.phenomap_model.predict_logits(
                f_vis,
                coords_exp,
                v_high=model.v_high,
                v_low=model.v_low,
            )
            risk = torch.softmax(logits, dim=-1)[..., 1].squeeze(-1) # [B]
            all_risk_scores.append(risk.cpu())

    return torch.cat(all_risk_scores) if all_risk_scores else torch.tensor([])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',      type=str, required=True, choices=['train', 'inference'])
    parser.add_argument('--config',    type=str, default='config.yaml')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Required for inference mode')
    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(config)

    torch.manual_seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))

    checkpoint_dir = config.get('training', {}).get('checkpoint_dir', './checkpoints')
    log_dir = config.get('training', {}).get('log_dir', './logs')
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='phenomap-{epoch:02d}',
        save_top_k=-1,
        save_last=True,
    )

    if args.mode == 'train':
        datamodule = PhenoMapDataModule(config)
        model      = PhenoMapLightningModule(config)

        logger = TensorBoardLogger(save_dir=log_dir, name='phenomap')

        device_str = config.get('device', 'cuda:0')
        if device_str.startswith('cuda'):
            accelerator = 'gpu'
            if ':' in device_str:
                device_idx = int(device_str.split(':')[1])
                devices_arg = [device_idx]
            else:
                devices_arg = 0
        else:
            accelerator = 'cpu'
            devices_arg = 'auto'

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices_arg,
            max_epochs=config['training']['num_epochs'],
            callbacks=[checkpoint_callback],
            logger=logger,
            log_every_n_steps=10,
            num_sanity_val_steps=0,
        )

        trainer.fit(model, datamodule=datamodule)

    elif args.mode == 'inference':
        if args.ckpt_path is None:
            default_dir = checkpoint_dir
            latest_ckpt = get_latest_checkpoint(default_dir)
            if latest_ckpt:
                print(f"No checkpoint provided. Loading latest: {latest_ckpt}")
                args.ckpt_path = latest_ckpt
            else:
                raise ValueError(f'--ckpt_path is None and no checkpoints found in {default_dir}')

        model = PhenoMapLightningModule.load_from_checkpoint(args.ckpt_path, strict=False)
        model.eval()
        device = int(config.get('device', 'cuda:0').split(':')[1])
        model = model.to(device)

        if not model.disable_text_similarity:
            model.text_encoder.eval()
            high_risk_text = config['model']['text_prompts']['high_risk']
            low_risk_text  = config['model']['text_prompts']['low_risk']
            with torch.no_grad():
                model.v_high = model.text_encoder(high_risk_text).to(model.device)
                model.v_low  = model.text_encoder(low_risk_text).to(model.device)

        he_folder = config['data']['he_folder']
        exts = ('*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg')
        he_files = []
        for ext in exts:
            he_files.extend(glob.glob(os.path.join(he_folder, ext)))
        he_files = sorted([f for f in he_files if '_mask' not in os.path.basename(f).lower()])
        if not he_files:
            raise ValueError(f"No image files found in {he_folder}")

        output_dir = config['data'].get('output_dir', os.path.join(he_folder, 'output'))
        os.makedirs(output_dir, exist_ok=True)
        skip_existing = config['data'].get('skip_existing', True)
        skip_existing_newer_than_ckpt = config['data'].get('skip_existing_newer_than_ckpt', False)
        print(f"Found {len(he_files)} images. Output -> {output_dir} (skip_existing={skip_existing})")

        vis_scale = config['data'].get('vis_scale', 0.25)  # downscale factor for output
        save_visualization = config['data'].get('save_visualization', True)
        save_patch_scores = config['data'].get('save_patch_scores', True)

        for he_path in he_files:
            stem = os.path.splitext(os.path.basename(he_path))[0]
            out_path = os.path.join(output_dir, f"{stem}_risk.png")
            score_map_path = os.path.join(output_dir, f"{stem}_score_map.npy")
            skip_path = out_path if save_visualization else score_map_path
            if skip_existing and os.path.exists(skip_path):
                if (
                    not skip_existing_newer_than_ckpt
                    or os.path.getmtime(skip_path) >= os.path.getmtime(args.ckpt_path)
                ):
                    print(f"Skipping {stem} (already done)")
                    continue
            print(f"Processing {stem} ...")

            mask_path = None
            mask_folder = config['data'].get('mask_folder', he_folder)
            for mask_suffix in ('', '_mask', '_tumor_mask', '_annotation'):
                for mask_ext in ('.tif', '.tiff', '.png', '.jpg'):
                    candidate = os.path.join(mask_folder, f"{stem}{mask_suffix}{mask_ext}")
                    if os.path.exists(candidate):
                        mask_path = candidate
                        print(f"  Found companion annotation: {os.path.basename(mask_path)}")
                        break
                if mask_path:
                    break

            dataset = InferenceDataset(config, he_path_override=he_path, mask_path=None)  # always full-slide inference
            loader  = DataLoader(
                dataset,
                batch_size=config.get('inference', {}).get('batch_size', config['training']['batch_size']),
                shuffle=False, num_workers=4, pin_memory=True
            )

            risk_scores = run_inference(model, loader)  # [N]
            coords_list = dataset.coords                # list of (x, y)

            if len(coords_list) == 0:
                print(f"  Skipping {stem}: no tissue patches found (try lowering tissue_thresh)")
                continue

            H, W = dataset.he_image.shape[:2]
            half = dataset.patch_size // 2

            scores_np = risk_scores.numpy()
            score_acc = np.zeros((H, W), dtype=np.float32)
            count_map = np.zeros((H, W), dtype=np.float32)
            for i, (x, y) in enumerate(coords_list):
                y0, y1 = y - half, y + half
                x0, x1 = x - half, x + half
                yy0, yy1 = max(0, y0), min(H, y1)
                xx0, xx1 = max(0, x0), min(W, x1)
                if yy0 >= yy1 or xx0 >= xx1:
                    continue
                score_acc[yy0:yy1, xx0:xx1] += scores_np[i]
                count_map[yy0:yy1, xx0:xx1] += 1
            valid = count_map > 0
            if not np.any(valid):
                print(f"  Skipping {stem}: no valid score pixels after patch placement")
                continue
            score_map = np.where(valid, score_acc / count_map, 0.0)

            smooth_sigma = config['data'].get('smooth_sigma', dataset.patch_size * 0.5)
            score_map = gaussian_filter(score_map, sigma=smooth_sigma)
            score_map = np.where(valid, score_map, 0.0)

            out_H = max(1, int(H * vis_scale))
            out_W = max(1, int(W * vis_scale))

            if save_visualization:
                vmin, vmax = score_map[valid].min(), score_map[valid].max()
                norm_map = np.clip((score_map - vmin) / (vmax - vmin + 1e-8), 0, 1)

                he_small = np.array(
                    Image.fromarray(dataset.he_image).resize((out_W, out_H), Image.LANCZOS)
                )
                norm_small = np.array(
                    Image.fromarray((norm_map * 255).astype(np.uint8)).resize((out_W, out_H), Image.LANCZOS)
                ) / 255.0

                colormap = cm.get_cmap('jet')
                heat_rgba = (colormap(norm_small) * 255).astype(np.uint8)
                heat_rgb  = heat_rgba[..., :3]
                alpha = config['data'].get('vis_alpha', 0.5)
                overlay = (he_small * (1 - alpha) + heat_rgb * alpha).astype(np.uint8)

                annotation_legend = None
                if mask_path is not None:
                    import tifffile as _tiff
                    if os.path.splitext(mask_path)[1].lower() in ('.tif', '.tiff'):
                        ann_raw = _tiff.imread(mask_path)
                    else:
                        ann_raw = np.array(Image.open(mask_path))
                    if ann_raw.ndim == 2:
                        ann_unique = np.unique(ann_raw)
                        if ann_unique.max() > 1:
                            ann_label = np.array(
                                Image.fromarray(ann_raw.astype(np.uint8)).resize((out_W, out_H), Image.NEAREST)
                            )
                            ann_color = TIGER_LABEL_COLORS[np.clip(ann_label, 0, len(TIGER_LABEL_COLORS) - 1)]
                            ann_alpha = (ann_label > 0).astype(np.float32) * 0.5
                            left_img = (
                                he_small * (1 - ann_alpha[..., None]) +
                                ann_color * ann_alpha[..., None]
                            ).astype(np.uint8)
                            left_title = 'TIGER tissue annotation'
                            label_names = {
                                1: 'invasive tumor',
                                2: 'tumor stroma',
                                3: 'in-situ tumor',
                                4: 'healthy glands',
                                5: 'necrosis',
                                6: 'inflamed stroma',
                                7: 'rest',
                            }
                            annotation_legend = [
                                mpatches.Patch(
                                    color=TIGER_LABEL_COLORS[i] / 255.0,
                                    label=label_names[i],
                                )
                                for i in sorted(set(ann_unique.tolist()) & set(label_names))
                            ]
                        else:
                            ann_bool = (ann_raw > 0).astype(np.uint8)
                            ann_resized = np.array(
                                Image.fromarray(ann_bool * 255).resize((out_W, out_H), Image.NEAREST)
                            ) / 255.0
                            ann_color = np.zeros((out_H, out_W, 3), dtype=np.uint8)
                            ann_color[..., 0] = (ann_resized * 255).astype(np.uint8)
                            left_img = (he_small * (1 - ann_resized[..., None] * 0.5) +
                                        ann_color * ann_resized[..., None] * 0.5).astype(np.uint8)
                            left_title = 'Annotation (red=positive)'
                    else:
                        ann_rgb = ann_raw[..., :3] if ann_raw.shape[2] >= 3 else ann_raw
                        left_img = np.array(
                            Image.fromarray(ann_rgb).resize((out_W, out_H), Image.LANCZOS)
                        )
                        left_title = 'Annotation'
                else:
                    left_img = he_small
                    left_title = 'HE Image'

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                axes[0].imshow(left_img)
                axes[0].set_title(left_title)
                axes[0].axis('off')
                if annotation_legend:
                    axes[0].legend(
                        handles=annotation_legend,
                        loc='lower left',
                        fontsize=7,
                        framealpha=0.75,
                    )
                axes[1].imshow(overlay)
                axes[1].set_title('Risk Score')
                axes[1].axis('off')
                sm = plt.cm.ScalarMappable(cmap='jet',
                                            norm=plt.Normalize(vmin=vmin, vmax=vmax))
                sm.set_array([])
                plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04, label='Risk Score')
                plt.tight_layout()
                plt.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved -> {out_path}")

            score_map_small = np.array(
                Image.fromarray(np.where(valid, score_map, 0.0).astype(np.float32))
                .resize((out_W, out_H), Image.LANCZOS)
            ).astype(np.float16)
            valid_small = np.array(
                Image.fromarray(valid.astype(np.uint8)).resize((out_W, out_H), Image.NEAREST)
            ).astype(bool)
            score_map_small[~valid_small] = np.nan
            np.save(score_map_path, score_map_small)

            if save_patch_scores:
                import pandas as pd
                df_out = pd.DataFrame({
                    'x': [c[0] for c in coords_list],
                    'y': [c[1] for c in coords_list],
                    'risk_score': scores_np,
                })
                df_out.to_csv(os.path.join(output_dir, f"{stem}_patch_scores.csv"), index=False)
                print(f"  Saved score_map.npy {score_map_small.shape} float16 and patch_scores.csv ({len(df_out)} patches)")
            else:
                print(f"  Saved score_map.npy {score_map_small.shape} float16")

        print("All done.")
