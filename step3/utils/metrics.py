import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import glob
import os

def compute_classification_metrics(logits, hazard_scores, threshold=0.0):
    with torch.no_grad():
        preds = torch.argmax(logits, dim=-1).view(-1)
        gt_labels = (hazard_scores > threshold).long().view(-1)

        tp = ((preds == 1) & (gt_labels == 1)).sum().float()
        fp = ((preds == 1) & (gt_labels == 0)).sum().float()
        fn = ((preds == 0) & (gt_labels == 1)).sum().float()
        tn = ((preds == 0) & (gt_labels == 0)).sum().float()

        epsilon = 1e-7
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        acc = (tp + tn) / (tp + tn + fp + fn + epsilon)

        return {
            'f1': f1.item(),
            'acc': acc.item()
        }


def get_gmm_peaks_from_scores(scores):
    print("Fitting a two-component GMM to hazard scores...")
    all_scores = np.asarray(scores, dtype=np.float32).reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, covariance_type='spherical', random_state=42)
    gmm.fit(all_scores)

    means = gmm.means_.flatten()
    peak_neg = min(means)
    peak_pos = max(means)

    print(f"Estimated hazard modes: low={peak_neg:.3f}, high={peak_pos:.3f}")
    return peak_neg, peak_pos

def get_latest_checkpoint(checkpoint_dir):
    list_of_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file
