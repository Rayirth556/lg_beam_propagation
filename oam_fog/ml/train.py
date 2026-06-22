"""
Training and evaluation functions for OAMNet.

Loss: KL divergence (batchmean) between log-predictions and true spectra.
  kl_div requires log-space input → we apply log(softmax_output + EPS).
  EPS guards against log(0) when softmax outputs are near zero.

Metric: mean per-sample Pearson correlation between predicted and true
  11-element spectra. NaN values (constant spectrum edge case) are excluded.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

_EPS = 1e-8  # numerical stability for log of softmax outputs


def train_epoch(model, loader, optimizer, device) -> float:
    """
    One full pass over the training DataLoader.

    Args:
        model:     OAMNet (or any nn.Module with matching I/O)
        loader:    DataLoader yielding (intensity, spectrum) batches
        optimizer: torch optimizer
        device:    torch.device

    Returns:
        mean KL divergence loss over all batches
    """
    model.train()
    running_loss = 0.0

    for intensities, spectra in loader:
        intensities = intensities.to(device)   # (B, 1, 128, 128)
        spectra = spectra.to(device)           # (B, 11)

        optimizer.zero_grad()
        preds = model(intensities)             # (B, 11), softmax output
        log_preds = torch.log(preds + _EPS)   # (B, 11), log for kl_div
        loss = F.kl_div(log_preds, spectra, reduction="batchmean")
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, device) -> Tuple[float, float]:
    """
    Evaluate the model on a DataLoader without gradient tracking.

    Args:
        model:  OAMNet (or compatible)
        loader: DataLoader yielding (intensity, spectrum) batches
        device: torch.device

    Returns:
        (mean_loss, mean_pearson_r)
          mean_loss     — mean KL divergence over all batches
          mean_pearson_r — mean per-sample Pearson correlation between
                          predicted and true 11-element spectra;
                          NaN samples (constant spectrum) are excluded
    """
    model.eval()
    running_loss = 0.0
    correlations = []

    with torch.no_grad():
        for intensities, spectra in loader:
            intensities = intensities.to(device)
            spectra = spectra.to(device)

            preds = model(intensities)             # (B, 11)
            log_preds = torch.log(preds + _EPS)
            loss = F.kl_div(log_preds, spectra, reduction="batchmean")
            running_loss += loss.item()

            preds_np = preds.cpu().numpy()         # (B, 11)
            spectra_np = spectra.cpu().numpy()     # (B, 11)

            for i in range(preds_np.shape[0]):
                r, _ = pearsonr(preds_np[i], spectra_np[i])
                if not np.isnan(r):
                    correlations.append(float(r))

    mean_loss = running_loss / len(loader)
    mean_pearson = float(np.mean(correlations)) if correlations else float("nan")
    return mean_loss, mean_pearson
