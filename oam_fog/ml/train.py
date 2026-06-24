import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

_EPS     = 1e-8
W_L1     = 1.0
W_SHAPE  = 0.3


def _loss(p_l1, shape, spectra, l1_idx, other_idx):
    """Combined MSE (l=+1 power) + KL-div (shape of remaining modes)."""
    true_l1 = spectra[:, l1_idx]
    loss_l1 = F.mse_loss(p_l1, true_l1)

    true_other = spectra[:, other_idx]
    true_shape = true_other / true_other.sum(dim=1, keepdim=True).clamp_min(_EPS)
    loss_shape = F.kl_div(torch.log(shape + _EPS), true_shape, reduction="batchmean")

    return W_L1 * loss_l1 + W_SHAPE * loss_shape


def train_epoch(model, loader, optimizer, device):
    model.train()
    running = 0.0
    for intensities, spectra, visibility in loader:
        intensities = intensities.to(device)
        spectra     = spectra.to(device)
        visibility  = visibility.to(device)
        optimizer.zero_grad()
        full, p_l1, shape = model(intensities, visibility)
        loss = _loss(p_l1, shape, spectra, model.l1_idx, model.other_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running += loss.item()
    return running / len(loader)


def evaluate(model, loader, device):
    model.eval()
    running      = 0.0
    correlations = []
    l1_true, l1_pred = [], []

    with torch.no_grad():
        for intensities, spectra, visibility in loader:
            intensities = intensities.to(device)
            spectra     = spectra.to(device)
            visibility  = visibility.to(device)
            full, p_l1, shape = model(intensities, visibility)
            running += _loss(p_l1, shape, spectra, model.l1_idx, model.other_idx).item()

            full_np    = full.cpu().numpy()
            spectra_np = spectra.cpu().numpy()
            for i in range(full_np.shape[0]):
                r, _ = pearsonr(full_np[i], spectra_np[i])
                if not np.isnan(r):
                    correlations.append(r)

            l1_true.extend(spectra[:, model.l1_idx].cpu().numpy().tolist())
            l1_pred.extend(p_l1.cpu().numpy().tolist())

    mean_loss    = running / len(loader)
    mean_pearson = float(np.mean(correlations)) if correlations else float("nan")
    l1_r, _      = pearsonr(l1_true, l1_pred)
    return mean_loss, mean_pearson, l1_r