import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

L1_IDX = 6
OTHER_IDX = [i for i in range(11) if i != L1_IDX]
_EPS = 1e-8
W_L1, W_SHAPE = 1.0, 0.3

def _loss(p_l1, shape, spectra):
    true_l1 = spectra[:, L1_IDX]
    loss_l1 = F.mse_loss(p_l1, true_l1)

    true_other = spectra[:, OTHER_IDX]
    true_shape = true_other / true_other.sum(dim=1, keepdim=True).clamp_min(_EPS)
    loss_shape = F.kl_div(torch.log(shape + _EPS), true_shape, reduction="batchmean")

    return W_L1 * loss_l1 + W_SHAPE * loss_shape

def train_epoch(model, loader, optimizer, device):
    model.train()
    running = 0.0
    for intensities, spectra, visibility in loader:
        intensities, spectra, visibility = intensities.to(device), spectra.to(device), visibility.to(device)
        optimizer.zero_grad()
        full, p_l1, shape = model(intensities, visibility)
        loss = _loss(p_l1, shape, spectra)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / len(loader)

def evaluate(model, loader, device):
    model.eval()
    running, correlations, l1_true, l1_pred = 0.0, [], [], []
    with torch.no_grad():
        for intensities, spectra, visibility in loader:
            intensities, spectra, visibility = intensities.to(device), spectra.to(device), visibility.to(device)
            full, p_l1, shape = model(intensities, visibility)
            running += _loss(p_l1, shape, spectra).item()
            full_np, spectra_np = full.cpu().numpy(), spectra.cpu().numpy()
            for i in range(full_np.shape[0]):
                r, _ = pearsonr(full_np[i], spectra_np[i])
                if not np.isnan(r):
                    correlations.append(r)
            l1_true.extend(spectra[:, L1_IDX].cpu().numpy().tolist())
            l1_pred.extend(p_l1.cpu().numpy().tolist())
    mean_loss = running / len(loader)
    mean_pearson = float(np.mean(correlations)) if correlations else float("nan")
    l1_r, _ = pearsonr(l1_true, l1_pred)
    return mean_loss, mean_pearson, l1_r