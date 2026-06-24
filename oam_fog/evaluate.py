import json, os, sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

ROOT    = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(ROOT, "outputs")
sys.path.insert(0, ROOT)

from ml.dataset import FogDataset
from ml.model   import OAMNet

_EPS = 1e-8


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load basis config from metadata so nothing is hardcoded
    with open(os.path.join(ROOT, "data", "metadata.json")) as f:
        meta = json.load(f)
    n_modes = meta["basis"]["n_modes"]
    modes   = meta["basis"]["modes"]          # [[p, l], ...]
    l1_idx  = next(i for i, (p, l) in enumerate(modes) if p == 0 and l == 1)
    mode_labels = [f"l={l}" for p, l in modes]
    print(f"Basis: {n_modes} modes  |  l=+1 index: {l1_idx}")

    # Dataset & test split (same seed as train.py)
    dataset  = FogDataset(save_dir=os.path.join(ROOT, "data"))
    n        = len(dataset)
    n_train  = int(0.8 * n)
    n_val    = int(0.1 * n)
    n_test   = n - n_train - n_val
    gen      = torch.Generator().manual_seed(42)
    _, _, test_set = random_split(dataset, [n_train, n_val, n_test], generator=gen)
    test_loader    = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

    # Load model
    model = OAMNet(n_modes=n_modes, l1_idx=l1_idx).to(device)
    model.load_state_dict(torch.load(os.path.join(OUTPUTS, "oamnet_best.pt"),
                                     map_location=device))
    model.eval()

    os.makedirs(OUTPUTS, exist_ok=True)
    all_preds, all_targets, all_intensities, all_l1_pred = [], [], [], []

    with torch.no_grad():
        # Dataset yields (intensity, spectrum, visibility) — keep order correct
        for intensities, spectra, visibility in test_loader:
            full, p_l1, shape = model(intensities.to(device), visibility.to(device))
            all_preds.append(full.cpu().numpy())
            all_l1_pred.append(p_l1.cpu().numpy())
            all_targets.append(spectra.numpy())
            all_intensities.append(intensities.numpy())

    all_preds       = np.concatenate(all_preds)
    all_targets     = np.concatenate(all_targets)
    all_intensities = np.concatenate(all_intensities)
    all_l1_pred     = np.concatenate(all_l1_pred)

    # ── Fig 1: Training curve ─────────────────────────────────────────────────
    with open(os.path.join(OUTPUTS, "losses.json")) as f:
        losses = json.load(f)
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs     = [d["epoch"]      for d in losses]
    train_loss = [d["train_loss"] for d in losses]
    val_loss   = [d["val_loss"]   for d in losses]
    ax.plot(epochs, train_loss, label="Train")
    ax.plot(epochs, val_loss,   label="Val")
    best_epoch = epochs[int(np.argmin(val_loss))]
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.6,
               label=f"Best val (epoch {best_epoch})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Weighted MSE Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS, "fig1_training_curve.png"), dpi=150)
    plt.close()

    # ── Fig 2: 4 example predictions ─────────────────────────────────────────
    x_ticks = range(n_modes)
    fig, axes = plt.subplots(4, 3, figsize=(14, 16))
    for i in range(4):
        axes[i, 0].imshow(all_intensities[i, 0], cmap="inferno")
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"Sample {i+1}: Intensity")
        axes[i, 1].bar(x_ticks, all_targets[i], color="steelblue")
        axes[i, 1].set_title(f"Sample {i+1}: True spectrum")
        axes[i, 1].set_ylim(0, 1)
        axes[i, 2].bar(x_ticks, all_preds[i], color="darkorange")
        axes[i, 2].set_title(f"Sample {i+1}: Predicted spectrum")
        axes[i, 2].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS, "fig2_example_predictions.png"), dpi=150)
    plt.close()

    # ── Fig 3: l=+1 scatter ───────────────────────────────────────────────────
    true_l1 = all_targets[:, l1_idx]
    pred_l1 = all_l1_pred
    a, b    = true_l1 - true_l1.mean(), pred_l1 - pred_l1.mean()
    denom   = np.sqrt((a**2).sum() * (b**2).sum())
    r       = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_l1, pred_l1, s=18, alpha=0.5, label=f"r = {r:.3f}")
    lo = min(true_l1.min(), pred_l1.min()) - 0.02
    hi = max(true_l1.max(), pred_l1.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel("True p(l=+1)"); ax.set_ylabel("Predicted p(l=+1)")
    ax.set_title("l=+1 mode power prediction")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS, "fig3_l1_scatter.png"), dpi=150)
    plt.close()

    # ── Fig 4: mean absolute error per mode ───────────────────────────────────
    mae_per_mode = np.abs(all_preds - all_targets).mean(axis=0)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(x_ticks, mae_per_mode, color="mediumpurple")
    ax.axvline(l1_idx, color="red", linestyle="--", label=f"l=+1 (idx {l1_idx})")
    ax.set_xlabel("Mode index"); ax.set_ylabel("Mean |error|")
    ax.set_title("Per-mode prediction error")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS, "fig4_per_mode_error.png"), dpi=150)
    plt.close()

    print(f"\nFigures saved to outputs/")
    print(f"l=+1 Pearson r : {r:.4f}")
    print(f"Mean MAE       : {mae_per_mode.mean():.4f}")
    print(f"l=+1 MAE       : {mae_per_mode[l1_idx]:.4f}")


if __name__ == "__main__":
    main()