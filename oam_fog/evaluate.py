import json, os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from ml.dataset import FogDataset
from ml.model import OAMNet

L1_IDX = 6  # ℓ=1 is index 6 in modes [-5..+5]
_EPS = 1e-8

# Resolve paths relative to this script, regardless of CWD
ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(ROOT, "outputs")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FogDataset(save_dir=os.path.join(ROOT, "data"))
    n = len(dataset)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    n_test = n - n_train - n_val
    gen = torch.Generator().manual_seed(42)
    _, _, test_set = random_split(dataset, [n_train, n_val, n_test], generator=gen)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

    model = OAMNet().to(device)
    model.load_state_dict(torch.load(os.path.join(OUTPUTS, "oamnet_best.pt"), map_location=device))
    model.eval()

    os.makedirs(OUTPUTS, exist_ok=True)
    all_preds, all_targets, all_intensities, all_l1_pred = [], [], [], []
    with torch.no_grad():
        for intensities, auxs, spectra in test_loader:
            full, p_l1, shape = model(intensities.to(device), auxs.to(device))
            all_preds.append(full.cpu().numpy())
            all_l1_pred.append(p_l1.cpu().numpy())
            all_targets.append(spectra.numpy())
            all_intensities.append(intensities.numpy())

    all_preds       = np.concatenate(all_preds)
    all_targets     = np.concatenate(all_targets)
    all_intensities = np.concatenate(all_intensities)
    all_l1_pred     = np.concatenate(all_l1_pred)

    # Fig 1: training curve
    with open(os.path.join(OUTPUTS, "losses.json")) as f:
        losses = json.load(f)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([d["epoch"] for d in losses], [d["train_loss"] for d in losses], label="Train")
    ax.plot([d["epoch"] for d in losses], [d["val_loss"]   for d in losses], label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Weighted MSE Loss"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUTS, "fig1_training_curve.png"), dpi=150); plt.close()

    # Fig 2: 4 example predictions
    fig, axes = plt.subplots(4, 3, figsize=(13, 16))
    for i in range(4):
        axes[i,0].imshow(all_intensities[i,0], cmap="inferno"); axes[i,0].axis("off")
        axes[i,0].set_title(f"Sample {i+1}: Intensity")
        axes[i,1].bar(range(11), all_targets[i], color="steelblue")
        axes[i,1].set_title(f"Sample {i+1}: True"); axes[i,1].set_ylim(0,1)
        axes[i,2].bar(range(11), all_preds[i],   color="darkorange")
        axes[i,2].set_title(f"Sample {i+1}: Predicted"); axes[i,2].set_ylim(0,1)
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUTS, "fig2_example_predictions.png"), dpi=150); plt.close()

    # Fig 3: l=1 scatter
    true_l1, pred_l1 = all_targets[:, L1_IDX], all_l1_pred
    a, b = true_l1 - true_l1.mean(), pred_l1 - pred_l1.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    r = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_l1, pred_l1, s=18, alpha=0.5, label=f"r={r:.3f}")
    lo, hi = min(true_l1.min(), pred_l1.min())-0.02, max(true_l1.max(), pred_l1.max())+0.02
    ax.plot([lo,hi],[lo,hi],"r--"); ax.set_xlabel("True l=1"); ax.set_ylabel("Predicted l=1")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUTS, "fig3_l1_scatter.png"), dpi=150); plt.close()

    print("Figures saved to outputs/")
    print(f"l=1 Pearson r: {r:.4f}")

if __name__ == "__main__":
    main()