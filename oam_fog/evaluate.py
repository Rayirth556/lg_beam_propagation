import json, os, sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

ROOT    = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(ROOT, "outputs")
sys.path.insert(0, ROOT)

from ml.dataset import FogDataset
from ml.model   import OAMNet

_EPS = 1e-8


def weighted_std(spectrum, l_values):
    """Weighted standard deviation of OAM spectrum over l-axis."""
    mean_l = np.sum(spectrum * l_values)
    return np.sqrt(np.sum(spectrum * (l_values - mean_l)**2))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load basis config and beam config from metadata
    with open(os.path.join(ROOT, "data", "metadata.json")) as f:
        meta = json.load(f)
    n_modes    = meta["basis"]["n_modes"]
    modes      = meta["basis"]["modes"]           # [[p, l], ...]
    l_values   = np.array([l for p, l in modes])  # l for each mode index
    input_lmodes = meta["beam"].get("l_modes", [meta["beam"]["l_in"]])  # multiplexed modes

    # Index of each input mode (p=0) in the basis
    input_indices = [i for i, (p, l) in enumerate(modes) if p == 0 and l in input_lmodes]
    l1_idx = next(i for i, (p, l) in enumerate(modes) if p == 0 and l == input_lmodes[0])

    print(f"Basis       : {n_modes} modes")
    print(f"Input modes : l = {input_lmodes}  (indices {input_indices})")

    # Dataset & test split (same seed as train.py)
    dataset = FogDataset(save_dir=os.path.join(ROOT, "data"))
    n       = len(dataset)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val
    gen     = torch.Generator().manual_seed(42)
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

    # Weighted std per sample (true and predicted)
    wstd_true = np.array([weighted_std(s, l_values) for s in all_targets])
    wstd_pred = np.array([weighted_std(s, l_values) for s in all_preds])

    x_ticks = np.arange(n_modes)

    # Colour map: highlight input modes in red, rest in steelblue/darkorange
    def bar_colors(base_color, highlight_indices, highlight_color="crimson"):
        colors = [base_color] * n_modes
        for idx in highlight_indices:
            colors[idx] = highlight_color
        return colors

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
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.7,
               label=f"Best val (epoch {best_epoch})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Weighted MSE Loss")
    ax.set_title("Training curve")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS, "fig1_training_curve.png"), dpi=150)
    plt.close()

    # ── Fig 2: 4 example predictions (input modes highlighted in red) ─────────
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    legend_patches = [
        mpatches.Patch(color="steelblue",  label="Other modes"),
        mpatches.Patch(color="darkorange", label="Other modes (pred)"),
        mpatches.Patch(color="crimson",    label=f"Input l={input_lmodes}"),
    ]
    for i in range(4):
        ws_t = weighted_std(all_targets[i], l_values)
        ws_p = weighted_std(all_preds[i],   l_values)
        axes[i, 0].imshow(all_intensities[i, 0], cmap="inferno")
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"Sample {i+1}: Intensity\n(σ_l true={ws_t:.3f}, pred={ws_p:.3f})")

        axes[i, 1].bar(x_ticks, all_targets[i],
                       color=bar_colors("steelblue", input_indices))
        axes[i, 1].set_title(f"Sample {i+1}: True spectrum  σ_l={ws_t:.3f}")
        axes[i, 1].set_ylim(0, 1); axes[i, 1].set_xlabel("Mode index")

        axes[i, 2].bar(x_ticks, all_preds[i],
                       color=bar_colors("darkorange", input_indices))
        axes[i, 2].set_title(f"Sample {i+1}: Predicted spectrum  σ_l={ws_p:.3f}")
        axes[i, 2].set_ylim(0, 1); axes[i, 2].set_xlabel("Mode index")

    fig.legend(handles=legend_patches, loc="upper center", ncol=3, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
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
    ax.set_xlabel(f"True p(l={input_lmodes[0]})")
    ax.set_ylabel(f"Predicted p(l={input_lmodes[0]})")
    ax.set_title(f"Primary mode (l={input_lmodes[0]}) power prediction")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS, "fig3_l1_scatter.png"), dpi=150)
    plt.close()

    # ── Fig 4: per-mode MAE (input modes highlighted) ─────────────────────────
    mae_per_mode = np.abs(all_preds - all_targets).mean(axis=0)
    colors_mae   = bar_colors("mediumpurple", input_indices, "crimson")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(x_ticks, mae_per_mode, color=colors_mae)
    ax.set_xlabel("Mode index"); ax.set_ylabel("Mean |error|")
    ax.set_title("Per-mode prediction error  (red = input modes)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS, "fig4_per_mode_error.png"), dpi=150)
    plt.close()

    # ── Fig 5: weighted std distribution ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(wstd_true, bins=40, alpha=0.6, label="True",      color="steelblue")
    ax.hist(wstd_pred, bins=40, alpha=0.6, label="Predicted", color="darkorange")
    ax.set_xlabel("Weighted OAM std  σ_l  [rad]")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of weighted OAM spread across test set")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS, "fig5_weighted_std.png"), dpi=150)
    plt.close()

    # ── Fig 6: weighted std scatter (true vs predicted) ───────────────────────
    a2, b2 = wstd_true - wstd_true.mean(), wstd_pred - wstd_pred.mean()
    d2     = np.sqrt((a2**2).sum() * (b2**2).sum())
    r_ws   = float(np.dot(a2, b2) / d2) if d2 > 0 else 0.0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(wstd_true, wstd_pred, s=18, alpha=0.5, label=f"r = {r_ws:.3f}")
    lo2 = min(wstd_true.min(), wstd_pred.min()) - 0.1
    hi2 = max(wstd_true.max(), wstd_pred.max()) + 0.1
    ax.plot([lo2, hi2], [lo2, hi2], "r--")
    ax.set_xlabel("True σ_l"); ax.set_ylabel("Predicted σ_l")
    ax.set_title("Weighted OAM std prediction")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS, "fig6_wstd_scatter.png"), dpi=150)
    plt.close()

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\nFigures saved to outputs/")
    print(f"\nPrimary mode (l={input_lmodes[0]}) Pearson r : {r:.4f}")
    print(f"Weighted std  Pearson r               : {r_ws:.4f}")
    print(f"Mean MAE (all modes)                  : {mae_per_mode.mean():.4f}")
    print(f"\nWeighted OAM std  σ_l:")
    print(f"  True  — mean={wstd_true.mean():.4f}  std={wstd_true.std():.4f}")
    print(f"  Pred  — mean={wstd_pred.mean():.4f}  std={wstd_pred.std():.4f}")


if __name__ == "__main__":
    main()