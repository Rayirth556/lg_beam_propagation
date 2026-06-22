import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless plotting support
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Add current directory to path to ensure robust imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import basis
from ml.dataset import FogDataset
from ml.model import OAMNet


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # 1. Plot Fig 1: Training Curve from losses.json
    losses_json_path = os.path.join(outputs_dir, "losses.json")
    if os.path.exists(losses_json_path):
        print(f"Loading training history from {losses_json_path}...")
        with open(losses_json_path, "r") as f:
            history = json.load(f)
        
        epochs = [x["epoch"] for x in history]
        train_losses = [x["train_loss"] for x in history]
        val_losses = [x["val_loss"] for x in history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=2)
        plt.plot(epochs, val_losses, label="Val Loss", color="#ff7f0e", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("KL Divergence Loss")
        plt.title("Training and Validation Loss Curve")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        fig1_path = os.path.join(outputs_dir, "fig1.png")
        plt.savefig(fig1_path, dpi=300)
        plt.savefig(os.path.join(outputs_dir, "fig1_training_curve.png"), dpi=300)
        plt.close()
        print(f"Saved Fig 1 to {fig1_path}")
    else:
        print(f"Warning: {losses_json_path} not found. Skipping Fig 1 generation.")

    # 2. Re-create the exact test dataset split
    print("Loading dataset for evaluation...")
    dataset = FogDataset(data_dir)
    total_samples = len(dataset)

    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    _, _, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    test_indices = test_set.indices
    
    # Load visibilities for coloring Fig 3
    visibilities_path = os.path.join(data_dir, "visibilities.npy")
    if os.path.exists(visibilities_path):
        visibilities = np.load(visibilities_path)[test_indices]
    else:
        visibilities = None

    # DataLoader for test set
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

    # Load Best Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OAMNet().to(device)
    best_model_path = os.path.join(outputs_dir, "oamnet_best.pt")
    
    if not os.path.exists(best_model_path):
        print(f"Error: Best model checkpoint not found at {best_model_path}")
        sys.exit(1)
        
    print(f"Loading best model from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Collect predictions and ground truths
    all_intensities = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch_intensities, batch_spectra in test_loader:
            batch_intensities_dev = batch_intensities.to(device)
            preds = model(batch_intensities_dev)
            
            all_intensities.append(batch_intensities.numpy())
            all_targets.append(batch_spectra.numpy())
            all_preds.append(preds.cpu().numpy())

    all_intensities = np.concatenate(all_intensities, axis=0) # shape (N_test, 1, 128, 128)
    all_targets = np.concatenate(all_targets, axis=0)         # shape (N_test, 11)
    all_preds = np.concatenate(all_preds, axis=0)             # shape (N_test, 11)

    # 3. Plot Fig 2: 4 Example Predictions (Intensity, True Spectrum, Predicted Spectrum side by side)
    print("Generating Fig 2 (4 example predictions)...")
    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    
    # Define labels for the 11 OAM modes (e.g. -5 to 5)
    mode_ticks = np.arange(len(basis.modes))
    mode_labels = [f"{l:+d}" for _, l in basis.modes]

    for row in range(4):
        # Sample index
        idx = row
        img = all_intensities[idx, 0]  # shape (128, 128)
        true_spec = all_targets[idx]
        pred_spec = all_preds[idx]

        # Column 0: Intensity Image
        im = axes[row, 0].imshow(img, cmap="inferno", extent=[-grid_extent_mm(base_dir)/2, grid_extent_mm(base_dir)/2, -grid_extent_mm(base_dir)/2, grid_extent_mm(base_dir)/2])
        axes[row, 0].set_title(f"Sample {row+1} Output Intensity")
        axes[row, 0].set_xlabel("x (mm)")
        axes[row, 0].set_ylabel("y (mm)")
        fig.colorbar(im, ax=axes[row, 0], fraction=0.046, pad=0.04)

        # Column 1: True Spectrum Bar Plot
        axes[row, 1].bar(mode_ticks, true_spec, color="#2ca02c", alpha=0.8, edgecolor="black")
        axes[row, 1].set_xticks(mode_ticks)
        axes[row, 1].set_xticklabels(mode_labels)
        axes[row, 1].set_ylim(0, 1.05)
        axes[row, 1].set_title(f"Sample {row+1} True OAM Spectrum")
        axes[row, 1].set_xlabel("OAM Mode (l)")
        axes[row, 1].set_ylabel("Normalized Power")
        axes[row, 1].grid(axis="y", linestyle="--", alpha=0.5)

        # Column 2: Predicted Spectrum Bar Plot
        axes[row, 2].bar(mode_ticks, pred_spec, color="#d62728", alpha=0.8, edgecolor="black")
        axes[row, 2].set_xticks(mode_ticks)
        axes[row, 2].set_xticklabels(mode_labels)
        axes[row, 2].set_ylim(0, 1.05)
        axes[row, 2].set_title(f"Sample {row+1} Predicted OAM Spectrum")
        axes[row, 2].set_xlabel("OAM Mode (l)")
        axes[row, 2].set_ylabel("Normalized Power")
        axes[row, 2].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig2_path = os.path.join(outputs_dir, "fig2.png")
    plt.savefig(fig2_path, dpi=300)
    plt.savefig(os.path.join(outputs_dir, "fig2_examples.png"), dpi=300)
    plt.close()
    print(f"Saved Fig 2 to {fig2_path}")

    # 4. Plot Fig 3: Scatter plot of true vs predicted l=1 power
    # Dynamically find the index corresponding to OAM mode l=1
    l1_idx = None
    for i, (p, l) in enumerate(basis.modes):
        if l == 1:
            l1_idx = i
            break
            
    if l1_idx is None:
        print("Warning: OAM mode l=1 not found in basis modes config! Defaulting to index 6.")
        l1_idx = 6

    true_l1 = all_targets[:, l1_idx]
    pred_l1 = all_preds[:, l1_idx]

    plt.figure(figsize=(7, 6))
    if visibilities is not None:
        # Color by visibility to show performance variation across fog densities
        sc = plt.scatter(true_l1, pred_l1, c=visibilities, cmap="viridis_r", alpha=0.7, edgecolors="none")
        cbar = plt.colorbar(sc)
        cbar.set_label("Visibility (m)", rotation=270, labelpad=15)
    else:
        plt.scatter(true_l1, pred_l1, alpha=0.7, color="#1f77b4", edgecolors="none")

    # Add 1:1 diagonal line
    plt.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Perfect Prediction")
    plt.xlabel("True Power (l = +1)")
    plt.ylabel("Predicted Power (l = +1)")
    plt.title("OAM mode l = +1 Power: True vs Predicted")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    
    fig3_path = os.path.join(outputs_dir, "fig3.png")
    plt.savefig(fig3_path, dpi=300)
    plt.savefig(os.path.join(outputs_dir, "fig3_scatter.png"), dpi=300)
    plt.close()
    print(f"Saved Fig 3 to {fig3_path}")


def grid_extent_mm(base_dir) -> float:
    """Helper to retrieve physical grid extent in mm from metadata.json if available."""
    metadata_path = os.path.join(base_dir, "data", "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                meta = json.load(f)
            return meta["grid"]["L_m"] * 1e3  # convert to mm
        except Exception:
            pass
    return 32.0  # default fallback value


if __name__ == "__main__":
    main()
