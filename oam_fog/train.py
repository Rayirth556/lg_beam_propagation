import os
import sys
import json
import torch
from torch.utils.data import DataLoader

# Add current directory to path to ensure robust imports of ml package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.dataset import FogDataset
from ml.model import OAMNet
from ml.train import train_epoch, evaluate


def main():
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    print("Loading dataset...")
    dataset = FogDataset(data_dir)
    total_samples = len(dataset)
    print(f"Total samples found: {total_samples}")

    # 80/10/10 split
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    # Reproducible seeded random split
    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    print(f"Split sizes: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

    # DataLoaders
    # num_workers=0 is standard and safe on Windows to avoid spawn/multiprocessing issues
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = OAMNet().to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Training loop
    epochs = 50
    losses_history = []
    best_val_loss = float("inf")
    best_model_path = os.path.join(outputs_dir, "oamnet_best.pt")

    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_pearson = evaluate(model, val_loader, device)
        scheduler.step()

        # Save history
        losses_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pearson": val_pearson
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            best_marker = " (Best Saved)"
        else:
            best_marker = ""

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Val Pearson: {val_pearson:.4f}{best_marker}")

    # Save training history to json
    losses_json_path = os.path.join(outputs_dir, "losses.json")
    with open(losses_json_path, "w") as f:
        json.dump(losses_history, f, indent=2)
    print(f"\nTraining metrics saved to {losses_json_path}")

    # Load best model for testing
    print(f"Loading best model from {best_model_path} for testing...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Evaluate on test set
    test_loss, test_pearson = evaluate(model, test_loader, device)
    print("\n" + "="*50)
    print("Test Set Performance:")
    print(f"  Test KL Loss:   {test_loss:.6f}")
    print(f"  Test Pearson R: {test_pearson:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
