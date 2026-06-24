"""Train OAMNet on the fog-distorted OAM dataset."""

import json, os, sys
import torch
from torch.utils.data import DataLoader, random_split

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from ml.dataset import FogDataset
from ml.model   import OAMNet
from ml.train   import evaluate, train_epoch


def get_l1_idx(modes):
    """Return index of (p=0, l=+1) in the modes list."""
    for i, (p, l) in enumerate(modes):
        if p == 0 and l == 1:
            return i
    raise ValueError("(p=0, l=+1) not found in basis modes — check config.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir  = os.path.join(ROOT, "data")
    meta_path = os.path.join(data_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    n_modes = meta["basis"]["n_modes"]
    modes   = meta["basis"]["modes"]          # list of [p, l] pairs
    l1_idx  = get_l1_idx(modes)
    print(f"Basis: {n_modes} modes  |  l=+1 index: {l1_idx}")

    dataset = FogDataset(save_dir=data_dir)
    n       = len(dataset)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val
    gen     = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    model     = OAMNet(n_modes=n_modes, l1_idx=l1_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    outputs_dir   = os.path.join(ROOT, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    best_val_loss = float("inf")
    losses        = []

    print(f"\nTraining for 100 epochs  ({n_train} train / {n_val} val / {n_test} test)\n")

    for epoch in range(1, 101):
        train_loss           = train_epoch(model, train_loader, optimizer, device)
        val_loss, _, val_l1r = evaluate(model, val_loader, device)
        scheduler.step()
        losses.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_l1_r": val_l1r})
        print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f} | val_l1_r={val_l1r:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(outputs_dir, "oamnet_best.pt"))
            print("           -> best saved")

    with open(os.path.join(outputs_dir, "losses.json"), "w") as f:
        json.dump(losses, f, indent=2)

    model.load_state_dict(torch.load(os.path.join(outputs_dir, "oamnet_best.pt"),
                                     map_location=device))
    test_loss, test_pearson, test_l1r = evaluate(model, test_loader, device)
    print(f"\n{'='*60}")
    print(f"Test loss:     {test_loss:.6f}")
    print(f"Mean Pearson r: {test_pearson:.4f}")
    print(f"l=+1 Pearson r: {test_l1r:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()