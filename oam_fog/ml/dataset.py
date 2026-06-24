import os
import numpy as np
import torch
from torch.utils.data import Dataset

class FogDataset(Dataset):
    def __init__(self, save_dir: str):
        self.intensities = np.load(os.path.join(save_dir, "intensities.npy"))
        self.spectra = np.load(os.path.join(save_dir, "spectra.npy"))
        self.visibilities = np.load(os.path.join(save_dir, "visibilities.npy")).astype(np.float32)

    def __len__(self):
        return len(self.intensities)

    def __getitem__(self, idx):
        intensity = self.intensities[idx].copy()
        mn, mx = intensity.min(), intensity.max()
        if mx > mn:
            intensity = (intensity - mn) / (mx - mn)
        intensity = torch.from_numpy(intensity).unsqueeze(0).float()
        spectrum = torch.from_numpy(self.spectra[idx]).float()
        visibility = torch.tensor([self.visibilities[idx]], dtype=torch.float32)
        return intensity, spectrum, visibility