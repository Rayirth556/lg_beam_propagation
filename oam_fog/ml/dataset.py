import os
import numpy as np
import torch
from torch.utils.data import Dataset

class FogDataset(Dataset):
    """
    PyTorch Dataset for OAM crosstalk simulation.
    Loads intensity maps and corresponding OAM power spectra.
    """
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        
        intensities_path = os.path.join(save_dir, "intensities.npy")
        spectra_path = os.path.join(save_dir, "spectra.npy")
        
        if not os.path.exists(intensities_path):
            raise FileNotFoundError(f"Intensities file not found at: {intensities_path}")
        if not os.path.exists(spectra_path):
            raise FileNotFoundError(f"Spectra file not found at: {spectra_path}")
            
        # Load dataset arrays
        self.intensities = np.load(intensities_path).astype(np.float32)
        self.spectra = np.load(spectra_path).astype(np.float32)
        
    def __len__(self) -> int:
        return len(self.intensities)
        
    def __getitem__(self, idx: int):
        # Retrieve the intensity profile and OAM spectrum
        intensity = self.intensities[idx]  # shape: (128, 128)
        spectrum = self.spectra[idx]      # shape: (11,)
        
        # Add channel dimension: (1, 128, 128)
        intensity = np.expand_dims(intensity, axis=0)
        
        # Convert to float32 PyTorch tensors
        intensity_tensor = torch.from_numpy(intensity)
        spectrum_tensor = torch.from_numpy(spectrum)
        
        return intensity_tensor, spectrum_tensor
