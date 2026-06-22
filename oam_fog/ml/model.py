import torch
import torch.nn as nn

class OAMNet(nn.Module):
    """
    CNN architecture for predicting OAM power spectra from intensity profiles.
    
    Input shape: (B, 1, 128, 128)
    Output shape: (B, 11) - Probability distribution over OAM topological charges.
    """
    def __init__(self):
        super(OAMNet, self).__init__()
        
        # Block 1: 1 -> 32 -> 32 channels. MaxPool reduces spatial dims 128x128 -> 64x64.
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: 32 -> 64 -> 64 channels. MaxPool reduces spatial dims 64x64 -> 32x32.
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: 64 -> 128 -> 128 channels. MaxPool reduces spatial dims 32x32 -> 16x16.
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global Average Pooling: reduces (B, 128, 16, 16) -> (B, 128, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # MLP Classifier head
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 11)
        )
        
        # Softmax to produce probabilities summing to 1
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)  # Flatten spatial dimensions
        
        x = self.mlp(x)
        x = self.softmax(x)
        
        return x
