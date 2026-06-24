import torch
import torch.nn as nn


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class OAMNet(nn.Module):
    """
    CNN that maps a fog-distorted intensity image → OAM power spectrum.

    Architecture:
        4× conv blocks (1→32→64→128→256) with MaxPool → Global Average Pool
        Concatenate visibility scalar → 257-dim feature
        Two heads:
          l1_head  : sigmoid  → scalar power in the l=+1 input mode
          shape_head: softmax → distribution over the remaining (n_modes-1) modes
        Reconstruct full n_modes spectrum from the two heads.

    Parameters
    ----------
    n_modes : int   total number of OAM basis modes (default 102)
    l1_idx  : int   index of the (p=0, l=+1) mode in the basis (default 9)
    """

    def __init__(self, n_modes: int = 102, l1_idx: int = 9):
        super().__init__()
        self.n_modes = n_modes
        self.l1_idx  = l1_idx
        self.other_idx = [i for i in range(n_modes) if i != l1_idx]

        self.block1 = _conv_block(1,   32)
        self.block2 = _conv_block(32,  64)
        self.block3 = _conv_block(64,  128)
        self.block4 = _conv_block(128, 256)
        self.gap    = nn.AdaptiveAvgPool2d(1)

        feat_dim = 256 + 1   # conv features + visibility scalar

        self.l1_head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 64),       nn.ReLU(inplace=True),
            nn.Linear(64, 1),         nn.Sigmoid(),
        )
        self.shape_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 128),      nn.ReLU(inplace=True),
            nn.Linear(128, n_modes - 1),
        )

    def forward(self, x, visibility):
        x    = self.block1(x)
        x    = self.block2(x)
        x    = self.block3(x)
        x    = self.block4(x)
        feat = self.gap(x).flatten(1)                          # (B, 256)
        feat = torch.cat([feat, visibility], dim=1)            # (B, 257)

        p_l1  = self.l1_head(feat).squeeze(1)                 # (B,)
        shape = torch.softmax(self.shape_head(feat), dim=1)   # (B, n_modes-1)

        # Reconstruct full spectrum
        full = torch.zeros(x.shape[0], self.n_modes, device=x.device)
        full[:, self.l1_idx]    = p_l1
        full[:, self.other_idx] = (1 - p_l1).unsqueeze(1) * shape
        return full, p_l1, shape