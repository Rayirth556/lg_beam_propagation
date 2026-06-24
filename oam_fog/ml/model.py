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
    L1_IDX = 6

    def __init__(self):
        super().__init__()
        self.block1 = _conv_block(1, 32)
        self.block2 = _conv_block(32, 64)
        self.block3 = _conv_block(64, 128)
        self.gap = nn.AdaptiveAvgPool2d(1)

        feat_dim = 128 + 1  # + visibility scalar

        self.l1_head = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.shape_head = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, 10),
        )

    def forward(self, x, visibility):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        feat = self.gap(x).flatten(1)
        feat = torch.cat([feat, visibility], dim=1)

        p_l1 = self.l1_head(feat).squeeze(1)          # (B,)
        shape = torch.softmax(self.shape_head(feat), dim=1)  # (B, 10)

        other_idx = [i for i in range(11) if i != self.L1_IDX]
        full = torch.zeros(x.shape[0], 11, device=x.device)
        full[:, self.L1_IDX] = p_l1
        full[:, other_idx] = (1 - p_l1).unsqueeze(1) * shape
        return full, p_l1, shape