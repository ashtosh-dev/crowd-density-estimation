"""
CSRNet Model Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CSRNet(nn.Module):
    """CSRNet for crowd density estimation."""
    
    def __init__(self, load_vgg_weights=True):
        super(CSRNet, self).__init__()
        self.frontend = self._make_layers(
            [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],
            in_ch=3
        )
        self.backend = self._make_layers(
            [512, 512, 512, 256, 128, 64],
            in_ch=512,
            dilation=True
        )
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if load_vgg_weights:
            self._init_vgg()
    
    def _make_layers(self, cfg, in_ch=3, dilation=False):
        layers = []
        d = 2 if dilation else 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_ch, v, kernel_size=3, padding=d, dilation=d),
                    nn.ReLU(inplace=True)
                ]
                in_ch = v
        return nn.Sequential(*layers)
    
    def _init_vgg(self):
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            vgg_features = list(vgg.features.children())[:23]
            vgg_seq = nn.Sequential(*vgg_features)
            vgg_state = vgg_seq.state_dict()
            frontend_state = self.frontend.state_dict()
            matched = {
                k: v for k, v in vgg_state.items()
                if k in frontend_state and frontend_state[k].shape == v.shape
            }
            frontend_state.update(matched)
            self.frontend.load_state_dict(frontend_state)
            print("[INFO] Loaded VGG16 frontend weights.")
        except Exception as e:
            print(f"[WARN] Could not load VGG weights: {e}")
    
    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        return x
