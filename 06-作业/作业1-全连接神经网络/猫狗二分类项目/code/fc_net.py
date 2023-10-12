import torch
import torch.nn as nn

class FcNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.Linear(3 * 64 * 64, 2048), nn.ReLU()
        )
        self.hidden_layer = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.out_layer = nn.Sequential(
            nn.Linear(32, 2), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.in_layer(x)
        x = self.hidden_layer(x)
        x = self.out_layer(x)
        return x

