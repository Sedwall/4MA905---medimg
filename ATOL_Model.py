import torch.nn as nn
import torch as pt

class Model(nn.Module):
    def __init__(self, chanels=16, dropout=.5):
        super(Model, self).__init__()
        self.CNNlayers = nn.Sequential(
            nn.Conv2d(3, chanels, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Conv2d(chanels, chanels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Conv2d(chanels, chanels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Flatten(),
        )
        self.dis = nn.Sequential(
            nn.Linear(chanels*11*11, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x, _):
        # x shape: (batch, 3, 96, 96)
        assert x.shape[1:] == (3, 96, 96)
        x = self.CNNlayers(x)
        x = self.dis(x)
        return x