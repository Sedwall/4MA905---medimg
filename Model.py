import torch.nn as nn
import torch as pt

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         dropout = 0.2
#         self.layers = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(dropout),
#             nn.Conv2d(chanels, chanels, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(dropout),
#             nn.Conv2d(chanels, chanels, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(dropout),
#             nn.Flatten(),
#             nn.Linear(chanels*11*11, 256),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, 2),
#         )

#     def forward(self, x):
#         # x shape: (batch, 96, 96, 3)
#         assert x.shape[1:] == (3, 96, 96)
#         x = self.layers(x)
#         return x
class PCamCNN(nn.Module):
    def __init__(self):
        super().__init__()
        act = nn.SiLU()  # funkar fint på små patcher

        self.features = nn.Sequential(
            # 96 -> 96
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), act,
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), act,
            nn.MaxPool2d(2),  # 96 -> 48

            # 48 -> 48
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), act,
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), act,
            nn.MaxPool2d(2),  # 48 -> 24

            # 24 -> 24
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), act,
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), act,
            nn.Dropout2d(0.2)
            #nn.MaxPool2d(2),  # 24 -> 12

            # 12 -> 12
            # nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(256), act,
            # nn.Dropout2d(0.2),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)  # global average pooling -> (B, C, 1, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),               # (B, 256)
            nn.Dropout(0.3),
            nn.Linear(128, 1)           # logit
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x.squeeze(1)  # (B,)

    def save(self, path):
        pt.save(self.state_dict(), path)

    def load(self, path, device):
        self.load_state_dict(pt.load(path, map_location=device))
