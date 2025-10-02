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
    """
    In:  [B, 3, 96, 96]
    Out: [B, 2]  (logits för CrossEntropyLoss)
    """
    def __init__(self, num_classes=2, drop_block=0.25, drop_fc=0.5):
        super().__init__()

        # Block 1: 96->94->92->90 -> pool -> 45
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0, bias=False),  # (96->94)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=False), # (94->92)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=False), # (92->90)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                             # (90->45)
            nn.Dropout(p=drop_block),
        )
        # Block 2: 45->43->41->39 -> pool -> 19
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=False), # (45->43)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False), # (43->41)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False), # (41->39)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                             # (39->19)
            nn.Dropout(p=drop_block),
        )
        # Block 3: 19->17->15->13 -> pool -> 6
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, bias=False), # (19->17)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, bias=False),# (17->15)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, bias=False),# (15->13)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                              # (13->6)
            nn.Dropout(p=drop_block),
        )

        # Flatten 6*6*128 = 4608 -> 256 -> 2
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # 4608
            nn.Linear(6 * 6 * 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_fc),
            nn.Linear(256, num_classes), # logits [B,2]
        )

    def forward(self, x):
        # förväntar [B,3,96,96]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x  # logits [B,2]
    
    def save(self, path):
        pt.save(self.state_dict(), path)

    def load(self, path, device):
        self.load_state_dict(pt.load(path, map_location=device))
