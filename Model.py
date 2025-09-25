import torch.nn as nn
import torch as pt


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden_layers = 256
        self.transInput = 192
        encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.transInput,
                    nhead=2,
                    dim_feedforward=2048,
                    activation='relu',
                    batch_first=True
                    )
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=nn.LayerNorm(self.transInput))
        self.layerNorm = nn.LayerNorm(normalized_shape=(9, self.transInput))
    
        self.discriminator = nn.Sequential(
            nn.Linear(192, self.hidden_layers),
            nn.ReLU(),
            nn.Linear(self.hidden_layers, 2),
        )
        self.outerCNN = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.midCNN = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        


    def forward(self, x):
        # x shape: (batch, 3, 96, 96)
        assert x.shape[1:] == (3, 96, 96)
        
        # Split each image into 3x3 grid of 32x32 patches
        # x shape: (batch, 3, 96, 96)
        patches = []
        for i in range(3):
            for j in range(3):
                patch = x[:, :, i*32:(i+1)*32, j*32:(j+1)*32]  # (batch, 3, 32, 32)
                if i == 1 and j == 1:
                    patch = self.midCNN(patch)
                else:
                    patch = self.outerCNN(patch)
                patches.append(patch)
        patches = pt.stack(patches, dim=1)  # (batch, 9, 3, 32, 32)
        patches = patches.reshape(x.shape[0], 9, 3*patches.shape[-1]**2)
        patches = self.layerNorm(patches)
        x = self.trans(patches)
        x = self.discriminator(x[:, -1, :])
        return x
    
    def save(self, filepath):
        pt.save(self.state_dict(), filepath)
    
    def load(self, filepath, device):
        self.load_state_dict(pt.load(filepath, map_location=device))
        self.to(device)
        self.eval()