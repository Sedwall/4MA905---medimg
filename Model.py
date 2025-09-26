import torch.nn as nn
import torch as pt


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden_layers = 256

        self.discriminator = nn.Sequential(
            nn.Linear(4*8*8, self.hidden_layers),
            nn.ReLU(),
            nn.Linear(self.hidden_layers, 2),
        )
        self.midCNN = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
    
    def forward(self, x):
        patch = x[:, :, 1*32:(1+1)*32, 1*32:(1+1)*32]

        x = self.midCNN(patch)

        return self.discriminator(x)

    
    def save(self, filepath):
        pt.save(self.state_dict(), filepath)
    
    def load(self, filepath, device):
        self.load_state_dict(pt.load(filepath, map_location=device))
        self.to(device)
        self.eval()