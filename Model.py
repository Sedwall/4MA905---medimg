import torch.nn as nn
import torch as pt

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(96*96*1, 2)
        )    
    
    def forward(self, data):
        X = self.model.forward(data)
        return X
    
    def save(self, path):
        pt.save(self.state_dict(), path)

    def load(self, path, device):
        self.load_state_dict(pt.load(path, map_location=device))
