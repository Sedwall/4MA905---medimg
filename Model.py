import torch.nn as nn
import torch as pt

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden_layers = 256
        encoder_layer = nn.TransformerEncoderLayer(
                    d_model=1024,
                    nhead=2,
                    dim_feedforward=2048,
                    activation='relu',
                    batch_first=True
                    )
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=nn.LayerNorm(1024))
        self.unfold = nn.Unfold(kernel_size=(16, 16), stride=16)
        self.layerNorm = nn.LayerNorm(normalized_shape=(36, 768))
        self.linear = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
        )
        self.lin = nn.Sequential(
            nn.Linear(1024, self.hidden_layers),
            nn.ReLU(),
            nn.Linear(self.hidden_layers, 2),
        )


    def forward(self, x):
        # x shape: (batch, 3, 96, 96)
        assert x.shape[1:] == (3, 96, 96)
        patches = self.unfold(x)
        patches = patches.permute(0, 2, 1)
        patches = self.layerNorm(patches)
        patches = self.linear(patches)
        x = self.trans(patches)
        x = self.lin(x[:, -1, :])
        return x
    
    def save(self, filepath):
        pt.save(self.state_dict(), filepath)
    
    def load(self, filepath, device):
        self.load_state_dict(pt.load(filepath, map_location=device))
        self.to(device)
        self.eval()