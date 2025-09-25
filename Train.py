import numpy as np
import torch
from Model import Model
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt
from Evaluate import Evaluate
from time import time

path = Path(__file__).parent.parent.parent / "dataset" / "pcam_pt"
print(f"Using data from {path}")
assert path.exists(), f"Path {path} does not exist"

# Training data
X = torch.load(path/"train_x.pt").permute(0, 3, 1, 2) # (N, channels, height, width)
y = torch.load(path/"train_y.pt").reshape(-1).to(torch.long)


# Validation data
X_val = torch.load(path/"valid_x.pt").permute(0, 3, 1, 2)[:2_000]
y_val = torch.load(path/"valid_y.pt").reshape(-1).to(torch.long)[:2_000]
print(f"Training data shape: {tuple(X.shape)}, Training labels shape: {tuple(y.shape)}")
print(f"Validation data shape: {tuple(X_val.shape)}, Validation labels shape: {tuple(y_val.shape)}")

n_epochs = 50
batch_size = 512 * 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print()
print(f"Using {DEVICE} device")

X = X
y = y
X_val = X_val.to(DEVICE)
y_val = y_val.to(DEVICE)

trainLoader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
model = Model().to(DEVICE)
loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_history = []
TOTAL_ITEMS = len(trainLoader.dataset) * n_epochs
ITEMS_IN_EPOCH = len(trainLoader.dataset)
start = time()

# Training loop
for epoch in range(n_epochs):
    for i, (Xbatch, ybatch) in enumerate(trainLoader):
        Xbatch = Xbatch.to(DEVICE)
        ybatch = ybatch.to(DEVICE)
        y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            progress = epoch * ITEMS_IN_EPOCH + i * len(Xbatch)
            stars = '=' * int(20 * (progress + 1) / TOTAL_ITEMS + 1)
            print(f"Train Epoch: {epoch} [{i * len(Xbatch):>5d}/{TOTAL_ITEMS} [{stars:-<19}] ({100. * (progress+1) / TOTAL_ITEMS:2.0f}%)]\tLoss: {loss.item():.6f}", end='\r')
    loss_history.append(loss.item())
print("\n")  # For newline after the last epoch

run_time = time() - start
hours, rem = divmod(run_time, 3600)
minutes, seconds = divmod(rem, 60)

del X, y  # Free memory
del Xbatch, ybatch  # Free memory

# Save the model
model.save(Path(__file__).parent / "model.pt")

# Evaluate the model
evaluator = Evaluate(model, DEVICE)
metrics = evaluator.evaluate(X_val, y_val)
metrics["training_time"] = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
evaluator.print_metrics(metrics)
evaluator.save_metrics(metrics, Path(__file__).parent / "metrics.txt")

# Plot training loss over epochs
# Note: We only have the last loss value from each epoch.
plt.plot(np.arange(n_epochs), loss_history, marker='o' )
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.tight_layout()
plt.savefig(Path(__file__).parent / "plot.png")