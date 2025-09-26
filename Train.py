import numpy as np
import torch
from Model import Model
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PCAMdataset import PCAMdataset
from pathlib import Path
import matplotlib.pyplot as plt
from Evaluate import Evaluate
from sklearn.metrics import accuracy_score
from time import time

dir = Path(__file__).parent.parent.parent.joinpath("dataset/pcam")

# Validation data
X_val = torch.load(dir.parent.joinpath("pcam_pt", "valid_x.pt")).permute(0, 3, 1, 2)[:2_000]
y_val = torch.load(dir.parent.joinpath("pcam_pt", "valid_y.pt")).reshape(-1).to(torch.long)[:2_000]
print(f"Validation data shape: {tuple(X_val.shape)}, Validation labels shape: {tuple(y_val.shape)}")

n_epochs = 100
batch_size = 512 * 6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")

Dataset = PCAMdataset(
    x_path=dir / 'camelyonpatch_level_2_split_train_x.h5',
    y_path=dir / 'camelyonpatch_level_2_split_train_y.h5',
)

trainLoader = DataLoader(Dataset, batch_size=batch_size, shuffle=True)
model = Model().to(DEVICE)
loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Weight decay
# Schedular
loss_history = []
TOTAL_ITEMS = len(trainLoader.dataset) * n_epochs
ITEMS_IN_EPOCH = len(trainLoader.dataset)
start = time()

it = trainLoader.__len__() / batch_size 

# Training loop
for epoch in range(n_epochs):
    for i, (Xbatch, ybatch) in enumerate(trainLoader):
        Xbatch = Xbatch.to(DEVICE)
        ybatch = ybatch.to(DEVICE)
        y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val.to(DEVICE))
        val_loss = loss_fn(val_pred, y_val.to(DEVICE)).item()
    val_acc = accuracy_score(y_val.cpu().numpy(), val_pred.argmax(dim=1).cpu().numpy())
    model.train()

    progress = epoch * ITEMS_IN_EPOCH + i * len(Xbatch)
    stars = '=' * int(20 * (progress + 1) / TOTAL_ITEMS + 1)

    print(
    f"Train Epoch: {epoch} [{epoch * ITEMS_IN_EPOCH:>5d}/{TOTAL_ITEMS} [{stars:-<19}] "
    f"({100. * (progress+1) / TOTAL_ITEMS:2.0f}%)]\t"
    f"Loss: {loss.item():.6f}\tVal Loss: {val_loss:.6f}\tVal Acc: {val_acc:.4f}",
    end='\r'
    )
    loss_history.append([loss.item(), val_acc])
print("\n")  # For newline after the last epoch

run_time = time() - start
hours, rem = divmod(run_time, 3600)
minutes, seconds = divmod(rem, 60)

del Xbatch, ybatch  # Free memory

# Save the model
model.save(Path(__file__).parent / "model.pt")
# Calculate model size
num_params = sum(p.numel() for p in model.parameters())
param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
param_size_mb = param_size_bytes / (1024 ** 2)

# Evaluate the model
evaluator = Evaluate(model, DEVICE)
metrics = evaluator.evaluate(X_val, y_val)
metrics["training_time"] = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
metrics["model_size"] = f"{param_size_mb:.2f} MB"
metrics["num_params"] = num_params
evaluator.print_metrics(metrics)
evaluator.save_metrics(metrics, Path(__file__).parent / "metrics.txt")

# Plot training loss over epochs
# Note: We only have the last loss value from each epoch.
plt.plot(np.arange(len(loss_history)), [items[0] for items in loss_history], marker='o', label = "Traning Loss")
plt.plot(np.arange(len(loss_history)), [items[1] for items in loss_history], marker='o', label = "Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.tight_layout()
plt.savefig(Path(__file__).parent / "plot.png")