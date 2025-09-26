import numpy as np
import torch
from Model import Model
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from Evaluate import Evaluate
from time import time
from dataset_test.PCAMdataset import PCAMdataset
from torchvision import transforms as T


mean_gray = 0.6043
std_gray  = 0.2530
mean = [0.7008, 0.5384, 0.6916]
std = [0.2350, 0.2774, 0.2129]

# Define transforms
train_tf = T.Compose([
    #T.Grayscale(num_output_channels=1), # convert to grayscale
    T.Normalize(mean_gray, std_gray), # standardize
])

eval_tf  = T.Compose([
    T.Normalize(mean_gray, std_gray),
])

# Create datasets
train_data = PCAMdataset(
    x_path='/home/p3_medimg/Project/dataset/pcam/camelyonpatch_level_2_split_train_x.h5',
    y_path='/home/p3_medimg/Project/dataset/pcam/camelyonpatch_level_2_split_train_y.h5',
    transform=train_tf
)

test_data = PCAMdataset(
    x_path='/home/p3_medimg/Project/dataset/pcam/camelyonpatch_level_2_split_test_x.h5',
    y_path='/home/p3_medimg/Project/dataset/pcam/camelyonpatch_level_2_split_test_y.h5',
    transform=eval_tf
)

train_dl = DataLoader(train_data, batch_size=512, shuffle=True,
                      num_workers=4, pin_memory=True, persistent_workers=True)
val_dl   = DataLoader(test_data, batch_size=512, shuffle=False,
                      num_workers=4, pin_memory=True, persistent_workers=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
torch.backends.cudnn.benchmark = True  # good for fixed-size images

<<<<<<< HEAD
# ---- Model, loss, optim ----
model = Model().to(device)              # Model must output logits of shape [B, 2]
loss_fn = nn.CrossEntropyLoss()         # targets: int64 class ids (0/1)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
=======
    # ---- Model, loss, optim ----
    model = Model().to(DEVICE)              # Model must output logits of shape [B, 2]
    loss_fn = nn.CrossEntropyLoss()         # targets: int64 class ids (0/1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

>>>>>>> c6dbb7f (Soon to be state of the art Model, Model acc. 87%)

matplotlib.use("Agg")
train_loss_history = []
val_loss_history   = []
# ---- Training / Eval loops ----
def run_epoch(loader, train=True):
    model.train(train)
    total_loss, total_correct, total = 0.0, 0, 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        # CE expects long labels; if your dataset yields float 0/1, this cast fixes it
        yb = yb.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            logits = model(xb)                  # expect [B, 2]
            # sanity guard (remove if bothersome)
            if logits.ndim != 2 or logits.size(1) != 2:
                raise RuntimeError(f"Model must return [B,2] for CrossEntropyLoss, got {tuple(logits.shape)}")
            loss = loss_fn(logits, yb)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * xb.size(0)
        with torch.no_grad():
            preds = logits.argmax(dim=1)       # [B]
            total_correct += (preds == yb).sum().item()
            total += xb.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    return avg_loss, acc

# ---- Fit ----
n_epochs = 10
start = time()

for epoch in range(1, n_epochs + 1):
    train_loss, train_acc = run_epoch(train_dl, train=True)
    val_loss, val_acc     = run_epoch(val_dl,   train=False)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    print(f"Epoch {epoch:02d} | "
          f"train: loss {train_loss:.4f}, acc {train_acc:.4f} | "
          f"val: loss {val_loss:.4f}, acc {val_acc:.4f}")

elapsed = time() - start
h, rem = divmod(elapsed, 3600)
m, s   = divmod(rem, 60)
print(f"Training time: {int(h):02d}:{int(m):02d}:{int(s):02d}")

# ---- Save checkpoint ----
torch.save({
    "model_state": model.state_dict(),
    "epoch": n_epochs,
}, "model.pt")
print("Saved to model.pt")

plt.figure()
plt.plot(range(1, len(train_loss_history)+1), train_loss_history, label="train")
plt.plot(range(1, len(val_loss_history)+1),   val_loss_history,   label="val")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss per epoch")
plt.legend(); plt.tight_layout()
plt.savefig("plot.png", dpi=150)
print("Saved loss plot to plot.png")



# del X, y  # Free memory
# del Xbatch, ybatch  # Free memory

# Evaluate the model
# evaluator = Evaluate(model, DEVICE)
# metrics = evaluator.evaluate(X_val, y_val)
# metrics["training_time"] = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
# evaluator.print_metrics(metrics)
# evaluator.save_metrics(metrics, Path(__file__).parent / "metrics.txt")

# # Plot training loss over epochs
# # Note: We only have the last loss value from each epoch.
# plt.plot(np.arange(n_epochs), loss_history, marker='o' )
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss over Epochs")
# plt.tight_layout()
# plt.savefig(Path(__file__).parent / "plot.png")