import numpy as np
import torch
from Model import Model
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from Evaluate import Evaluate
from time import time
from torch.utils.data import DataLoader
from PCAMdataset import PCAMdataset
from torchvision import transforms as T

if __name__ == '__main__':
    mean_gray = 0.6043
    std_gray  = 0.2530
    mean = [0.7008, 0.5384, 0.6916]
    std = [0.2350, 0.2774, 0.2129]

    # Define transforms
    train_tf = T.Compose([
        #T.Grayscale(num_output_channels=1), # convert to grayscale
        T.Normalize(mean_gray, std_gray), # standardize
        T.CenterCrop(32)
    ])

    eval_tf  = T.Compose([
        T.Normalize(mean_gray, std_gray),
        T.CenterCrop(32)
    ])

    # Setting up directory
    path_dir = Path(__file__).parent.parent.parent.joinpath('./dataset/pcam/')
    print(f'Using data from: {path_dir}')

    # Create datasets
    train_data = PCAMdataset(
        x_path=path_dir / 'camelyonpatch_level_2_split_train_x.h5',
        y_path=path_dir /'camelyonpatch_level_2_split_train_y.h5',
        transform=train_tf
    )

    test_data = PCAMdataset(
        x_path=path_dir / 'camelyonpatch_level_2_split_test_x.h5',
        y_path=path_dir / 'camelyonpatch_level_2_split_test_y.h5',
        transform=eval_tf
    )

    train_dl = DataLoader(train_data, batch_size=512, shuffle=True,
                        num_workers=4, pin_memory=True, persistent_workers=True)
    val_dl   = DataLoader(test_data, batch_size=512, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {DEVICE} DEVICE")
    torch.backends.cudnn.benchmark = True  # good for fixed-size images

    # ---- Model, loss, optim ----
    model = Model().to(DEVICE)              # Model must output logits of shape [B, 2]
    loss_fn = nn.CrossEntropyLoss()         # targets: int64 class ids (0/1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


    matplotlib.use("Agg")
    train_loss_history = []
    val_loss_history   = []
    # ---- Training / Eval loops ----
    def run_epoch(loader, train=True):
        model.train(train)
        total_loss, total_correct, total = 0.0, 0, 0

        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            # CE expects long labels; if your dataset yields float 0/1, this cast fixes it
            yb = yb.to(DEVICE, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
                logits = model(xb)                  # expect [B, 2]
                # sanity guard (remove if bothersome)
                if logits.ndim != 2 or logits.size(1) != 2:
                    raise RuntimeError(f"Model must return [B,2] for CrossEntropyLoss, got {tuple(logits.shape)}")
                loss = loss_fn(logits, yb)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            with torch.no_grad():
                preds = logits.argmax(dim=1)       # [B]
                total_correct += (preds == yb).sum().item()
                total += xb.size(0)

        avg_loss = total_loss / max(total, 1)
        acc = total_correct / max(total, 1)
        return avg_loss, acc

    # ---- Fit ----
    n_epochs = 15
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

    # Evaluate Model
    N_eval_points = 2_000
    eval_dl = DataLoader(test_data, batch_size=N_eval_points, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)
    
    evaluator = Evaluate(model, eval_dl, DEVICE)
    metrics = evaluator.evaluate()

    # Calculate model size
    num_params = sum(p.numel() for p in model.parameters())
    param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_size_mb = param_size_bytes / (1024 ** 2)
    metrics["model_size"] = f"{param_size_mb:.2f} MB"
    metrics["num_params"] = num_params

    # Save and print metrics
    metrics["training_time"] = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    evaluator.print_metrics(metrics)
    evaluator.save_metrics(metrics, Path(__file__).parent / "metrics.txt")