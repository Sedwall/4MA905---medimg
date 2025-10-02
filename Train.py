import numpy as np
import torch
import os
from Model import PCamCNN, Model
import torch.nn as nn
import torch.optim as optim
import numpy
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from Evaluate import Evaluate
from time import time
from torch.utils.data import DataLoader
from PCAMdataset import PCAMdataset
from torchvision import transforms as T
import torchvision.transforms.v2 as TV2
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import InterpolationMode as IM

# set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE} DEVICE")
torch.backends.cudnn.benchmark = True  # kan ge snabbare träning för fasta input-storlekar
args = None

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

def build_dataloaders(args, DEVICE, g=g):
    # mean and std computed from training set
    # for grayscale and RGB images
    mean_gray = 0.6043
    std_gray  = 0.2530
    mean_rgb = [0.7008, 0.5384, 0.6916]
    std_rgb = [0.2350, 0.2774, 0.2129]
    

    # Define gpu transforms
    gpu_train_tf = TV2.Compose([
        TV2.RandomResizedCrop(
            size=(96, 96),
            scale=(0.9, 1.0),                    # lite bredare zoom-range
            interpolation=IM.BICUBIC,
            antialias=True,
        ),
        TV2.RandomHorizontalFlip(),
        TV2.RandomVerticalFlip(),
        TV2.RandomRotation(
            degrees=15,
            interpolation=IM.BILINEAR,
            fill=0,                              # eller tuple med din bakgrundsmedel
        ),
        TV2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.03),
        TV2.RandomApply([TV2.GaussianBlur(kernel_size=3)], p=0.2),
        TV2.ToDtype(torch.float32, scale=True),
        TV2.Normalize(mean=mean_rgb, std=std_rgb),
    ]).to(DEVICE)

    gpu_eval_tf = TV2.Compose([
        # TV2.Resize(size=96),
        # TV2.CenterCrop(size=96),
        TV2.ToDtype(torch.float32, scale=True),
        TV2.Normalize(mean=mean_rgb, std=std_rgb),
    ]).to(DEVICE)

     # Setting up directory
    path_dir = Path(__file__).parent.parent.parent.joinpath('./dataset/pcam/')
    #path_dir = Path('/home/helga/projects/4MA905 DL project/data/pcam/')
    print(f'Using data from: {path_dir}')

    # Create datasets
    train_data = PCAMdataset(
        x_path=path_dir / 'camelyonpatch_level_2_split_train_x.h5',
        y_path=path_dir /'camelyonpatch_level_2_split_train_y.h5',
        transform=None  # inga CPU-transforms
    )

    test_data = PCAMdataset(
        x_path=path_dir / 'camelyonpatch_level_2_split_test_x.h5',
        y_path=path_dir / 'camelyonpatch_level_2_split_test_y.h5',
        transform=None  # inga CPU-transforms
    )
    
    train_dl = DataLoader(train_data, batch_size=512, shuffle=True,
                        num_workers=4, pin_memory=True, persistent_workers=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                        )
    val_dl   = DataLoader(test_data, batch_size=512, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                        )
    return train_dl, val_dl, gpu_train_tf, gpu_eval_tf, test_data

def build_model_and_optimizer(args, DEVICE):
    model = Model().to(DEVICE)             # Model must output logits of shape [B, 2]
    #set_bn_momentum(model, m=0.005)
    loss_fn = nn.CrossEntropyLoss()         # targets: int64 class ids (0/1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) 
    return model, loss_fn, optimizer

def set_bn_momentum(model, m=0.01):
    import torch.nn as nn
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.momentum = m

if __name__ == '__main__':
    # ---- Data ----
    train_dl, val_dl, gpu_train_tf, gpu_eval_tf, test_data = build_dataloaders(args, DEVICE)

    # ---- Model, loss, optim ----
    model, loss_fn, optimizer = build_model_and_optimizer(args, DEVICE)

    train_loss_history = []
    val_loss_history   = []
    # ---- Training / Eval loops ----
    def run_epoch(loader, epoch, train=True):
        model.train(train)
        total_loss = total_correct = total = 0

        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True).long().view(-1)  # CE targets: [B] Long

            # GPU-transforms i FP32 – enklast: kör dem UTAN autocast
            if train:
                xb = gpu_train_tf(xb)   # se till att gpu_train_tf.to(DEVICE) är gjort där du bygger den
            else:
                xb = gpu_eval_tf(xb)

            optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                # Nya autocast-API:t
                with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
                    logits = model(xb)              # [B, K]
                    loss = loss_fn(logits, yb)      # yb: [B] Long

                if train:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * xb.size(0)
            with torch.no_grad():
                preds = logits.argmax(dim=1)    # [B]
                total_correct += (preds == yb).sum().item()
                total += xb.size(0)

        avg_loss = total_loss / max(total, 1)
        acc = total_correct / max(total, 1)
        return avg_loss, acc

    def get_lr(optim):
    # funkar oavsett scheduler; tar första param_group
        return optim.param_groups[0]["lr"]
    
    # ---- Fit ----
    n_epochs = 40

    start = time()
    #scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
    #                          threshold=1e-4)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=n_epochs)

    # ---- Main Loop ----
    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = run_epoch(train_dl, epoch, train=True)
        val_loss, val_acc     = run_epoch(val_dl,   epoch, train=False)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        scheduler.step()
        #scheduler.step(val_loss)
        lr = get_lr(optimizer)
        
        print(f"Epoch {epoch:02d} | "
            f"train: loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"val: loss {val_loss:.4f}, acc {val_acc:.4f}"
            f" | lr {lr:.6f}")
        

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
                        num_workers=4, pin_memory=True, persistent_workers=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                        )

    evaluator = Evaluate(model, eval_dl, DEVICE, eval_tf=gpu_eval_tf)
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