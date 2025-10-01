import numpy as np
import torch
import os
from Model import PCamCNN
import torch.nn as nn
import torch.optim as optim
import numpy
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from Evaluate import Evaluate
from time import time
from torch.utils.data import DataLoader
from PCAMdataset import PCAMdataset
from torchvision import transforms as T
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
import argparse

parser = argparse.ArgumentParser()
# ... dina andra args ...
parser.add_argument("--profile", action="store_true",
                    help="Profilera run_epoch med PyTorch Profiler (CPU+CUDA)")
args = parser.parse_args()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


if __name__ == '__main__':
    # mean and std computed from training set
    # for grayscale and RGB images
    mean_gray = 0.6043
    std_gray  = 0.2530
    mean_rgb = [0.7008, 0.5384, 0.6916]
    std_rgb = [0.2350, 0.2774, 0.2129]


    # Define transforms
    train_tf = T.Compose([
        T.RandomResizedCrop(size=96, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
        T.RandomApply([T.GaussianBlur(3)], p=0.2),
        # Normalisering
        T.Normalize(mean=mean_rgb, std=std_rgb),
    ])
    eval_tf = T.Compose([
        T.Resize(size=96),
        T.CenterCrop(size=96),
        T.Normalize(mean=mean_rgb, std=std_rgb),
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
                        num_workers=4, pin_memory=True, persistent_workers=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                        )
    val_dl   = DataLoader(test_data, batch_size=512, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                        )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {DEVICE} DEVICE")
    torch.backends.cudnn.benchmark = True  # good for fixed-size images

    # ---- Model, loss, optim ----
    model = PCamCNN().to(DEVICE)              # Model must output logits of shape [B, 2]
    loss_fn = nn.BCEWithLogitsLoss()         # targets: int64 class ids (0/1)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

    matplotlib.use("Agg")
    train_loss_history = []
    val_loss_history   = []
    # ---- Training / Eval loops ----
    def run_epoch(loader, epoch, train=True):
        model.train(train)
        total_loss, total_correct, total = 0.0, 0, 0

        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            # CE expects long labels; if your dataset yields float 0/1, this cast fixes it
            yb = yb.to(DEVICE, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
                logits = model(xb)                  # expect [B, 2]
                # sanity guard (remove if bothersome)
                # if logits.ndim != 2 or logits.size(1) != 2:
                #     raise RuntimeError(f"Model must return [B,2] for CrossEntropyLoss, got {tuple(logits.shape)}")
                loss = loss_fn(logits, yb)

            if train:
                loss.backward()
                optimizer.step()
                

            total_loss += loss.item() * xb.size(0)
            with torch.no_grad():
                # preds = logits.argmax(dim=1)       # [B]
                # total_correct += (preds == yb).sum().item()
                # total += xb.size(0)

                probs = torch.sigmoid(logits)               # [B]
                preds = (probs >= 0.5).long()               # [B]
                total_correct += (preds == yb.long().view(-1)).sum().item()
                total += xb.size(0)

        avg_loss = total_loss / max(total, 1)
        acc = total_correct / max(total, 1)
        return avg_loss, acc

    # ---- Fit ----
    n_epochs = 10

    start = time()
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, end_factor=0.001, total_iters=n_epochs)

    if args.profile:
        model.train(True)
        steps_to_profile = 120  # kort men representativt
        prof_sched = schedule(wait=10, warmup=10, active=100, repeat=1)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=prof_sched,
            on_trace_ready=tensorboard_trace_handler("tb_logs/pcam"),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,     # sätt True om du vill ha call stacks (dyrare)
        ) as prof:
            it = iter(train_dl)
            for step in range(steps_to_profile):
                xb, yb = next(it)
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True).float()  # BCE

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=(DEVICE.type=="cuda")):
                    logits = model(xb)
                    loss = loss_fn(logits.view(-1), yb.view(-1))
                loss.backward()
                optimizer.step()

                prof.step()  # VIKTIGT: steppa per träningssteg

        # Skriv en snabb sammanfattning till terminalen
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
        import sys; sys.exit(0)  # avsluta efter profilering (ta bort om du vill fortsätta träna)

    # ---- Main Loop ----
    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = run_epoch(train_dl, epoch, train=True)
        val_loss, val_acc     = run_epoch(val_dl,   epoch, train=False)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        scheduler.step()

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
                        num_workers=4, pin_memory=True, persistent_workers=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                        )
    
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