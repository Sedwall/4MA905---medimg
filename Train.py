import torch
from Model import Model
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib
from Evaluate import Evaluate
from time import time
from torch.utils.data import DataLoader
from PCAMdataset import PCAMdataset
from torchvision import transforms as T


def traning_run(train_data, test_data, batch_size, N_EPOCHS) -> tuple[nn.Module, dict, Evaluate]:

        train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, persistent_workers=True,
                            )
        val_dl   = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True,
                            )
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {DEVICE} DEVICE")
        torch.backends.cudnn.benchmark = True  # good for fixed-size images

        # ---- Model, loss, optim ----
        model = Model().to(DEVICE)              # Model must output logits of shape [B, 2]
        loss_fn = nn.CrossEntropyLoss()         # targets: int64 class ids (0/1)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        matplotlib.use("Agg")
        # ---- Training / Eval loops ----
        def run_epoch(loader, model, optimizer, n_epochs, train=True):
            model.train(train)
            total_loss, total_correct, total = 0.0, 0, 0

            for xb, fb, yb in loader:
                xb = xb.to(DEVICE, non_blocking=True)
                # CE expects long labels; if your dataset yields float 0/1, this cast fixes it
                yb = yb.to(DEVICE, non_blocking=True).long()
                fb = fb.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
                    logits = model(xb, fb)                  # expect [B, 2]
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
        start = time()
        # scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, end_factor=0.001, total_iters=n_epochs)

        start = time()
        
        # ---- Main Loop ----
        for epoch in range(1, N_EPOCHS + 1):
            train_loss, train_acc = run_epoch(train_dl, model, optimizer, epoch, train=True)
            val_loss, val_acc     = run_epoch(val_dl,   model, optimizer, epoch, train=False)

            # scheduler.step()

            print(f"Epoch {epoch:02d} | "
                f"train: loss {train_loss:.4f}, acc {train_acc:.4f} | "
                f"val: loss {val_loss:.4f}, acc {val_acc:.4f}")
        

        elapsed = time() - start
        h, rem = divmod(elapsed, 3600)
        m, s   = divmod(rem, 60)

        # Evaluate Model
        N_eval_points = 2_000
        eval_dl = DataLoader(test_data, batch_size=N_eval_points, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
        
        evaluator = Evaluate(model, eval_dl, DEVICE)
        # Calculate model size
        metrics = evaluator.evaluate()
        
        # Save and print metrics
        metrics["training_time"] = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
        return model, metrics, evaluator



if __name__ == '__main__':
    N_RUNS = 1
    BATCH_SIZE = 512
    N_EPOCHS = 1

    mean = [0.7008, 0.5384, 0.6916]
    std = [0.2350, 0.2774, 0.2129]
    # Define transforms
    train_tf = T.Compose([
        #T.Grayscale(num_output_channels=1), # convert to grayscale
        T.Normalize(mean, std), # standardize
    ])

    eval_tf  = T.Compose([
        T.Normalize(mean, std),
    ])

    # Setting up directory
    path_dir = Path(__file__).parent.parent.parent.joinpath('./dataset/')
    print(f'Using data from: {path_dir}')

    # Create datasets
    train_data = PCAMdataset(
        x_path=path_dir / 'pcam' / 'camelyonpatch_level_2_split_train_x.h5',
        y_path=path_dir / 'pcam' /'camelyonpatch_level_2_split_train_y.h5',
        feature_path=path_dir / 'pcam_HOG_h5' / 'train_x.h5',
        transform=train_tf
    )

    test_data = PCAMdataset(
        x_path=path_dir / 'pcam' / 'camelyonpatch_level_2_split_test_x.h5',
        y_path=path_dir / 'pcam' / 'camelyonpatch_level_2_split_test_y.h5',
        feature_path=path_dir / 'pcam_HOG_h5' / 'test_x.h5',
        transform=eval_tf
    )

    
    AVG_metrics = {}
    for i in range(N_RUNS):
        model, metrics, evaluator = traning_run(train_data, test_data, BATCH_SIZE, N_EPOCHS)

        evaluator.save_metrics(metrics, Path(__file__).parent / "runs" / f"metrics{i}.txt")

        for key, value in zip(metrics.keys(), metrics.values()):
            if key in AVG_metrics.keys() and isinstance(value, float):
                AVG_metrics[key] += value
            else:
                AVG_metrics[key] = value


    for key, value in zip(AVG_metrics.keys(), AVG_metrics.values()):
        if isinstance(value, float):
            AVG_metrics[key] /= N_RUNS

    print(f"{'*' * 7:s}Final Metrics{'*' * 7:s}")
    evaluator.print_metrics(metrics)
    evaluator.save_metrics(metrics, Path(__file__).parent/ f"final_metrics.txt")
