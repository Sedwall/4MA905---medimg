from torch import nn
import torch
from torch.utils.data import DataLoader
from Utils.Evaluate import Evaluate
from time import time
import numpy as np
from pathlib import Path
from torch import nn, optim
from matplotlib import pyplot as plt



def run_experiment(Model, train_data, test_data, BATCH_SIZE, N_EPOCHS, N_RUNS):
    ####### Traning Of Model #######
    AVG_metrics = {}
    loss_hist = []
    for i in range(N_RUNS):
        model = Model()

        ## Define loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, end_factor=0.001, total_iters=N_EPOCHS)


        model, metrics, evaluator, losses = traning_run(model, train_data, test_data, loss_fn, optimizer, BATCH_SIZE, N_EPOCHS, scheduler)

        loss_hist.append(losses)
        if not Path(__file__).parent.joinpath("runs").exists():
            Path(__file__).parent.joinpath("runs").mkdir()
        evaluator.save_metrics(metrics, Path(__file__).parent / "runs" / f"metrics{i}.txt")

        for key, value in zip(metrics.keys(), metrics.values()):
            if key in AVG_metrics.keys():
                AVG_metrics[key].append(value)
            else:
                AVG_metrics[key] = [value]

    
    plot_losses(loss_hist)

    # Calculate and print average metrics
    metrics_avg(evaluator, AVG_metrics, __file__)


########## Training Loop Function ##########
#-- Helper function for the deep ML models --
def traning_run(model, train_data, test_data, loss_fn, optimizer, batch_size, N_EPOCHS, scheduler=None) -> tuple[nn.Module, dict, Evaluate, np.ndarray]:

        train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                            num_workers=8, pin_memory=True, persistent_workers=True,
                            )
        val_dl   = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True,
                            )
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {DEVICE} DEVICE")
        torch.backends.cudnn.benchmark = True  # good for fixed-size images

        # ---- Model, loss, optim ----
        model = model.to(DEVICE) # Model must output logits of shape [B, 2]

        # ---- Training / Eval loops ----
        def run_epoch(loader, model, optimizer, train=True):
            model.train(train)
            total_loss, total_correct, total = 0.0, 0, 0

            for xb, fb, yb in loader:
                xb = xb.to(DEVICE, non_blocking=True).float()
                # CE expects long labels; if your dataset yields float 0/1, this cast fixes it
                yb = yb.to(DEVICE, non_blocking=True).long()
                fb = fb.to(DEVICE, non_blocking=True).float()

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

        loss_hist = {}
        # ---- Main Loop ----
        for epoch in range(1, N_EPOCHS + 1):
            train_loss, train_acc = run_epoch(train_dl, model, optimizer, train=True)
            val_loss, val_acc     = run_epoch(val_dl,   model, optimizer, train=False)
            
            if scheduler != None:
                scheduler.step()
            
            if loss_hist.keys():
                loss_hist['val_loss'].append(val_loss)
                loss_hist['train_loss'].append(train_loss)
            else:
                loss_hist['val_loss'] = [val_loss]
                loss_hist['train_loss'] = [train_loss]

            print(f"Epoch {epoch:02d} | "
                f"train: loss {train_loss:.4f}, acc {train_acc:.4f} | "
                f"val: loss {val_loss:.4f}, acc {val_acc:.4f}")
        

        elapsed = time() - start

        # Evaluate Model
        eval_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
        
        evaluator = Evaluate(model, eval_dl, DEVICE)
        # Calculate model size
        metrics = evaluator.evaluate()
        metrics['training_time'] = elapsed
        return model, metrics, evaluator, loss_hist



def metrics_avg(evaluator: Evaluate, AVG_metrics: dict, file):
    AUC_STD = np.std(AVG_metrics['roc_auc'])
    ACC_STD = np.std(AVG_metrics['accuracy'])
    
    for key, value in list(AVG_metrics.items()):
        if isinstance(value, float):
            AVG_metrics[key] = np.mean(value).item()


    AVG_metrics['roc_auc_std'] = AUC_STD.item()
    AVG_metrics['accuracy_std'] = ACC_STD.item()

    print(f"{'*' * 7:s}Final Metrics{'*' * 7:s}")
    evaluator.print_metrics(AVG_metrics)
    file_name = str(file).split('/')[-1].split('.')[0]
    evaluator.save_metrics(AVG_metrics, Path(file).parent.parent / f"{file_name}_final_metrics.txt")


def plot_losses(loss_hist):
    for i, data in enumerate(loss_hist):
        train_loss = data['train_loss']
        val_loss = data['val_loss']
        plt.plot(train_loss, label = f'TL {i}')
        plt.plot(val_loss, label = f'VL {i}')
    plt.legend()
    plt.savefig(Path(__file__).parent.parent.parent / f"Loss_Plot.png")