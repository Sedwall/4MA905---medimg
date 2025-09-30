import torch
from Utilities.Model import Model
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import optuna
from optuna.trial import TrialState
from Utilities.Utils import Data, run_epoch


torch.backends.cudnn.benchmark = True  # good for fixed-size images

def define_model(trial: optuna.Trial, device):
    channels = trial.suggest_categorical("channels", [8, 16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    model = Model(channels, dropout).to(device)
    return model


def define_optimizer(trial: optuna.Trial, model):
    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    return optimizer


def define_scheduler(trial: optuna.Trial, optimizer):
    # Let Optuna choose which scheduler to use
    scheduler_name = trial.suggest_categorical(
        "scheduler", ["StepLR", "CosineAnnealingLR"]
    )

    if scheduler_name == "StepLR":
        step_size = trial.suggest_int("step_size", 5, 30)
        gamma = trial.suggest_float("gamma", 0.1, 0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_name == "CosineAnnealingLR":
        T_max = trial.suggest_int("T_max", 10, 50)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    elif scheduler_name == "ExponentialLR":
        gamma = trial.suggest_float("gamma", 0.85, 0.999)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    else:
        scheduler = None  # in case you want to run without a scheduler

    return scheduler


def objective(trial: optuna.Trial):
    n_epochs = 1

    path = Path(__file__).parent.parent.parent.joinpath('./dataset/pcam/')
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ---- Model, loss, optim ----
    model = define_model(trial, DEVICE)            # Model must output logits of shape [B, 2]
    optimizer = define_optimizer(trial, model)
    loss_fn = nn.CrossEntropyLoss()         # targets: int64 class ids (0/1)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, end_factor=0.001, total_iters=n_epochs)
    
    N = trial.number
    train_dl, val_dl, final_eval_dl = Data(path, seed=0, N_eval_points=2_000).get_data()
    writer = SummaryWriter(Path(__file__).parent.joinpath('runs', f'experiment{N}'))
    # ---- Main Loop ----
    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = run_epoch(train_dl, model, optimizer, loss_fn, DEVICE, train=True)
        val_loss, val_acc     = run_epoch(val_dl, model, optimizer, loss_fn, DEVICE, train=False)

        scheduler.step()

        print(f"Epoch {epoch:02d} | "
            f"train: loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"val: loss {val_loss:.4f}, acc {val_acc:.4f}", end="\r")
        
        # ---- TensorBoard logging ----
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("LearningRate", current_lr, epoch)

        trial.report(val_acc, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(val_acc)


if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, timeout=60*60*3, n_jobs=2)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    
    filepath = Path(__file__).parent / 'Results' / 'HyperResult.txt'
    print("  Value: ", trial.value)
    longest_name = max(len(m) for m, _ in trial.params.items())
    with open(filepath, 'w') as f:
        f.write("=" * (longest_name + 14) + "\n")
        f.write("Model Evaluation Metrics\n")
        f.write("=" * (longest_name + 14) + "\n")
        for metric, value in trial.params.items():
            if not isinstance(metric, float): f.write(f"{metric:<{longest_name}} : {value:>8}\n")
            else: f.write(f"{metric:<{longest_name}} : {value:>8.4f}\n")
        f.write("=" * (longest_name + 14) + "\n")
    print(f"Metrics saved to {filepath}")