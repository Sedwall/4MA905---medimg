from torch.utils.data import DataLoader
from Utilities.PCAMdataset import PCAMdataset
from torchvision import transforms as T
from pathlib import Path
import torch
import numpy
import random
import os


class Data():
    def __init__(self, path_dir, seed, N_eval_points):
        self.g = torch.Generator()
        self.g.manual_seed(seed)
        self.N_eval_points = N_eval_points
        torch.manual_seed(seed)
        self.path_dir = path_dir

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # 0=all, 1=info, 2=warning, 3=error
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 0=all, 1=info, 2=warning, 3=error


    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)


    def get_data(self):
        mean_gray = 0.6043
        std_gray  = 0.2530
        mean = [0.7008, 0.5384, 0.6916]
        std = [0.2350, 0.2774, 0.2129]


        # Define transforms
        train_tf = T.Compose([
            #T.Grayscale(num_output_channels=1), # convert to grayscale
            T.Normalize(mean_gray, std_gray), # standardize
            # T.CenterCrop((32,32))
        ])

        eval_tf  = T.Compose([
            T.Normalize(mean_gray, std_gray),
            # T.CenterCrop((32,32))
        ])

        # Setting up directory
        

        # Create datasets
        train_data = PCAMdataset(
            x_path=self.path_dir / 'camelyonpatch_level_2_split_train_x.h5',
            y_path=self.path_dir /'camelyonpatch_level_2_split_train_y.h5',
            transform=train_tf
        )

        test_data = PCAMdataset(
            x_path=self.path_dir / 'camelyonpatch_level_2_split_test_x.h5',
            y_path=self.path_dir / 'camelyonpatch_level_2_split_test_y.h5',
            transform=eval_tf
        )

        train_dl = DataLoader(train_data, batch_size=512, shuffle=True,
                            num_workers=1, pin_memory=True, persistent_workers=True,
                            worker_init_fn=self.seed_worker,
                            generator=self.g,
                            )
        val_dl   = DataLoader(test_data, batch_size=512, shuffle=False,
                            num_workers=1, pin_memory=True, persistent_workers=True,
                            worker_init_fn=self.seed_worker,
                            generator=self.g,
                            )
        
        eval_dl = DataLoader(test_data, batch_size=self.N_eval_points, shuffle=False,
                            num_workers=1, pin_memory=True, persistent_workers=True,
                            worker_init_fn=self.seed_worker,
                            generator=self.g,
                            )
        
        return train_dl, val_dl, eval_dl
    


def run_epoch(loader, model, optimizer, loss_fn, DEVICE, train=True):
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