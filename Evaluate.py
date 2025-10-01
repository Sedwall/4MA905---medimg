import torch
from PCAMdataset import PCAMdataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

class Evaluate:
    def __init__(self, model, val_data: DataLoader, device):
        self.model = model
        self.device = device
        self.strObj = ['training_time', 'num_params', 'model_size']
        self.dataset = val_data

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            # Get one big batch from the DataLoader
            data_iter = iter(self.dataset)
            X, y = next(data_iter)
            X = X.to(self.device)
            y = y.to(self.device)

            y_pred = self.model(X)
            # om cross-entropy:
            # y_pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
            # y_true = y.cpu().numpy()

            # om BCE:
            logits = y_pred.view(-1)
            targets = y.view(-1)
            probs = torch.sigmoid(logits)
            y_pred_labels = (probs >= 0.5).long().cpu().numpy()
            y_true = targets.long().cpu().numpy()

            accuracy = accuracy_score(y_true, y_pred_labels)
            precision = precision_score(y_true, y_pred_labels, zero_division=0)
            recall = recall_score(y_true, y_pred_labels, zero_division=0)
            f1 = f1_score(y_true, y_pred_labels, zero_division=0)
            # For binary classification, use y_pred[:, 1] as probability for class 1
            #roc_auc = roc_auc_score(y_true, y_pred[:, 1].cpu().numpy()) if y_pred.shape[1] > 1 else 0.0
            
            # CrossEntropy: ta softmax
            if y_pred.ndim == 2 and y_pred.size(1) > 1:
                probs = torch.softmax(y_pred, dim=1)[:, 1].detach().cpu().numpy()
            else:
                # BCE: ta sigmoid(logit)
                probs = torch.sigmoid(y_pred.view(-1)).detach().cpu().numpy()

            y_true = y.view(-1).long().cpu().numpy()
            try:
                roc_auc = roc_auc_score(y_true, probs)
            except ValueError:
                roc_auc = float("nan")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }
    
    def save_metrics(self, metrics, filepath):
        longest_name = max(len(m) for m in metrics.keys())
        with open(filepath, 'w') as f:
            f.write("=" * (longest_name + 14) + "\n")
            f.write("Model Evaluation Metrics\n")
            f.write("=" * (longest_name + 14) + "\n")
            for metric, value in metrics.items():
                if metric in self.strObj: f.write(f"{metric:<{longest_name}} : {value:>8}\n")
                else: f.write(f"{metric:<{longest_name}} : {value:>8.4f}\n")
            f.write("=" * (longest_name + 14) + "\n")
        print(f"Metrics saved to {filepath}")


    def print_metrics(self, metrics):
        longest_name = max(len(m) for m in metrics.keys())
        print("=" * (longest_name + 14))
        print("Model Evaluation Metrics")
        print("=" * (longest_name + 14))
        for metric, value in metrics.items():
            if metric in self.strObj: print(f"{metric:<{longest_name}} : {value:>8}")
            else: print(f"{metric:<{longest_name}} : {value:>8.4f}")
        print("=" * (longest_name + 14))
        print()