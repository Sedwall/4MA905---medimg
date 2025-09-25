class Evaluate:
    def __init__(self):
        pass

    def save_metrics(metrics, filepath):
        longest_name = max(len(m) for m in metrics.keys())
        with open(filepath, 'w') as f:
            f.write("=" * (longest_name + 14) + "\n")
            f.write("Model Evaluation Metrics\n")
            f.write("=" * (longest_name + 14) + "\n")
            for metric, value in metrics.items():
                if metric is 'training_time': f.write(f"{metric:<{longest_name}} : {value:>8}\n")
                else: f.write(f"{metric:<{longest_name}} : {value:>8.4f}\n")
            f.write("=" * (longest_name + 14) + "\n")
        print(f"Metrics saved to {filepath}")


    def print_metrics(metrics):
        longest_name = max(len(m) for m in metrics.keys())
        print("=" * (longest_name + 14))
        print("Model Evaluation Metrics")
        print("=" * (longest_name + 14))
        for metric, value in metrics.items():
            if metric is 'training_time': print(f"{metric:<{longest_name}} : {value:>8}")
            else: print(f"{metric:<{longest_name}} : {value:>8.4f}")
        print("=" * (longest_name + 14))
        print()