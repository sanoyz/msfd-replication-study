"""
Logging utilities for the MSFD replication pipeline.
"""
import os
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd


class CrystalLogger:
    """Clean logger that doesn't hijack stdout/stderr."""

    def __init__(self, log_dir: str = '/kaggle/working/'):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.timestamp = timestamp
        self.log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')
        self.metrics_file = os.path.join(log_dir, f'metrics_{timestamp}.csv')
        self.results_file = os.path.join(log_dir, f'results_{timestamp}.txt')
        self.ablation_file = os.path.join(log_dir, f'ablation_{timestamp}.txt')
        self.per_class_file = os.path.join(log_dir, f'per_class_report_{timestamp}.csv')

        self.metrics: List[Dict] = []
        self.ablation_results: Dict[str, Dict] = {}
        self.start_time = datetime.now()

        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MESSy CRYSTAL COMPLETE - TRAINING LOG\n")
            f.write(f"Started: {self.start_time.isoformat()}\n")
            f.write("=" * 80 + "\n\n")

    def log(self, message: str, level: str = "INFO") -> None:
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted = f"[{timestamp}] [{level}] {message}"
        print(formatted)
        with open(self.log_file, 'a') as f:
            f.write(formatted + "\n")
            f.flush()

    def log_metrics(self, experiment: str, epoch: int, metrics: Dict[str, float]) -> None:
        row = {'experiment': experiment, 'epoch': epoch,
               'timestamp': datetime.now().isoformat()}
        row.update(metrics)
        self.metrics.append(row)
        pd.DataFrame(self.metrics).to_csv(self.metrics_file, index=False)

    def log_ablation(self, name: str, accuracy: float, f1: float, params: int) -> None:
        self.ablation_results[name] = {'accuracy': accuracy, 'f1': f1, 'params': params}
        with open(self.ablation_file, 'a') as f:
            f.write(f"{name}: Acc={accuracy:.4f}, F1={f1:.4f}, Params={params:,}\n")

    def save_results(self, results: Dict[str, Any]) -> None:
        elapsed = (datetime.now() - self.start_time).total_seconds()
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)

        with open(self.results_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FINAL RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Completed: {datetime.now().isoformat()}\n")
            f.write(f"Duration: {hours}h {minutes}m\n\n")
            for key, value in results.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")


# Module-level singleton logger
logger = CrystalLogger('/kaggle/working/')