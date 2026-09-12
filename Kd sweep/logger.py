"""
Logging utilities for the MSFD KD sweep.
"""
import os
from datetime import datetime
from typing import Any, Dict


class SimpleLogger:
    def __init__(self, output_dir='/kaggle/working/'):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(output_dir, f'kd_sweep_log_{timestamp}.txt')
        self.results_file = os.path.join(output_dir, f'kd_sweep_results_{timestamp}.txt')
        self.start_time = datetime.now()

        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("KD SWEEP LOG\n")
            f.write(f"Started: {self.start_time.isoformat()}\n")
            f.write("=" * 80 + "\n\n")

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted = f"[{timestamp}] [{level}] {message}"
        print(formatted)
        with open(self.log_file, 'a') as f:
            f.write(formatted + "\n")
            f.flush()

    def save_results(self, results: Dict[str, Any]):
        with open(self.results_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("KD SWEEP RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Completed: {datetime.now().isoformat()}\n\n")
            for key, value in results.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                elif isinstance(value, list):
                    f.write(f"{key}:\n")
                    for item in value:
                        f.write(f"  {item}\n")
                else:
                    f.write(f"{key}: {value}\n")