import os
import time
import psutil
import torch
import shutil
import subprocess


class EfficiencyMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.started = time.perf_counter()
        self.cpu_started = sum(self.process.cpu_times()[:2])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def finish(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-self.started
        utilization = None
        if torch.cuda.is_available():
            try:
                utilization = torch.cuda.utilization()
            except (RuntimeError, ModuleNotFoundError, AttributeError):
                utilization = None
            if utilization is None and shutil.which("nvidia-smi"):
                try:
                    result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=5,check=True)
                    utilization = float(result.stdout.strip().splitlines()[0])
                except (ValueError, IndexError, subprocess.SubprocessError):
                    utilization = None
        return {"round_seconds": elapsed, "rss_bytes": self.process.memory_info().rss,
                "process_cpu_core_percent": 100*(sum(self.process.cpu_times()[:2])-self.cpu_started)/max(elapsed, 1e-9),
                "gpu_utilization_snapshot_percent": utilization,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
