import psutil
import time
import threading
import matplotlib.pyplot as plt
import statistics

class CpuProfiler:

    def __init__(self, interval=0.1):
        self.interval = interval
        self.records = []
        self.running = False
        self.thread = None

    def _monitor(self):
        while self.running:
            cpu_usage = psutil.cpu_percent(interval=self.interval, percpu=True)
            timestamp = time.time()
            self.records.append((timestamp, cpu_usage))

    def start(self):
        self.running = True
        self.records = []
        self.thread = threading.Thread(target=self._monitor)
        print("cpu_profiler started")
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        print("cpu_profiler stopped")
        return self.records

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        self.report()

    def report(self):
        if not self.records:
            print("No data recorded")
            return

        print(f"Total records collected: {len(self.records)}")
        times = [r[0] - self.records[0][0] for r in self.records]
        num_cores = len(self.records[0][1])
        core_data = {f"Core {i}": [r[1][i] for r in self.records] for i in range(num_cores)}

        print("\nCPU Usage Summary (%)")
        print(f"{'Core':<10} {'Avg':<10} {'Max':<10}")
        print("-" * 30)
        for i in range(num_cores):
            vals = core_data[f"Core {i}"]
            avg_val = statistics.mean(vals)
            max_val = max(vals)
            print(f"Core {i:<5} {avg_val:<10.1f} {max_val:<10.1f}")

        plt.figure(figsize=(10, 6))
        for core_name, values in core_data.items():
            plt.plot(times, values, label=core_name)

        plt.title(f"CPU Usage per Core (Interval: {self.interval}s)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("CPU Usage (%)")
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("cpu_profile.png")
        print("\nPlot saved to 'cpu_profile.png'")
        plt.show()

if __name__ == "__main__":
    profiler = CpuProfiler(interval=0.01)
    with profiler:
        import ass2
        ass2.run_experiment(1)
