
import numpy as np
import time
import matplotlib.pyplot as plt

def gauss_seidel(f):
    """
    Performs one iteration of Gauss-Seidel relaxation.
    Updates f in-place.
    """
    for i in range(1, f.shape[0]-1):
        for j in range(1, f.shape[1]-1):
            f[i, j] = 0.25 * (f[i, j+1] + f[i, j-1] + f[i+1, j] + f[i-1, j])
    return f

def run_benchmark():
    grid_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    execution_times = []

    print(f"{'Grid Size (N)':<15} | {'Time (seconds)':<15}")
    print("-" * 35)

    for N in grid_sizes:
        f = np.random.rand(N, N)

        f[0, :] = 0
        f[-1, :] = 0
        f[:, 0] = 0
        f[:, -1] = 0

        start_time = time.time()
        
        for _ in range(1000):
            gauss_seidel(f)
            
        end_time = time.time()
        elapsed = end_time - start_time
        
        execution_times.append(elapsed)
        print(f"{N:<15} | {elapsed:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(grid_sizes, execution_times, marker='o', linestyle='-', color='b')
    plt.title('Gauss-Seidel Performance: Python Loops')
    plt.xlabel('Grid Size (N)')
    plt.ylabel('Time for 1000 Iterations (seconds)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_benchmark()
