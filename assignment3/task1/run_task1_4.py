import numpy as np
import time
import matplotlib.pyplot as plt

import task1_4

def gauss_seidel_python(f):
    for i in range(1, f.shape[0]-1):
        for j in range(1, f.shape[1]-1):
            f[i, j] = 0.25 * (f[i, j+1] + f[i, j-1] + f[i+1, j] + f[i-1, j])
    return f

def run_benchmark():
    grid_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    python_times = []
    cython_times = []

    print(f"{'N':<5} | {'Python (s)':<12} | {'Cython (s)':<12} | {'Speedup'}")
    print("-" * 50)

    for N in grid_sizes:
        f_base = np.random.rand(N, N)
        f_base[0, :] = 0; f_base[-1, :] = 0; f_base[:, 0] = 0; f_base[:, -1] = 0
        
        f_py = f_base.copy()
        f_cy = f_base.copy()

        start = time.time()
        for _ in range(1000):
            gauss_seidel_python(f_py)
        py_time = time.time() - start
        python_times.append(py_time)

        start = time.time()
        for _ in range(1000):
            task1_4.gauss_seidel_cython(f_cy)
        cy_time = time.time() - start
        cython_times.append(cy_time)
        
        if cy_time < 1e-9: cy_time = 1e-9
            
        speedup = py_time / cy_time
        print(f"{N:<5} | {py_time:<12.4f} | {cy_time:<12.4f} | {speedup:.1f}x")

    plt.figure(figsize=(10, 6))
    plt.plot(grid_sizes, python_times, 'o-', color='red', label='Original Python')
    plt.plot(grid_sizes, cython_times, 's-', color='blue', label='Optimized Cython')
    
    plt.title('Performance Comparison: Python vs Cython')
    plt.xlabel('Grid Size (N)')
    plt.ylabel('Time for 1000 Iterations (seconds)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.show()

if __name__ == "__main__":
    try:
        run_benchmark()
    except AttributeError:
        print("\nERROR: Could not find function 'gauss_seidel_cython' in task1_4.")
        print("Please check your .pyx file and ensure the function name matches.")