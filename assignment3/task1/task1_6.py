import cupy as cp
import time
import matplotlib.pyplot as plt

import h5py
import numpy as np

def save_to_hdf5(grid, filename):
    """
    Saves a GPU grid (PyTorch or CuPy) to an HDF5 file on disk.
    Automatically handles moving data from GPU -> CPU.
    """
    print(f"Saving grid to {filename}...")
    
    # 1. Convert PyTorch Tensor -> NumPy (CPU)
    if hasattr(grid, 'cpu'):
        grid_cpu = grid.cpu().numpy()
        
    # 2. Convert CuPy Array -> NumPy (CPU)
    elif hasattr(grid, 'get'):
        grid_cpu = grid.get() # .get() moves data from GPU to CPU
        
    # 3. Already NumPy? Just use it.
    else:
        grid_cpu = grid

    # 4. Save using h5py
    with h5py.File(filename, "w") as f:
        dset = f.create_dataset("newgrid", data=grid_cpu)
        dset.attrs["description"] = "Final simulation state"
        dset.attrs["grid_size"] = grid_cpu.shape[0]
        
    print("Save complete!")

def gauss_seidel_cupy(f):
    """
    Vectorized implementation using CuPy slicing.
    This mimics the NumPy syntax but runs on the GPU.
    """
    f[1:-1, 1:-1] = 0.25 * (f[0:-2, 1:-1] + f[2:, 1:-1] + 
                            f[1:-1, 0:-2] + f[1:-1, 2:])
    
    f[0, :] = 0; f[-1, :] = 0; f[:, 0] = 0; f[:, -1] = 0
    
    return f

def run_benchmark():
    grid_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000, 2000, 5000, 10000]
    execution_times = []

    print(f"{'N':<10} | {'Time (s)':<15}")
    print("-" * 30)

    for N in grid_sizes:
        f = cp.random.rand(N, N, dtype=cp.float32)
        
        f[0, :] = 0; f[-1, :] = 0; f[:, 0] = 0; f[:, -1] = 0
        
        # warmup
        for _ in range(10):
            gauss_seidel_cupy(f)
        cp.cuda.Stream.null.synchronize() # wait for warmup

        start_time = time.time()
        
        for _ in range(1000):
            gauss_seidel_cupy(f)
            
        # we must synchronize before stopping the timer
        cp.cuda.Stream.null.synchronize()
        end_time = time.time()
        
        elapsed = end_time - start_time
        execution_times.append(elapsed)
        print(f"{N:<10} | {elapsed:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(grid_sizes, execution_times, 'm-o', label='CuPy (GPU)')
    plt.title('CuPy GPU Performance')
    plt.xlabel('Grid Size (N)')
    plt.ylabel('Time for 1000 Iterations (s)')
    plt.grid(True)
    plt.legend()

    save_to_hdf5(f, "cupy_result.h5")

if __name__ == "__main__":
    run_benchmark()