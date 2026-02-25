import torch
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
    
    if hasattr(grid, 'cpu'):
        grid_cpu = grid.cpu().numpy()
        
    elif hasattr(grid, 'get'):
        grid_cpu = grid.get()
        
    else:
        grid_cpu = grid

    with h5py.File(filename, "w") as f:
        dset = f.create_dataset("newgrid", data=grid_cpu)
        dset.attrs["description"] = "Final simulation state"
        dset.attrs["grid_size"] = grid_cpu.shape[0]
        
    print("Save complete!")

def gauss_seidel_pytorch_roll(f):
    """
    Vectorized solver using torch.roll.
    Technically performs a Jacobi update since it uses old values for all neighbors.
    """
    
    term_up = torch.roll(f, shifts=-1, dims=0)
    term_down = torch.roll(f, shifts=1, dims=0)
    term_right = torch.roll(f, shifts=-1, dims=1)
    term_left = torch.roll(f, shifts=1, dims=1)
    
    f_new = 0.25 * (term_up + term_down + term_right + term_left)
    
    f_new[0, :] = 0
    f_new[-1, :] = 0
    f_new[:, 0] = 0
    f_new[:, -1] = 0
    
    return f_new

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    grid_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    execution_times = []

    print(f"{'N':<10} | {'Time (s)':<15}")
    print("-" * 30)

    for N in grid_sizes:
        # Initialize on GPU directly
        f = torch.rand((N, N), dtype=torch.float32, device=device)

        f[0, :] = 0; f[-1, :] = 0; f[:, 0] = 0; f[:, -1] = 0
        
        # warmup
        for _ in range(10):
            f = gauss_seidel_pytorch_roll(f)
        
        torch.cuda.synchronize() # wait for warmup
        
        start_time = time.time()
        
        for _ in range(1000):
            f = gauss_seidel_pytorch_roll(f)
            
        torch.cuda.synchronize() # wait for GPU
        end_time = time.time()
        
        elapsed = end_time - start_time
        execution_times.append(elapsed)
        print(f"{N:<10} | {elapsed:.4f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(grid_sizes, execution_times, 'g-o', label='PyTorch (GPU)')
    plt.title('PyTorch GPU Performance (torch.roll)')
    plt.xlabel('Grid Size (N)')
    plt.ylabel('Time for 1000 Iterations (s)')
    plt.grid(True)
    plt.legend()
    plt.show()

    save_to_hdf5(f, "pytorch_result.h5")

if __name__ == "__main__":
    run_benchmark()