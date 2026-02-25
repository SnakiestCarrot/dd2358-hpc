import numpy as np
import dask
import dask.array as da
from dask.distributed import Client
import time
import matplotlib.pyplot as plt

# --- Configuration ---
GRID_SIZE = 2000  # Fixed grid size to isolate chunk performance
TIME_STEPS = 50
ALPHA = 0.1
BETA = 0.02

def laplacian(field):
    return (
        np.roll(field, shift=1, axis=0) +
        np.roll(field, shift=-1, axis=0) +
        np.roll(field, shift=1, axis=1) +
        np.roll(field, shift=-1, axis=1) -
        4 * field
    )

def dask_laplacian(field):
    return field.map_overlap(
        laplacian,
        depth=1,
        boundary='reflect',
        dtype=field.dtype
    )

def update_ocean_serial(u, v, temperature, wind):
    u_new = u + ALPHA * laplacian(u) + BETA * wind
    v_new = v + ALPHA * laplacian(v) + BETA * wind
    temp_new = temperature + 0.01 * laplacian(temperature) 
    return u_new, v_new, temp_new

def update_ocean_dask(u, v, temperature, wind):
    u_new = u + ALPHA * dask_laplacian(u) + BETA * wind
    v_new = v + ALPHA * dask_laplacian(v) + BETA * wind
    temp_new = temperature + 0.01 * dask_laplacian(temperature)
    return u_new, v_new, temp_new

def run_serial():
    np.random.seed(42)
    temperature = np.random.uniform(5, 30, size=(GRID_SIZE, GRID_SIZE))
    u = np.random.uniform(-1, 1, size=(GRID_SIZE, GRID_SIZE))
    v = np.random.uniform(-1, 1, size=(GRID_SIZE, GRID_SIZE))
    wind = np.random.uniform(-0.5, 0.5, size=(GRID_SIZE, GRID_SIZE))

    start_time = time.time()
    for t in range(TIME_STEPS):
        u, v, temperature = update_ocean_serial(u, v, temperature, wind)
    end_time = time.time()
    
    return end_time - start_time

def run_dask(chunk_dim):
    da.random.seed(42)
    chunk_size = (chunk_dim, chunk_dim)
    temperature = da.random.uniform(5, 30, size=(GRID_SIZE, GRID_SIZE), chunks=chunk_size)
    u = da.random.uniform(-1, 1, size=(GRID_SIZE, GRID_SIZE), chunks=chunk_size)
    v = da.random.uniform(-1, 1, size=(GRID_SIZE, GRID_SIZE), chunks=chunk_size)
    wind = da.random.uniform(-0.5, 0.5, size=(GRID_SIZE, GRID_SIZE), chunks=chunk_size)

    start_time = time.time()
    for t in range(TIME_STEPS):
        u, v, temperature = update_ocean_dask(u, v, temperature, wind)

    dask.compute(u, v, temperature)
    end_time = time.time()
    
    return end_time - start_time

if __name__ == '__main__':
    client = Client(processes=False)
    print(f"Dask Dashboard: {client.dashboard_link}\n")
    print(f"--- Benchmarking Fixed Grid Size ({GRID_SIZE}x{GRID_SIZE}) ---")
    print("Running Serial baseline...")
    serial_time = run_serial()
    chunk_dims = [2000, 1000, 500, 250, 125]
    dask_times = []
    print(f"\n{'Chunk Dimensions':<20} | {'Total Chunks':<15} | {'Execution Time (s)':<20}")
    print("-" * 60)
    print(f"{'Serial (No Chunks)':<20} | {'1':<15} | {serial_time:<20.2f}")
    
    for dim in chunk_dims:
        num_chunks = (GRID_SIZE // dim) ** 2
        t_dask = run_dask(dim)
        dask_times.append(t_dask)
        print(f"{str((dim, dim)):<20} | {num_chunks:<15} | {t_dask:<20.2f}")

    client.close()
    num_chunks_list = [(GRID_SIZE // d)**2 for d in chunk_dims]
    plt.figure(figsize=(10, 6))
    plt.plot(num_chunks_list, dask_times, marker='o', label='Dask (Varying Chunks)', color='blue', linewidth=2.5)
    plt.axhline(y=serial_time, color='red', linestyle='--', label='Serial Baseline (NumPy)', linewidth=2.5)
    plt.xlabel('Total Number of Chunks')
    plt.ylabel(f'Execution Time for {TIME_STEPS} steps (seconds)')
    plt.title(f'Effect of Chunk Size on Performance ({GRID_SIZE}x{GRID_SIZE} Grid)')
    plt.xscale('log', base=2)
    plt.xticks(num_chunks_list, num_chunks_list)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()