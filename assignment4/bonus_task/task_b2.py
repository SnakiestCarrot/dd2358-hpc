import numpy as np
import dask
import dask.array as da
from dask.distributed import Client
import time
import multiprocessing

GRID_SIZE = 200  
TIME_STEPS = 500
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

def update_ocean_dask(u, v, temperature, wind):
    u_new = u + ALPHA * dask_laplacian(u) + BETA * wind
    v_new = v + ALPHA * dask_laplacian(v) + BETA * wind
    temp_new = temperature + 0.01 * dask_laplacian(temperature)
    return u_new, v_new, temp_new

def run_dask(chunk_dim):
    da.random.seed(42)
    chunk_size = (chunk_dim, chunk_dim)
    
    print(f"\nBuilding graph for chunk size: {chunk_size}...")
    temperature = da.random.uniform(5, 30, size=(GRID_SIZE, GRID_SIZE), chunks=chunk_size)
    u = da.random.uniform(-1, 1, size=(GRID_SIZE, GRID_SIZE), chunks=chunk_size)
    v = da.random.uniform(-1, 1, size=(GRID_SIZE, GRID_SIZE), chunks=chunk_size)
    wind = da.random.uniform(-0.5, 0.5, size=(GRID_SIZE, GRID_SIZE), chunks=chunk_size)

    start_time = time.time()
    for t in range(TIME_STEPS):
        u, v, temperature = update_ocean_dask(u, v, temperature, wind)
    
    print("Executing...")
    dask.compute(u, v, temperature)
    end_time = time.time()
    
    print(f"Finished in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    client = Client()
    print(f"\n---> Dask Dashboard is running at: {client.dashboard_link} <---")
    print("Open this link in your browser and go to the 'Task Stream' tab.")

    input("\nPress Enter to run 200x200 chunks (1 Total Chunk)...")
    run_dask(200)

    input("\nPress Enter to run 100x100 chunks (4 Total Chunks)...")
    run_dask(100)

    input("\nPress Enter to run 50x50 chunks (16 Total Chunks)...")
    run_dask(50)

    print("\nAll tests complete! Closing client.")
    client.close()