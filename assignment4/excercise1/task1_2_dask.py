import numpy as np
import matplotlib.pyplot as plt
import random
import time
import multiprocessing
import dask
import dask.array as da
from dask.distributed import Client

# Constants
GRID_SIZE = 800
FIRE_SPREAD_PROB = 0.3
BURN_TIME = 3
DAYS = 60
NUM_SIMULATIONS = 10

# State definitions
EMPTY = 0
TREE = 1
BURNING = 2
ASH = 3

def initialize_forest():
    """Creates a forest grid with all trees and ignites one random tree."""
    forest = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)
    burn_time = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    
    x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    forest[x, y] = BURNING
    burn_time[x, y] = 1
    
    return forest, burn_time

def get_neighbors(x, y):
    """Returns the neighboring coordinates of a cell in the grid."""
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            neighbors.append((nx, ny))
    return neighbors

@dask.delayed
def simulate_wildfire_delayed(sim_id):
    """Simulates wildfire spread. Decorated to run lazily via Dask."""
    # Ensure independent random states per worker
    np.random.seed()
    random.seed()
    
    forest, burn_time = initialize_forest()
    fire_spread = []
    
    for day in range(DAYS):
        new_forest = forest.copy()
        
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if forest[x, y] == BURNING:
                    burn_time[x, y] += 1
                    
                    if burn_time[x, y] >= BURN_TIME:
                        new_forest[x, y] = ASH
                    
                    for nx, ny in get_neighbors(x, y):
                        if forest[nx, ny] == TREE and random.random() < FIRE_SPREAD_PROB:
                            new_forest[nx, ny] = BURNING
                            burn_time[nx, ny] = 1
        
        forest = new_forest.copy()
        current_burning = np.sum(forest == BURNING)
        fire_spread.append(current_burning)
        
        if current_burning == 0:
            break
            
    while len(fire_spread) < DAYS:
        fire_spread.append(0)
        
    return np.array(fire_spread)

if __name__ == '__main__':
    num_cores = multiprocessing.cpu_count()
    client = Client(n_workers=num_cores, threads_per_worker=1)
    
    print(f"Dask Dashboard is running at: {client.dashboard_link}")
    print("-> Open this link in your browser NOW to monitor the task execution.\n")
    
    start_time = time.time()

    print(f"Building Dask task graph for {NUM_SIMULATIONS} simulations...")
    lazy_results = [simulate_wildfire_delayed(i) for i in range(NUM_SIMULATIONS)]
    dask_arrays = [da.from_delayed(res, shape=(DAYS,), dtype=int) for res in lazy_results]
    stacked_array = da.stack(dask_arrays, axis=0)
    average_spread_lazy = stacked_array.mean(axis=0)
    print("Executing computation graph across distributed workers...")
    average_spread, all_results = dask.compute(average_spread_lazy, stacked_array)
    
    end_time = time.time()
    print(f"Simulations completed in {end_time - start_time:.2f} seconds.")
    client.close()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    for run in all_results:
        plt.plot(range(DAYS), run, alpha=0.2, color='gray') 
        
    plt.plot(range(DAYS), average_spread, color='blue', linewidth=2.5, label="Average Burning Trees (Dask)")
    
    plt.xlabel("Days")
    plt.ylabel("Average Number of Burning Trees")
    plt.title("Wildfire Spread Over Time (Dask Parallelization)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
