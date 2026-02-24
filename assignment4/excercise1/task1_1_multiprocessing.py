import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing
import time

# Constants
GRID_SIZE = 800         # 800x800 forest grid
FIRE_SPREAD_PROB = 0.3  # Probability that fire spreads to a neighboring tree
BURN_TIME = 3           # Time before a tree turns into ash
DAYS = 60               # Maximum simulation time
NUM_SIMULATIONS = 10   # Number of independent Monte Carlo simulations to run

# State definitions
EMPTY = 0    # No tree
TREE = 1     # Healthy tree 
BURNING = 2  # Burning tree 
ASH = 3      # Burned tree 

def initialize_forest():
    """Creates a forest grid with all trees and ignites one random tree."""
    forest = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)  # All trees
    burn_time = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)  # Tracks how long a tree burns
    
    # Ignite a random tree
    x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    forest[x, y] = BURNING
    burn_time[x, y] = 1  # Fire starts burning
    
    return forest, burn_time

def get_neighbors(x, y):
    """Returns the neighboring coordinates of a cell in the grid."""
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            neighbors.append((nx, ny))
    return neighbors

def run_single_simulation(sim_id):
    """Simulates wildfire spread over time (pure computation, no plotting)."""
    # Re-seed random generators to ensure independent Monte Carlo runs
    np.random.seed()
    random.seed()
    
    forest, burn_time = initialize_forest()
    fire_spread = []  # Track number of burning trees each day
    
    for day in range(DAYS):
        new_forest = forest.copy()
        
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if forest[x, y] == BURNING:
                    burn_time[x, y] += 1  # Increase burn time
                    
                    # If burn time exceeds threshold, turn to ash
                    if burn_time[x, y] >= BURN_TIME:
                        new_forest[x, y] = ASH
                    
                    # Spread fire to neighbors
                    for nx, ny in get_neighbors(x, y):
                        if forest[nx, ny] == TREE and random.random() < FIRE_SPREAD_PROB:
                            new_forest[nx, ny] = BURNING
                            burn_time[nx, ny] = 1
        
        forest = new_forest.copy()
        current_burning = np.sum(forest == BURNING)
        fire_spread.append(current_burning)
        
        if current_burning == 0:  # Stop if no more fire
            break
            
    # Pad the results with zeros up to DAYS to make averaging easy later
    while len(fire_spread) < DAYS:
        fire_spread.append(0)
        
    return fire_spread


if __name__ == '__main__':
    print(f"Starting {NUM_SIMULATIONS} simulations using multiprocessing...")
    start_time = time.time()
    
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores.")
    
    # Create a multiprocessing pool
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Map the simulation function to run NUM_SIMULATIONS times
        # We pass range(NUM_SIMULATIONS) just to give each run a unique sim_id
        results = pool.map(run_single_simulation, range(NUM_SIMULATIONS))
        
    end_time = time.time()
    print(f"Simulations completed in {end_time - start_time:.2f} seconds.")
    
    # Aggregate results
    # results is a list of lists, we convert it to a NumPy array for easy column-wise math
    results_array = np.array(results)
    average_spread = np.mean(results_array, axis=0)
    
    # Plot the aggregated results
    plt.figure(figsize=(10, 6))

    for run in results:
        plt.plot(range(DAYS), run, alpha=0.2, color='gray') 
        
    # Plot the average across all runs
    plt.plot(range(DAYS), average_spread, color='red', linewidth=2.5, label="Average Burning Trees")
    
    plt.xlabel("Days")
    plt.ylabel("Number of Burning Trees")
    plt.title(f"Wildfire Spread Over Time ({NUM_SIMULATIONS} Monte Carlo Simulations)")
    plt.legend()
    plt.show()