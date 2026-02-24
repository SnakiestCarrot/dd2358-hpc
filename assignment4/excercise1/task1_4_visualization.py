import numpy as np
import random
import pyvtk
import os

# Constants
GRID_SIZE = 800  
FIRE_SPREAD_PROB = 0.3  
BURN_TIME = 3  
DAYS = 60  

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

def save_forest_vtk(filename, forest):
    """
    Saves the 2D forest grid as a VTK file for ParaView.
    """
    nx, ny = forest.shape
    
    # Flatten the array for VTK. Transposing aligns the visual orientation 
    # to match what we normally see in matplotlib.
    forest_flat = forest.T.flatten()

    # Create VTK structure with a single scalar field: 'tree_state'
    vtk_data = pyvtk.VtkData(
        pyvtk.StructuredPoints([nx, ny, 1]),  # 2D structured grid
        pyvtk.PointData(
            pyvtk.Scalars(forest_flat, name="tree_state")  
        )
    )
    vtk_data.tofile(filename)
    print(f"Saved: {filename}")

def simulate_wildfire_and_save_vtk():
    """Simulates wildfire spread and exports a VTK file for every day."""
    # Create a directory to hold the frames to keep your folder clean
    os.makedirs("vtk_output", exist_ok=True)
    
    forest, burn_time = initialize_forest()
    
    # Save the initial state (Day 0)
    save_forest_vtk(f"vtk_output/wildfire_000.vtk", forest)
    
    for day in range(1, DAYS + 1):
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
        
        # Save the frame for this day
        save_forest_vtk(f"vtk_output/wildfire_{day:03d}.vtk", forest)
        
        if np.sum(forest == BURNING) == 0:
            print(f"Fire burned out on day {day}. Stopping early.")
            break

if __name__ == '__main__':
    print("Starting simulation and generating VTK files...")
    simulate_wildfire_and_save_vtk()
    print("Done! You can now open the vtk_output folder in ParaView.")