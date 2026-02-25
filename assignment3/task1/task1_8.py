import h5py
import numpy as np

def save_grid_to_hdf5(grid, filename="simulation_result.h5"):
    """
    Saves the numpy/cupy/torch grid to an HDF5 file.
    """
    if hasattr(grid, 'get'):  # If it's a CuPy array
        grid = grid.get()
    elif hasattr(grid, 'cpu'): # If it's a PyTorch tensor
        grid = grid.cpu().numpy()
        
    with h5py.File(filename, "w") as f:
        dset = f.create_dataset("newgrid", data=grid)
        
        dset.attrs["description"] = "Final state of Gauss-Seidel solver"
        dset.attrs["grid_size"] = grid.shape[0]
        
    print(f"Successfully saved grid to {filename}")
