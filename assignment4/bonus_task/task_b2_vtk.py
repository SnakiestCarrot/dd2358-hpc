import numpy as np
import pyvtk
import os

GRID_SIZE = 200
TIME_STEPS = 100
ALPHA = 0.1
BETA = 0.02

def laplacian(field):
    """Computes the discrete Laplacian of a 2D field."""
    return (
        np.roll(field, shift=1, axis=0) +
        np.roll(field, shift=-1, axis=0) +
        np.roll(field, shift=1, axis=1) +
        np.roll(field, shift=-1, axis=1) -
        4 * field
    )

def update_ocean(u, v, temperature, wind):
    """Updates ocean fields for one time step."""
    u_new = u + ALPHA * laplacian(u) + BETA * wind
    v_new = v + ALPHA * laplacian(v) + BETA * wind
    temp_new = temperature + 0.01 * laplacian(temperature) 
    return u_new, v_new, temp_new

def save_ocean_vtk(filename, temperature, u, v):
    """Saves temperature and velocity fields into a VTK file."""
    nx, ny = temperature.shape

    temp_flat = temperature.T.flatten()
    u_flat = u.T.flatten()
    v_flat = v.T.flatten()
    w_flat = np.zeros_like(u_flat)

    vtk_data = pyvtk.VtkData(
        pyvtk.StructuredPoints([nx, ny, 1]),
        pyvtk.PointData(
            pyvtk.Scalars(temp_flat, name="temperature"),
            pyvtk.Vectors(np.column_stack((u_flat, v_flat, w_flat)), name="velocity")
        )
    )
    vtk_data.tofile(filename)

def run_simulation_and_export():
    os.makedirs("vtk_output_ocean", exist_ok=True)
    
    np.random.seed(42)
    temperature = np.random.uniform(5, 30, size=(GRID_SIZE, GRID_SIZE))
    u = np.random.uniform(-1, 1, size=(GRID_SIZE, GRID_SIZE))
    v = np.random.uniform(-1, 1, size=(GRID_SIZE, GRID_SIZE))
    wind = np.random.uniform(-0.5, 0.5, size=(GRID_SIZE, GRID_SIZE))

    print("Starting Ocean Simulation & VTK Export...")

    save_ocean_vtk(f"vtk_output_ocean/ocean_{0:03d}.vtk", temperature, u, v)

    for t in range(1, TIME_STEPS + 1):
        u, v, temperature = update_ocean(u, v, temperature, wind)

        save_ocean_vtk(f"vtk_output_ocean/ocean_{t:03d}.vtk", temperature, u, v)
        
        if t % 10 == 0:
            print(f"Processed {t}/{TIME_STEPS} steps...")

    print("Done! Open the 'vtk_output_ocean' folder in ParaView.")

if __name__ == '__main__':
    run_simulation_and_export()