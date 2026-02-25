import matplotlib.pyplot as plt
import numpy as np

# --- Data from previous tasks ---
# Task 1.4: Original Python (Stops at N=100 because it gets too slow)
n_py = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
t_py = [0.0496, 0.2431, 0.5811, 1.0643, 1.6960, 2.4731, 3.4077, 4.4924, 5.7092, 7.0802]

# Task 1.4: Optimized Cython (Stops at N=100 in your data, but is fast)
n_cy = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
t_cy = [0.0006, 0.0014, 0.0032, 0.0059, 0.0103, 0.0163, 0.0236, 0.0323, 0.0424, 0.0538]

# Task 1.5: PyTorch GPU (Goes up to N=10,000)
n_torch = [10, 100, 200, 500, 1000, 2000, 5000, 10000]
t_torch = [0.0903, 0.0916, 0.0927, 0.0944, 0.1298, 0.4840, 2.8922, 11.5040]

# Task 1.6: CuPy GPU (Goes up to N=10,000)
n_cupy = [10, 100, 200, 500, 1000, 2000, 5000, 10000]
t_cupy = [0.1172, 0.1140, 0.1139, 0.1150, 0.1155, 0.3022, 1.7760, 7.0486]

# --- Plotting ---
plt.figure(figsize=(10, 6))

# Plot CPU lines (Solid)
plt.plot(n_py, t_py, 'r-o', label='Python (CPU)')
plt.plot(n_cy, t_cy, 'b-s', label='Cython (CPU)')

# Plot GPU lines (Dashed)
plt.plot(n_torch, t_torch, 'g--^', label='PyTorch (GPU)')
plt.plot(n_cupy, t_cupy, 'm--d', label='CuPy (GPU)')

plt.title('Performance Comparison: CPU vs GPU (Log-Log Scale)')
plt.xlabel('Grid Size (N)')
plt.ylabel('Time (seconds)')
plt.yscale('log') # Log scale is crucial to see the differences!
plt.xscale('log') # Log scale for N helps visualize the huge range
plt.grid(True, which="both", ls="-")
plt.legend()

plt.savefig('performance_comparison.png')
print("Plot saved as performance_comparison.png")