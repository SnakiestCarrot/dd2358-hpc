import time
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

# Import modules
try:
    import mandelbrot_cython
except ImportError:
    print("Please compile Cython code first!")
    exit()

import mandelbrot_numpy
import mandelbrot_gpu

# --- Baseline ---
def mandelbrot_baseline(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def mandelbrot_set_baseline(width, height, x_min, x_max, y_min, y_max, max_iter):
    x_vals = np.linspace(x_min, x_max, width)
    y_vals = np.linspace(y_min, y_max, height)
    image = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            c = complex(x_vals[j], y_vals[i])
            image[i, j] = mandelbrot_baseline(c, max_iter)
    return image

def benchmark():
    # Parameters
    width, height = 1000, 800
    x_min, x_max, y_min, y_max = -2, 1, -1, 1
    max_iter = 100

    print(f"Benchmarking Mandelbrot ({width}x{height}, iter={max_iter})")
    print("-" * 60)
    print(f"{'Implementation':<20} | {'Device':<6} | {'Time (s)':<10} | {'Speedup'}")
    print("-" * 60)

    # 1. Baseline
    start = time.time()
    _ = mandelbrot_set_baseline(width, height, x_min, x_max, y_min, y_max, max_iter)
    t_base = time.time() - start
    print(f"{'Baseline (Loops)':<20} | {'CPU':<6} | {t_base:<10.4f} | {'1.0x'}")

    # 2. Vectorized NumPy
    start = time.time()
    _ = mandelbrot_numpy.mandelbrot_numpy(width, height, x_min, x_max, y_min, y_max, max_iter)
    t_numpy = time.time() - start
    print(f"{'Vectorized (NumPy)':<20} | {'CPU':<6} | {t_numpy:<10.4f} | {t_base/t_numpy:.1f}x")

    # 3. Cython
    start = time.time()
    _ = mandelbrot_cython.mandelbrot_set_cython(width, height, x_min, x_max, y_min, y_max, max_iter)
    t_cy = time.time() - start
    print(f"{'Cython (Compiled)':<20} | {'CPU':<6} | {t_cy:<10.4f} | {t_base/t_cy:.1f}x")

    # 4. Vectorized GPU (CuPy)
    # Warmup
    _ = mandelbrot_gpu.mandelbrot_set_gpu(100, 100, x_min, x_max, y_min, y_max, max_iter)
    cp.cuda.Stream.null.synchronize()
    
    start = time.time()
    _ = mandelbrot_gpu.mandelbrot_set_gpu(width, height, x_min, x_max, y_min, y_max, max_iter)
    cp.cuda.Stream.null.synchronize()
    t_gpu = time.time() - start
    print(f"{'Vectorized (CuPy)':<20} | {'GPU':<6} | {t_gpu:<10.4f} | {t_base/t_gpu:.1f}x")

if __name__ == "__main__":
    benchmark()