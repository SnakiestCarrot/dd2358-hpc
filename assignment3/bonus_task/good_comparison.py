import time
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

# --- Imports (Ensure these files exist from previous steps) ---
try:
    import mandelbrot_cython
    import mandelbrot_gpu
except ImportError:
    print("Error: Ensure mandelbrot_cython.so and mandelbrot_gpu.py exist.")
    exit()

# --- 1. Baseline Implementation (Nested Loops) ---
def mandelbrot_baseline_pixel(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def run_baseline(width, height, max_iter):
    x = np.linspace(-2, 1, width)
    y = np.linspace(-1, 1, height)
    img = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            img[i, j] = mandelbrot_baseline_pixel(complex(x[j], y[i]), max_iter)
    return img

# --- 2. Vectorized CPU Implementation (NumPy) ---
def run_numpy(width, height, max_iter):
    x = np.linspace(-2, 1, width, dtype=np.float32)
    y = np.linspace(-1, 1, height, dtype=np.float32)
    real, imag = np.meshgrid(x, y)
    c = real + 1j * imag
    z = np.zeros_like(c)
    image = np.zeros((height, width), dtype=np.int32)
    mask = np.ones((height, width), dtype=bool)

    for n in range(max_iter):
        if not mask.any(): break
        z[mask] = z[mask] * z[mask] + c[mask]
        diverged = np.abs(z) > 2
        escaped_now = diverged & mask
        image[escaped_now] = n
        mask[diverged] = False
    image[mask] = max_iter
    return image

# --- Benchmark Driver ---
def main():
    # Test Resolutions (N x N)
    resolutions = [100, 200, 500, 1000, 2000, 4000]
    max_iter = 200
    
    # Store results
    results = {
        "Baseline": [],
        "NumPy": [],
        "Cython": [],
        "CuPy": []
    }
    
    # Track pixels for x-axis
    pixels = [r*r for r in resolutions]

    print(f"{'Size (NxN)':<12} | {'Baseline':<10} | {'NumPy':<10} | {'Cython':<10} | {'CuPy (GPU)':<10}")
    print("-" * 65)

    for N in resolutions:
        width, height = N, N

        if N <= 4000:
            start = time.time()
            run_baseline(width, height, max_iter)
            t_base = time.time() - start
            results["Baseline"].append(t_base)
        else:
            t_base = None # Too slow
            results["Baseline"].append(None)

        # 2. NumPy Vectorized
        start = time.time()
        run_numpy(width, height, max_iter)
        t_numpy = time.time() - start
        results["NumPy"].append(t_numpy)

        # 3. Cython
        start = time.time()
        mandelbrot_cython.mandelbrot_set_cython(width, height, -2, 1, -1, 1, max_iter)
        t_cy = time.time() - start
        results["Cython"].append(t_cy)

        # 4. CuPy (GPU)
        # Warmup for small sizes
        if N == 100:
            mandelbrot_gpu.mandelbrot_set_gpu(100, 100, -2, 1, -1, 1, max_iter)
            cp.cuda.Stream.null.synchronize()

        start = time.time()
        mandelbrot_gpu.mandelbrot_set_gpu(width, height, -2, 1, -1, 1, max_iter)
        cp.cuda.Stream.null.synchronize()
        t_gpu = time.time() - start
        results["CuPy"].append(t_gpu)

        # Print row
        s_base = f"{t_base:.4f}" if t_base else "-"
        print(f"{N}x{N:<8} | {s_base:<10} | {t_numpy:<10.4f} | {t_cy:<10.4f} | {t_gpu:<10.4f}")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    # Filter out Nones for plotting
    valid_base = [(p, t) for p, t in zip(pixels, results["Baseline"]) if t is not None]
    if valid_base:
        plt.plot(*zip(*valid_base), 'r-o', label='Baseline (Loops)')

    plt.plot(pixels, results["NumPy"], 'y-s', label='Vectorized (NumPy)')
    plt.plot(pixels, results["Cython"], 'b-^', label='Cython (Compiled)')
    plt.plot(pixels, results["CuPy"], 'g-d', label='Vectorized (CuPy GPU)')

    plt.xlabel('Total Pixels (N*N)')
    plt.ylabel('Time (seconds)')
    plt.title('Mandelbrot Generation Performance')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.yscale('log')
    plt.xscale('log')
    
    plt.savefig('mandelbrot_benchmark_comprehensive.png')
    print("\nPlot saved as mandelbrot_benchmark_comprehensive.png")

if __name__ == "__main__":
    main()