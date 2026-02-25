import numpy as np

def mandelbrot_numpy(width, height, x_min, x_max, y_min, y_max, max_iter):
    """
    Vectorized CPU implementation using NumPy and Boolean Masking.
    """
    # 1. Create the complex grid
    x = np.linspace(x_min, x_max, width, dtype=np.float32)
    y = np.linspace(y_min, y_max, height, dtype=np.float32)
    real, imag = np.meshgrid(x, y)
    c = real + 1j * imag

    # 2. Initialize Z and Output Image
    z = np.zeros_like(c)
    image = np.zeros((height, width), dtype=np.int32)
    mask = np.ones((height, width), dtype=bool)

    # 3. Vectorized Loop
    for n in range(max_iter):
        if not mask.any():
            break
            
        # Update only points that haven't diverged
        z[mask] = z[mask] * z[mask] + c[mask]
        
        # Check divergence
        diverged_now = np.abs(z) > 2
        
        # Find points that JUST diverged
        escaped = diverged_now & mask
        image[escaped] = n
        
        # Update mask
        mask[diverged_now] = False
        
    image[mask] = max_iter
    return image