import cupy as cp

def mandelbrot_set_gpu(width, height, x_min, x_max, y_min, y_max, max_iter):
    """
    Vectorized GPU implementation using Boolean Masking.
    """
    x = cp.linspace(x_min, x_max, width, dtype=cp.float32)
    y = cp.linspace(y_min, y_max, height, dtype=cp.float32)
    
    real, imag = cp.meshgrid(x, y)
    c = real + 1j * imag
    
    z = cp.zeros_like(c)
    image = cp.zeros((height, width), dtype=cp.int32)
    
    mask = cp.ones((height, width), dtype=bool)

    for n in range(max_iter):
        if not mask.any():
            break
            
        z[mask] = z[mask] * z[mask] + c[mask]
        diverged_now = (z.real**2 + z.imag**2) > 4.0
        escaped = diverged_now & mask
        image[escaped] = n
        mask[diverged_now] = False
        
    image[mask] = max_iter
    
    return image