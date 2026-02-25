# mandelbrot_cython.pyx
import numpy as np
cimport numpy as cnp
import cython

# 1. Turn off safety checks for raw speed
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int mandelbrot_kernel(double complex c, int max_iter) nogil:
    """
    Optimized C-function. 
    'nogil' allows it to run without the Python lock if we used threading.
    """
    cdef double complex z = 0
    cdef int n
    
    # We use z.real*z.real + z.imag*z.imag > 4 instead of abs(z) > 2
    # to avoid the square root calculation (computational trick).
    for n in range(max_iter):
        if (z.real * z.real + z.imag * z.imag) > 4.0:
            return n
        z = z * z + c
    return max_iter

@cython.boundscheck(False)
@cython.wraparound(False)
def mandelbrot_set_cython(int width, int height, double x_min, double x_max, double y_min, double y_max, int max_iter):
    """
    Cython wrapper that fills the image using typed loops.
    """
    # Create the numpy array
    cdef cnp.ndarray[cnp.int32_t, ndim=2] image = np.zeros((height, width), dtype=np.int32)
    
    # Create typed memoryview for fast access
    cdef int[:, :] image_view = image

    cdef double dx = (x_max - x_min) / width
    cdef double dy = (y_max - y_min) / height
    cdef double real, imag
    cdef double complex c
    cdef int i, j

    # Typed loops (C speed)
    for i in range(height):
        imag = y_min + i * dy
        for j in range(width):
            real = x_min + j * dx
            c = complex(real, imag)
            image_view[i, j] = mandelbrot_kernel(c, max_iter)

    return image