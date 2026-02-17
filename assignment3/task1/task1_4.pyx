# solver_cython.pyx
import cython
import numpy as np
cimport numpy as cnp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) 
def gauss_seidel_cython(double[:, :] f):
    """
    Optimized Gauss-Seidel solver using Cython.
    """
    cdef int i, j, n, m
    
    n = f.shape[0]
    m = f.shape[1]

    for i in range(1, n-1):
        for j in range(1, m-1):
            f[i, j] = 0.25 * (f[i, j+1] + f[i, j-1] + f[i+1, j] + f[i-1, j])
            
    return f