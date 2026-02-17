import numpy as np

def gauss_seidel(f):
    for i in range(1, f.shape[0]-1):
        for j in range(1, f.shape[1]-1):
            f[i, j] = 0.25 * (f[i, j+1] + f[i, j-1] + f[i+1, j] + f[i-1, j])
    return f