import numpy as np
import array
import pytest

# TASK 2.1
N = 100
rng = np.random.default_rng()

def dgemm_numpy_loop(A, B, C, size):
    for i in range(size):
        for j in range(size):
            for k in range(size):
                C[i, j] += A[i, k] * B[k, j]

def dgemm_list(A, B, C, size):
    for i in range(size):
        for j in range(size):
            for k in range(size):
                C[i][j] += A[i][k] * B[k][j]
    
def dgemm_array(A, B, C, size):
    for i in range(size):
        for j in range(size):
            for k in range(size):
                C[i * N + j] += A[i * N + k] * B[k * N + j]

# TASK 2.2
def test_dgemm_array(benchmark):
    A = array.array('d', [rng.random() for _ in range(N*N)])
    B = array.array('d', [rng.random() for _ in range(N*N)])
    C = array.array('d', [rng.random() for _ in range(N*N)])

    benchmark(dgemm_array, A, B, C, N)

def test_dgemm_numpy_loop(benchmark):
    A = rng.random((N, N))
    B = rng.random((N, N))
    C = rng.random((N, N))
    
    benchmark(dgemm_numpy_loop, A, B, C, N)

def test_dgemm_list(benchmark):
    A = [[rng.random() for _ in range(N)] for _ in range(N)]
    B = [[rng.random() for _ in range(N)] for _ in range(N)]
    C = [[rng.random() for _ in range(N)] for _ in range(N)]

    benchmark(dgemm_list, A, B, C, N)


# NOTE: RUN "pytest test_task2.py" for 2.2


# TASK 2.3 (N=100)
"""
RESULTS:
1. Python Lists (Fastest Loop):
   - Mean: 57.33 ms ± 0.89 ms
   
2. Python Array (~3x slower than Lists):
   - Mean: 178.03 ms ± 1.85 ms

3. NumPy Loops (~6x slower than Lists):
   - Mean: 336.33 ms ± 7.77 ms
"""

# TASK 2.3 (N=100)
"""
1. THEORETICAL CALCULATIONS
   - Total Operations (2 * N^3): 2,000,000
   - Theoretical Peak (assuming 1 op/cycle): 4.3 GFLOPS (AMD Ryzen 5 5625U with 4.3 GHz Boost)

2. MEASURED PERFORMANCE
   - Python Lists:
     Mean Time: 57.33 ms
     Performance: ~34.89 MFLOPS
     Efficiency: ~0.81% of Theoretical Peak

   - Python Arrays:
     Mean Time: 178.03 ms
     Performance: ~11.23 MFLOPS
     Efficiency: ~0.26% of Theoretical Peak

   - NumPy Loops:
     Mean Time: 336.33 ms
     Performance: ~5.95 MFLOPS
     Efficiency: ~0.14% of Theoretical Peak
"""

# TASK 2.5 - BLAS OPTIMIZATION
def dgemm_blas(A, B, C):
    C += A @ B 

def test_dgemm_numpy_blas(benchmark):
    A = rng.random((N, N))
    B = rng.random((N, N))
    C = rng.random((N, N))
    
    benchmark(dgemm_blas, A, B, C)
    
"""
1. RAW PERFORMANCE (BLAS)
   - Mean Time: 35.93 us (microseconds)
   - Operations: 2,000,000
   - Performance: ~55.7 GFLOPS

2. COMPARISON VS PYTHON LOOPS
   - Speedup vs Lists: ~1,595x faster (57,331 us / 35.93 us)
   - Speedup vs NumPy Loops: ~9,360x faster (336,325 us / 35.93 us)

3. COMPARISON VS SIMPLIFIED PEAK
   - Simplified Peak (1 op/cycle): 4.3 GFLOPS
   - Measured BLAS: 55.7 GFLOPS
   - The actual performance is ~13x faster than the simplified peak.
"""