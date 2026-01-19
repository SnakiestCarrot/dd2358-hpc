import time
import timeit
import numpy as np

nanoseconds_per_second = 1_000_000_000

def main():
    loops = 1_000_000
    
    print("granularity:")

    time = check_time_dot_time(loops)
    timeit = check_timeit(loops)
    time_ns = check_time_ns(loops)

    print(f"time.time():\t\t{time}")
    print(f"timeit.default_timer():\t{timeit}")
    print(f"time.time_ns():\t\t{time_ns}")

def check_time_dot_time(loops = 200):
    M = loops
    timesfound = np.empty((M,))
    for i in range(M):
        t1 =  time.time() # get timestamp from timer
        t2 = time.time() # get timestamp from timer
        while (t2 - t1) < 1e-16: # if zero then we are below clock granularity, retake timing
            t2 = time.time() # get timestamp from timer
        t1 = t2 # this is outside the loop
        timesfound[i] = t1 # record the time stamp
    minDelta = 1000000
    Delta = np.diff(timesfound) # it should be cast to int only when needed
    minDelta = Delta.min()
    return minDelta

def check_timeit(loops = 200):
    M = loops
    timesfound = np.empty((M,))
    for i in range(M):
        t1 =  timeit.default_timer() # get timestamp from timer
        t2 = timeit.default_timer() # get timestamp from timer
        while (t2 - t1) < 1e-16: # if zero then we are below clock granularity, retake timing
            t2 = timeit.default_timer() # get timestamp from timer
        t1 = t2 # this is outside the loop
        timesfound[i] = t1 # record the time stamp
    minDelta = 1000000
    Delta = np.diff(timesfound) # it should be cast to int only when needed
    minDelta = Delta.min()
    return minDelta

def check_time_ns(loops = 200):
    M = loops
    timesfound = np.empty((M,))
    for i in range(M):
        t1 =  time.time_ns() / nanoseconds_per_second # get timestamp from timer
        t2 = time.time_ns() / nanoseconds_per_second # get timestamp from timer
        while (t2 - t1) < 1e-16: # if zero then we are below clock granularity, retake timing
            t2 = time.time_ns() / nanoseconds_per_second # get timestamp from timer
        t1 = t2 # this is outside the loop
        timesfound[i] = t1 # record the time stamp
    minDelta = 1000000
    Delta = np.diff(timesfound) # it should be cast to int only when needed
    minDelta = Delta.min()
    return minDelta

if __name__ == "__main__":
    main()