"""
Game of Life Module
====================

A simple Python/matplotlib implementation of Conway's Game of Life.

This module provides functions to simulate Conway's Game of Life cellular automaton,
including support for random initial states and classic patterns like gliders
and the Gosper Glider Gun.

Example usage::

    python game_of_life.py --grid-size 100 --glider

Author: Casper Johansson and Filip Boive
"""

import sys, argparse, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
ON = 255
OFF = 0
vals = [ON, OFF]


def randomGrid(N):
    """
    Generate a grid of NxN random values.

    Creates a square grid where each cell is randomly set to ON (255) or OFF (0),
    with a 20% probability of being ON and 80% probability of being OFF.

    :param N: The size of the grid (creates an NxN grid).
    :type N: int
    :return: A 2D numpy array of shape (N, N) with random ON/OFF values.
    :rtype: numpy.ndarray
    """
    return np.random.choice(vals, N * N, p=[0.2, 0.8]).reshape(N, N)


def addGlider(i, j, grid):
    """
    Add a glider pattern to the grid.

    A glider is a pattern that moves diagonally across the grid over time.
    This function places the glider with its top-left cell at position (i, j).

    :param i: Row index for the top-left corner of the glider.
    :type i: int
    :param j: Column index for the top-left corner of the glider.
    :type j: int
    :param grid: The grid to add the glider to (modified in place).
    :type grid: numpy.ndarray
    """
    glider = np.array([[0, 0, 255], [255, 0, 255], [0, 255, 255]])
    grid[i : i + 3, j : j + 3] = glider


def addGosperGliderGun(i, j, grid):
    """
    Add a Gosper Glider Gun pattern to the grid.

    The Gosper Glider Gun is a pattern that periodically emits gliders.
    It was the first known finite pattern with unbounded growth, discovered
    by Bill Gosper in 1970. The gun occupies an 11x38 cell region.

    :param i: Row index for the top-left corner of the gun.
    :type i: int
    :param j: Column index for the top-left corner of the gun.
    :type j: int
    :param grid: The grid to add the gun to (modified in place).
    :type grid: numpy.ndarray
    """
    gun = np.zeros(11 * 38).reshape(11, 38)

    gun[5][1] = gun[5][2] = 255
    gun[6][1] = gun[6][2] = 255

    gun[3][13] = gun[3][14] = 255
    gun[4][12] = gun[4][16] = 255
    gun[5][11] = gun[5][17] = 255
    gun[6][11] = gun[6][15] = gun[6][17] = gun[6][18] = 255
    gun[7][11] = gun[7][17] = 255
    gun[8][12] = gun[8][16] = 255
    gun[9][13] = gun[9][14] = 255

    gun[1][25] = 255
    gun[2][23] = gun[2][25] = 255
    gun[3][21] = gun[3][22] = 255
    gun[4][21] = gun[4][22] = 255
    gun[5][21] = gun[5][22] = 255
    gun[6][23] = gun[6][25] = 255
    gun[7][25] = 255

    gun[3][35] = gun[3][36] = 255
    gun[4][35] = gun[4][36] = 255

    grid[i : i + 11, j : j + 38] = gun

def update_grid(grid, N):
    """
    Compute the next generation of the Game of Life grid.

    Applies Conway's Game of Life rules to compute the next generation
    using vectorized NumPy operations:

    - A live cell with fewer than 2 live neighbors dies (underpopulation).
    - A live cell with 2 or 3 live neighbors survives.
    - A live cell with more than 3 live neighbors dies (overpopulation).
    - A dead cell with exactly 3 live neighbors becomes alive (reproduction).

    Uses toroidal boundary conditions via ``np.roll`` (edges wrap around).

    :param grid: The current grid state (modified in place).
    :type grid: numpy.ndarray
    :param N: The size of the grid (NxN).
    :type N: int
    """
    # compute 8-neighbor sum using np.roll for toroidal boundary conditions
    neighbors = (
        np.roll(grid, 1, axis=0) +   # up
        np.roll(grid, -1, axis=0) +   # down
        np.roll(grid, 1, axis=1) +    # left
        np.roll(grid, -1, axis=1) +   # right
        np.roll(np.roll(grid, 1, axis=0), 1, axis=1) +    # up-left
        np.roll(np.roll(grid, 1, axis=0), -1, axis=1) +   # up-right
        np.roll(np.roll(grid, -1, axis=0), 1, axis=1) +   # down-left
        np.roll(np.roll(grid, -1, axis=0), -1, axis=1)    # down-right
    ) // 255

    # apply Conway's rules vectorized
    alive = grid == ON
    # survive: alive and 2 or 3 neighbors
    survive = alive & ((neighbors == 2) | (neighbors == 3))
    # birth: dead and exactly 3 neighbors
    birth = ~alive & (neighbors == 3)

    grid[:] = OFF
    grid[survive | birth] = ON


def update(frameNum, img, grid, N):
    """
    Animation callback to update the grid and image for one generation.

    This function wraps :func:`update_grid` for use with matplotlib's
    FuncAnimation.

    :param frameNum: The current frame number (used by matplotlib animation).
    :type frameNum: int
    :param img: The matplotlib image object to update.
    :type img: matplotlib.image.AxesImage
    :param grid: The current grid state (modified in place).
    :type grid: numpy.ndarray
    :param N: The size of the grid (NxN).
    :type N: int
    :return: A tuple containing the updated image object.
    :rtype: tuple
    """
    update_grid(grid, N)
    img.set_data(grid)
    return (img,)


def main():
    """
    Main entry point for Conway's Game of Life simulation.

    Parses command line arguments and initializes the simulation with the
    specified parameters. Supports the following command line options:

    - ``--grid-size``: Size of the grid (default: 100)
    - ``--mov-file``: Output movie file path
    - ``--interval``: Animation update interval in milliseconds (default: 50)
    - ``--glider``: Start with a single glider pattern
    - ``--gosper``: Start with a Gosper Glider Gun pattern
    - ``--no-anim``: Disable animation and run silently for a fixed number of iterations
    - ``--iterations``: Number of iterations to run (default: 100)

    If neither --glider nor --gosper is specified, the grid is initialized
    with random values.
    """
    # Command line args are in sys.argv[1], sys.argv[2] ..
    # sys.argv[0] is the script name itself and can be ignored
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Runs Conway's Game of Life simulation."
    )
    # add arguments
    parser.add_argument("--grid-size", dest="N", required=False)
    parser.add_argument("--mov-file", dest="movfile", required=False)
    parser.add_argument("--interval", dest="interval", required=False)
    parser.add_argument("--glider", action="store_true", required=False)
    parser.add_argument("--gosper", action="store_true", required=False)
    parser.add_argument("--no-anim", dest="no_anim", action="store_true", required=False)
    parser.add_argument("--iterations", dest="iterations", type=int, default=100, required=False)
    args = parser.parse_args()

    # set grid size
    N = 100
    if args.N and int(args.N) > 8:
        N = int(args.N)

    # set animation update interval
    updateInterval = 50
    if args.interval:
        updateInterval = int(args.interval)

    # declare grid
    grid = np.array([])
    # check if "glider" demo flag is specified
    if args.glider:
        grid = np.zeros(N * N).reshape(N, N)
        addGlider(1, 1, grid)
    elif args.gosper:
        grid = np.zeros(N * N).reshape(N, N)
        addGosperGliderGun(10, 10, grid)
    else:
        # populate grid with random on/off - more off than on
        grid = randomGrid(N)

    # run without animation if --no-anim is specified
    if args.no_anim:
        start_time = time.time()
        for _ in range(args.iterations):
            update_grid(grid, N)
        elapsed_time = time.time() - start_time
        print(f"Grid size: {N}x{N}")
        print(f"Iterations: {args.iterations}")
        print(f"Total time: {elapsed_time:.4f} seconds")
        print(f"Time per iteration: {elapsed_time / args.iterations * 1000:.4f} ms")
        return

    # set up animation
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation="nearest")
    ani = animation.FuncAnimation(
        fig,
        update,
        fargs=(
            img,
            grid,
            N,
        ),
        frames=args.iterations,
        interval=updateInterval,
        save_count=50,
    )

    # set output file
    if args.movfile:
        ani.save(args.movfile, fps=30, extra_args=["-vcodec", "libx264"])

    plt.show()


# call main
if __name__ == "__main__":
    main()
