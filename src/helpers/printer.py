import numpy as np
import numpy.typing as npt


BLOCK_SIZE = 3
GRID_SIZE = 9

test_sudoku = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
])

def print_sudoku(grid: npt.NDArray[np.int_]) -> None:
    """
    Prints a Sudoku grid in a human-readable format.
    Parameters:
    - grid: np.ndarray, the Sudoku grid to be printed.

    Returns:
    - None
    """
    print('\n')
    for row_idx in range(GRID_SIZE):
        if row_idx % BLOCK_SIZE == 0 and row_idx != 0:
            print("— " * (GRID_SIZE + BLOCK_SIZE - 1)) # separator between blocks
        row = ""
        for col_idx in range(GRID_SIZE):
            if col_idx % BLOCK_SIZE == 0 and col_idx != 0:
                row += "| " # separator between blocks
            row += f"{grid[row_idx, col_idx]} "
        print(row.strip())
    print('\n')
    
if __name__ == "__main__":
    print_sudoku(test_sudoku)  