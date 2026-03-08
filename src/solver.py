import numpy as np
import numpy.typing as npt
from helpers.printer import print_sudoku


GRID_SIZE = 9
BLOCK_SIZE = 3
VALID_VALUES = set(range(1, GRID_SIZE + 1))
valid_pool = np.array(sorted(VALID_VALUES))


def parse_sudoku() -> npt.NDArray[np.int_]:
    """Read a Sudoku grid from stdin, expecting comma separated rows."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for row_idx in range(GRID_SIZE):
        raw_row = input(f"please enter row {row_idx + 1}: ")
        grid[row_idx] = list(map(int, raw_row.split(',')))
    return grid


def contains_only_valid_numbers(grid: npt.NDArray[np.int_]) -> bool:
    return bool(np.all(np.isin(grid, valid_pool)))


def rows_are_unique(grid: npt.NDArray[np.int_]) -> bool:
    return all(len(np.unique(grid[row_idx, :])) == GRID_SIZE for row_idx in range(GRID_SIZE))


def columns_are_unique(grid: npt.NDArray[np.int_]) -> bool:
    return all(len(np.unique(grid[:, col_idx])) == GRID_SIZE for col_idx in range(GRID_SIZE))


def blocks_are_unique(grid: npt.NDArray[np.int_]) -> bool:
    for start_row in range(0, GRID_SIZE, BLOCK_SIZE):
        for start_col in range(0, GRID_SIZE, BLOCK_SIZE):
            block = grid[start_row:start_row + BLOCK_SIZE, 
                         start_col:start_col + BLOCK_SIZE]
            if len(np.unique(block)) != GRID_SIZE:
                return False
    return True


def is_valid_sudoku(grid: npt.NDArray[np.int_]) -> bool:
    """
    Checks if a given Sudoku grid is valid by ensuring that it contains only valid numbers and that each row, column, and 3x3 block contains unique values. 

    Parameters:
    - grid: np.ndarray, the Sudoku grid to be checked.

    Returns:
    - bool, True if the grid is a valid Sudoku grid, False otherwise.
    """
    
    if not contains_only_valid_numbers(grid):
        return False
    return rows_are_unique(grid) and columns_are_unique(grid) and blocks_are_unique(grid)


EXAMPLE_GRID = np.array([[4, 1, 5, 2, 7, 9, 3, 8, 6],
                         [7, 3, 9, 6, 1, 8, 4, 5, 2],
                         [2, 6, 8, 4, 5, 3, 9, 7, 1],
                         [3, 2, 6, 5, 8, 7, 1, 4, 9],
                         [1, 5, 7, 9, 6, 4, 2, 3, 8],
                         [8, 9, 4, 1, 3, 2, 5, 6, 7],
                         [9, 8, 2, 7, 4, 5, 6, 1, 3],
                         [5, 7, 1, 3, 2, 6, 8, 9, 4],
                         [6, 4, 3, 8, 9, 1, 7, 2, 5]])

EXAMPLE_GRID_HOLES = np.array([
    [9, 0, 0, 5, 0, 8, 0, 0, 7],
    [0, 8, 0, 3, 0, 2, 9, 0, 5],
    [0, 5, 4, 0, 0, 0, 0, 8, 0],
    [0, 7, 0, 6, 8, 0, 0, 3, 2],
    [1, 0, 0, 0, 0, 4, 0, 0, 8],
    [5, 0, 0, 2, 1, 9, 0, 6, 0],
    [0, 0, 0, 9, 0, 6, 0, 0, 1],
    [7, 2, 6, 0, 0, 1, 0, 4, 0],
    [0, 0, 1, 4, 7, 0, 0, 5, 6]])


def identify_candidates(grid: npt.NDArray[np.int_], row: int, column: int) -> set[int]:
    """finds candidates in a sudoku cell"""
    
    #check row and column
    used = set(grid[row]) | {grid[i, column] for i in range(9)}
    
    #check block
    block_row_start = (row // BLOCK_SIZE) * BLOCK_SIZE
    block_col_start = (column // BLOCK_SIZE) * BLOCK_SIZE
    for r in range(block_row_start, block_row_start + BLOCK_SIZE):
        for c in range(block_col_start, block_col_start + BLOCK_SIZE):
            used.add(grid[r, c])
            
    return VALID_VALUES - used


def markup_sudoku(grid: npt.NDArray[np.int_]) -> npt.NDArray[np.object_]:
    """returns a 2d list of sets of candidates for each cell in the sudoku grid"""
    markup = np.copy(grid).astype(object)
    
    for row, col in np.ndindex(grid.shape): 
        if grid[row, col] == 0:
            candidates = identify_candidates(grid, row, col)
            markup[row, col] = candidates

    return markup


def fill_naked_single(grid: npt.NDArray[np.int_], markup: npt.NDArray[np.object_]) -> bool:
    change = False
    for row, column in np.ndindex(grid.shape):
        if type(markup[row, column]) == int:
            continue
        elif len(markup[row, column]) == 1:
            grid[row, column] = next(iter(markup[row, column]))
            change = True
            
    return change
    

def fill_hidden_single(grid: npt.NDArray[np.int_], markup: npt.NDArray[np.object_]) -> bool:
    change = False
    
    return change


if __name__ == "__main__":
    # sudoku_grid = parse_sudoku()
    print(f"sudoku_check: {is_valid_sudoku(EXAMPLE_GRID)}")
    print(f'cell: {EXAMPLE_GRID_HOLES[0, 1]}')
    print(f'candidates: {identify_candidates(EXAMPLE_GRID_HOLES, 0, 1)}')
    print_sudoku(EXAMPLE_GRID_HOLES)
    print(f'markup: {markup_sudoku(EXAMPLE_GRID_HOLES)}')
    fill_naked_single(EXAMPLE_GRID_HOLES, markup_sudoku(EXAMPLE_GRID_HOLES))
    print(f'sudoku after filling naked singles: {print_sudoku(EXAMPLE_GRID_HOLES)}')
