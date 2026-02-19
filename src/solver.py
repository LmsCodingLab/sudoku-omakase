import numpy as np

GRID_SIZE = 9
BLOCK_SIZE = 3
VALID_VALUES = tuple(range(1, GRID_SIZE + 1))


def parse_sudoku() -> np.ndarray:
    """Read a Sudoku grid from stdin, expecting comma separated rows."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for row_idx in range(GRID_SIZE):
        raw_row = input(f"please enter row {row_idx + 1}: ")
        grid[row_idx] = list(map(int, raw_row.split(',')))
    return grid


def contains_only_valid_numbers(grid: np.ndarray) -> bool:
    return bool(np.all(np.isin(grid, VALID_VALUES)))


def rows_are_unique(grid: np.ndarray) -> bool:
    return all(len(np.unique(grid[row_idx, :])) == GRID_SIZE for row_idx in range(GRID_SIZE))


def columns_are_unique(grid: np.ndarray) -> bool:
    return all(len(np.unique(grid[:, col_idx])) == GRID_SIZE for col_idx in range(GRID_SIZE))


def blocks_are_unique(grid: np.ndarray) -> bool:
    for start_row in range(0, GRID_SIZE, BLOCK_SIZE):
        for start_col in range(0, GRID_SIZE, BLOCK_SIZE):
            block = grid[start_row:start_row + BLOCK_SIZE, 
                         start_col:start_col + BLOCK_SIZE]
            if len(np.unique(block)) != GRID_SIZE:
                return False
    return True


def is_valid_sudoku(grid: np.ndarray) -> bool:
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


if __name__ == "__main__":
    sudoku_grid = parse_sudoku()
    print(f"sudoku_check: {is_valid_sudoku(sudoku_grid)}")
    print(sudoku_grid)