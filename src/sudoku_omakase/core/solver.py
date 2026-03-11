import numpy as np
import numpy.typing as npt
from collections import Counter
from itertools import combinations
from sudoku_omakase.core.sudoku import Sudoku

<<<<<<<< HEAD:src/sudoku_omakase/core/solver.py
GRID_SIZE = Sudoku.GRID_SIZE
BLOCK_SIZE = Sudoku.BLOCK_SIZE
VALID_VALUES = Sudoku.VALID_VALUES

# TODO Move these to tests/assets.py
EXAMPLE_GRID = np.array([
    [4, 1, 5, 2, 7, 9, 3, 8, 6],
    [7, 3, 9, 6, 1, 8, 4, 5, 2],
    [2, 6, 8, 4, 5, 3, 9, 7, 1],
    [3, 2, 6, 5, 8, 7, 1, 4, 9],
    [1, 5, 7, 9, 6, 4, 2, 3, 8],
    [8, 9, 4, 1, 3, 2, 5, 6, 7],
    [9, 8, 2, 7, 4, 5, 6, 1, 3],
    [5, 7, 1, 3, 2, 6, 8, 9, 4],
    [6, 4, 3, 8, 9, 1, 7, 2, 5]])
========
# MOVED TO CORE/SOLVER.PY
GRID_SIZE = 9
BLOCK_SIZE = 3
VALID_VALUES = set(range(1, GRID_SIZE + 1))
valid_pool = np.array(sorted(VALID_VALUES))

EXAMPLE_GRID = np.array([[4, 1, 5, 2, 7, 9, 3, 8, 6],
                         [7, 3, 9, 6, 1, 8, 4, 5, 2],
                         [2, 6, 8, 4, 5, 3, 9, 7, 1],
                         [3, 2, 6, 5, 8, 7, 1, 4, 9],
                         [1, 5, 7, 9, 6, 4, 2, 3, 8],
                         [8, 9, 4, 1, 3, 2, 5, 6, 7],
                         [9, 8, 2, 7, 4, 5, 6, 1, 3],
                         [5, 7, 1, 3, 2, 6, 8, 9, 4],
                         [6, 4, 3, 8, 9, 1, 7, 2, 5]])
>>>>>>>> 6867c7e (71 add complete the core (#78)):src/sudoku_omakase/solver.py

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

EXAMPLE_GRID_SHORTZ = np.array([
    [0, 3, 9, 5, 0, 0, 0, 0, 0],
    [0, 0, 1, 8, 0, 9, 0, 7, 0],
    [0, 0, 0, 0, 1, 0, 9, 0, 4],
    [1, 0, 0, 4, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 7, 0, 0, 0, 8, 6, 0],
    [0, 0, 6, 7, 0, 8, 2, 0, 0],
    [0, 1, 0, 0, 9, 0, 0, 0, 5],
    [0, 0, 0, 0, 0, 1, 0, 0, 8]])

EXAMPLE_GRID_X_WING = np.array([
    [0, 0, 0, 2, 0, 0, 0, 6, 3],
    [3, 0, 0, 0, 0, 5, 4, 0, 1],
    [0, 0, 1, 0, 0, 3, 9, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 9, 0],
    [0, 0, 0, 5, 3, 8, 0, 0, 0],
    [0, 3, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 6, 3, 0, 0, 5, 0, 0],
    [5, 0, 3, 7, 0, 0, 0, 0, 8],
    [4, 7, 0, 0, 0, 1, 0, 0, 0]
])

<<<<<<<< HEAD:src/sudoku_omakase/core/solver.py
def solve_sudoku(sudoku: Sudoku) -> bool:
========
def parse_sudoku() -> npt.NDArray[np.int8]:
    """
    Reads a Sudoku grid from standard input as comma-separated rows.

    Parameters:
    - None

    Returns:
    - np.ndarray, the populated Sudoku grid.
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    for row_idx in range(GRID_SIZE):
        raw_row = input(f"please enter row {row_idx + 1}: ")
        grid[row_idx] = list(map(int, raw_row.split(',')))
    return grid

# MOVED TO CORE/SOLVER.PY
def solve_sudoku(grid: npt.NDArray[np.int8]) -> bool:
>>>>>>>> 6867c7e (71 add complete the core (#78)):src/sudoku_omakase/solver.py
    """
    Runs Crook-style passes until no deduction rule makes further progress.

    Parameters:
    - sudoku: Sudoku, the instance of the Sudoku class containing the board to solve.

    Returns:
    - bool, True if the resulting grid is a valid Sudoku, False otherwise.
    """
    run_passes(sudoku.board)
    has_zero = np.any(sudoku.board == 0)

    if has_zero:
        if not _dfs(sudoku.board):
            return False            
    
<<<<<<<< HEAD:src/sudoku_omakase/core/solver.py
    return sudoku.solved

========
    return is_solved_sudoku(grid)

# MOVED TO CORE/SUDOKU.PY
def is_solved_sudoku(grid: npt.NDArray[np.int8]) -> bool:
    """
    Checks if a given Sudoku grid is solved by ensuring that it contains only valid numbers and that each row, column, and 3x3 block contains unique values. 

    Parameters:
    - grid: np.ndarray, the Sudoku grid to be checked.

    Returns:
    - bool, True if the grid is a solved Sudoku grid, False otherwise.
    """
    
    if not contains_only_valid_numbers(grid):
        return False
    return rows_are_unique(grid) and columns_are_unique(grid) and blocks_are_unique(grid)


# Helper functions for Sudoku validation and solving
# MOVED TO CORE/SUDOKU.PY
def contains_only_valid_numbers(grid: npt.NDArray[np.int8]) -> bool:
    """
    Checks whether every entry falls within the allowed digit range.

    Parameters:
    - grid: np.ndarray, the Sudoku grid to inspect.

    Returns:
    - bool, True if all values are between 1 and 9, False otherwise.
    """
    return bool(np.all(np.isin(grid, valid_pool)))

# MOVED TO CORE/SUDOKU.PY
def rows_are_unique(grid: npt.NDArray[np.int8]) -> bool:
    """
    Checks whether each row contains nine distinct digits.

    Parameters:
    - grid: np.ndarray, the Sudoku grid to inspect.

    Returns:
    - bool, True if every row has unique values, False otherwise.
    """
    return all(len(np.unique(grid[row_idx, :])) == GRID_SIZE for row_idx in range(GRID_SIZE))

# MOVED TO CORE/SUDOKU.PY
def columns_are_unique(grid: npt.NDArray[np.int8]) -> bool:
    """
    Checks whether each column contains nine distinct digits.

    Parameters:
    - grid: np.ndarray, the Sudoku grid to inspect.

    Returns:
    - bool, True if every column has unique values, False otherwise.
    """
    return all(len(np.unique(grid[:, col_idx])) == GRID_SIZE for col_idx in range(GRID_SIZE))

# MOVED TO CORE/SUDOKU.PY
def blocks_are_unique(grid: npt.NDArray[np.int8]) -> bool:
    """
    Checks whether each 3x3 block holds nine distinct digits.

    Parameters:
    - grid: np.ndarray, the Sudoku grid to inspect.

    Returns:
    - bool, True if every block has unique values, False otherwise.
    """
    for start_row in range(0, GRID_SIZE, BLOCK_SIZE):
        for start_col in range(0, GRID_SIZE, BLOCK_SIZE):
            block = grid[start_row:start_row + BLOCK_SIZE, 
                         start_col:start_col + BLOCK_SIZE]
            if len(np.unique(block)) != GRID_SIZE:
                return False
    return True

# MOVED TO CORE/SOLVER.PY
def identify_candidates(grid: npt.NDArray[np.int8], row: int, column: int) -> set[int]:
    """
    Finds the set of digits that can legally occupy a given cell.

    Parameters:
    - grid: np.ndarray, the Sudoku grid used for context.
    - row: int, the row index of the target cell.
    - column: int, the column index of the target cell.

    Returns:
    - set[int], the remaining viable digits for the cell.
    """
    
    #check row and column
    used = set(grid[row]) | {grid[i, column] for i in range(GRID_SIZE)}
    
    #check block
    block_row_start = (row // BLOCK_SIZE) * BLOCK_SIZE
    block_col_start = (column // BLOCK_SIZE) * BLOCK_SIZE
    for r in range(block_row_start, block_row_start + BLOCK_SIZE):
        for c in range(block_col_start, block_col_start + BLOCK_SIZE):
            used.add(grid[r, c])
            
    return VALID_VALUES - used

# MOVED TO CORE/SOLVER.PY
def markup_sudoku(grid: npt.NDArray[np.int8]) -> npt.NDArray[np.object_]:
    """
    Builds a grid of candidate sets for every unsolved cell.

    Parameters:
    - grid: np.ndarray, the Sudoku grid to annotate.

    Returns:
    - np.ndarray, an object grid containing either digits or candidate sets.
    """
    markup = np.copy(grid).astype(object)
    
    for row, col in np.ndindex(grid.shape): 
        if grid[row, col] == 0:
            candidates = identify_candidates(grid, row, col)
            markup[row, col] = candidates

    return markup

# MOVED TO CORE/SOLVER.PY
>>>>>>>> 6867c7e (71 add complete the core (#78)):src/sudoku_omakase/solver.py
def run_passes(grid: npt.NDArray[np.int8]) -> None:
    """
    Applies deduction strategies repeatedly until no further progress occurs.

    Parameters:
    - grid: np.ndarray, the Sudoku grid to mutate in place.

    Returns:
    - None
    """
    markup = markup_sudoku(grid)

    while True:
        progress = False

        if apply_naked_singles(grid, markup):
            markup = markup_sudoku(grid)
            progress = True
            continue

        if apply_hidden_singles(grid, markup):
            markup = markup_sudoku(grid)
            progress = True
            continue

        if apply_naked_subsets(markup):
            progress = True
            continue

        if not progress:
            break
        
def markup_sudoku(grid: npt.NDArray[np.int8]) -> npt.NDArray[np.object_]:
    """
    Builds a grid of candidate sets for every unsolved cell.

<<<<<<<< HEAD:src/sudoku_omakase/core/solver.py
    Parameters:
    - grid: np.ndarray, the Sudoku grid to annotate.

    Returns:
    - np.ndarray, an object grid containing either digits or candidate sets.
    """
    markup = np.copy(grid).astype(object)
    
    for row, col in np.ndindex(grid.shape): 
        if grid[row, col] == 0:
            candidates = identify_candidates(grid, row, col)
            markup[row, col] = candidates

    return markup

def identify_candidates(grid: npt.NDArray[np.int8], row: int, column: int) -> set[int]:
    """
    Finds the set of digits that can legally occupy a given cell.

    Parameters:
    - grid: np.ndarray, the Sudoku grid used for context.
    - row: int, the row index of the target cell.
    - column: int, the column index of the target cell.

    Returns:
    - set[int], the remaining viable digits for the cell.
    """
    
    #check row and column
    used = set(grid[row]) | {grid[i, column] for i in range(GRID_SIZE)}
    
    #check block
    block_row_start = (row // BLOCK_SIZE) * BLOCK_SIZE
    block_col_start = (column // BLOCK_SIZE) * BLOCK_SIZE
    for r in range(block_row_start, block_row_start + BLOCK_SIZE):
        for c in range(block_col_start, block_col_start + BLOCK_SIZE):
            used.add(grid[r, c])
            
    return VALID_VALUES - used

========
# MOVED TO CORE/SOLVER.PY
>>>>>>>> 6867c7e (71 add complete the core (#78)):src/sudoku_omakase/solver.py
def apply_naked_singles(grid: npt.NDArray[np.int8], markup: npt.NDArray[np.object_]) -> bool:
    """
    Places digits whenever a cell has exactly one candidate.

    Parameters:
    - grid: np.ndarray, the Sudoku grid to mutate in place.
    - markup: np.ndarray, the candidate grid aligned with the Sudoku grid.

    Returns:
    - bool, True if any placements were made, False otherwise.
    """
    change = False
    for row, column in np.ndindex(grid.shape):
        if isinstance(markup[row, column], set) and len(markup[row, column]) == 1:
            grid[row, column] = next(iter(markup[row, column]))
            change = True
    return change
    
# MOVED TO CORE/SOLVER.PY
def apply_hidden_singles(grid: npt.NDArray[np.int8], markup: npt.NDArray[np.object_]) -> bool:
    """
    Searches rows, columns, and blocks for digits with a single viable location.

    Parameters:
    - grid: np.ndarray, the Sudoku grid to mutate in place.
    - markup: np.ndarray, the candidate grid aligned with the Sudoku grid.

    Returns:
    - bool, True if any digits were placed, False otherwise.
    """
    if _apply_hidden_single_rows(grid, markup):
        return True
    if _apply_hidden_single_columns(grid, markup):
        return True
    if _apply_hidden_single_blocks(grid, markup):
        return True
    return False

# MOVED TO CORE/SOLVER.PY
def apply_naked_subsets(markup: npt.NDArray[np.object_]) -> bool:
    """
    Removes digits covered by naked pairs, triples, or quads in each unit.

    Parameters:
    - markup: np.ndarray, the candidate grid aligned with the Sudoku grid.

    Returns:
    - bool, True if any candidates were removed, False otherwise.
    """
    changed = False
    if _apply_naked_subsets_rows(markup):
        changed = True
    if _apply_naked_subsets_columns(markup):
        changed = True
    if _apply_naked_subsets_blocks(markup):
        changed = True
    return changed

# MOVED TO CORE/SOLVER.PY
def _dfs(grid: npt.NDArray[np.int8]) -> bool:
    """
    Solves the Sudoku puzzle using backtracking search.
    
    Parameters:
    - grid: np.ndarray, the Sudoku grid to solve in place.
    
    Returns:
        - bool, True if a solution was found, False otherwise.
    """
    markup = markup_sudoku(grid)
    empty_cells = [(row, column) for row, column in np.ndindex(GRID_SIZE, GRID_SIZE) if grid[row, column] == 0]
    if len(empty_cells) == 0:
        return True
    
    cells_with_zero = [
        (row, column) for (row, column) in empty_cells if len(markup[row, column]) == 0
    ]
    
    if cells_with_zero:
        return False
    
    row, column = min(empty_cells, key = lambda pos: len(markup[pos]))
    for value in markup[row, column]:
        grid[row, column] = value
        if _dfs(grid):
            return True
        grid[row, column] = 0
        
    return False

# MOVED TO CORE/SOLVER.PY
def _apply_hidden_single_rows(grid: npt.NDArray[np.int8], markup: npt.NDArray[np.object_]) -> bool:
    """
    Finds digits that appear in only one candidate cell within a row.

    Parameters:
    - grid: np.ndarray, the Sudoku grid to mutate in place.
    - markup: np.ndarray, the candidate grid aligned with the Sudoku grid.

    Returns:
    - bool, True if a digit was placed, False otherwise.
    """
    for row_idx in range(GRID_SIZE):
        freq: Counter[int] = Counter()
        last_col: dict[int, int] = {}
        for col_idx in range(GRID_SIZE):
            cell_candidates = markup[row_idx, col_idx]
            if not isinstance(cell_candidates, set):
                continue
            for value in cell_candidates:
                freq[value] += 1
                last_col[value] = col_idx
        unique = [value for value, count in freq.items() if count == 1]
        if unique:
            value = unique[0]
            target_col = last_col[value]
            grid[row_idx, target_col] = value
            markup[row_idx, target_col] = value
            return True
    return False

# MOVED TO CORE/SOLVER.PY
def _apply_hidden_single_columns(grid: npt.NDArray[np.int8], markup: npt.NDArray[np.object_]) -> bool:
    """
    Finds digits confined to a single candidate cell within a column.

    Parameters:
    - grid: np.ndarray, the Sudoku grid to mutate in place.
    - markup: np.ndarray, the candidate grid aligned with the Sudoku grid.

    Returns:
    - bool, True if a digit was placed, False otherwise.
    """
    for col_idx in range(GRID_SIZE):
        freq: Counter[int] = Counter()
        last_row: dict[int, int] = {}
        for row_idx in range(GRID_SIZE):
            cell_candidates = markup[row_idx, col_idx]
            if not isinstance(cell_candidates, set):
                continue
            for value in cell_candidates:
                freq[value] += 1
                last_row[value] = row_idx
        unique = [value for value, count in freq.items() if count == 1]
        if unique:
            value = unique[0]
            target_row = last_row[value]
            grid[target_row, col_idx] = value
            markup[target_row, col_idx] = value
            return True
    return False

# MOVED TO CORE/SOLVER.PY
def _apply_hidden_single_blocks(grid: npt.NDArray[np.int8], markup: npt.NDArray[np.object_]) -> bool:
    """
    Resolves digits that fit only one location inside a 3x3 block.

    Parameters:
    - grid: np.ndarray, the Sudoku grid to mutate in place.
    - markup: np.ndarray, the candidate grid aligned with the Sudoku grid.

    Returns:
    - bool, True if a digit was placed, False otherwise.
    """
    for start_row in range(0, GRID_SIZE, BLOCK_SIZE):
        for start_col in range(0, GRID_SIZE, BLOCK_SIZE):
            freq: Counter[int] = Counter()
            last_pos: dict[int, tuple[int, int]] = {}
            for row_idx in range(start_row, start_row + BLOCK_SIZE):
                for col_idx in range(start_col, start_col + BLOCK_SIZE):
                    cell_candidates = markup[row_idx, col_idx]
                    if not isinstance(cell_candidates, set):
                        continue
                    for value in cell_candidates:
                        freq[value] += 1
                        last_pos[value] = (row_idx, col_idx)
            unique = [value for value, count in freq.items() if count == 1]
            if unique:
                value = unique[0]
                target_row, target_col = last_pos[value]
                grid[target_row, target_col] = value
                markup[target_row, target_col] = value
                return True
    return False

# MOVED TO CORE/SOLVER.PY
def _apply_naked_subsets_rows(markup: npt.NDArray[np.object_]) -> bool:
    """
    Identifies naked subsets along each row and removes their digits from peers.

    Parameters:
    - markup: np.ndarray, the candidate grid aligned with the Sudoku grid.

    Returns:
    - bool, True if any candidates were removed, False otherwise.
    """
    changed = False
    for row_idx in range(GRID_SIZE):
        candidate_cells: list[tuple[int, set[int]]] = []
        union_vals: set[int] = set()
        for col_idx in range(GRID_SIZE):
            cell = markup[row_idx, col_idx]
            if isinstance(cell, set) and 1 < len(cell) < GRID_SIZE:
                candidate_cells.append((col_idx, cell))
                union_vals.update(cell)
        if len(candidate_cells) < 2:
            continue
        max_subset = min(len(union_vals), GRID_SIZE - 1)
        for subset_size in range(2, max_subset + 1):
            for subset in combinations(union_vals, subset_size):
                subset_set = set(subset)
                matching_cols = [col for col, cell in candidate_cells if cell.issubset(subset_set)]
                if len(matching_cols) != subset_size or len(matching_cols) <= 1:
                    continue
                member_cols = set(matching_cols)
                for col_idx in range(GRID_SIZE):
                    if col_idx in member_cols:
                        continue
                    cell = markup[row_idx, col_idx]
                    if isinstance(cell, set):
                        before = len(cell)
                        cell.difference_update(subset_set)
                        if len(cell) != before:
                            changed = True
    return changed

# MOVED TO CORE/SOLVER.PY
def _apply_naked_subsets_columns(markup: npt.NDArray[np.object_]) -> bool:
    """
    Identifies naked subsets along each column and removes their digits from peers.

    Parameters:
    - markup: np.ndarray, the candidate grid aligned with the Sudoku grid.

    Returns:
    - bool, True if any candidates were removed, False otherwise.
    """
    changed = False
    for col_idx in range(GRID_SIZE):
        candidate_cells: list[tuple[int, set[int]]] = []
        union_vals: set[int] = set()
        for row_idx in range(GRID_SIZE):
            cell = markup[row_idx, col_idx]
            if isinstance(cell, set) and 1 < len(cell) < GRID_SIZE:
                candidate_cells.append((row_idx, cell))
                union_vals.update(cell)
        if len(candidate_cells) < 2:
            continue
        max_subset = min(len(union_vals), GRID_SIZE - 1)
        for subset_size in range(2, max_subset + 1):
            for subset in combinations(union_vals, subset_size):
                subset_set = set(subset)
                matching_rows = [row for row, cell in candidate_cells if cell.issubset(subset_set)]
                if len(matching_rows) != subset_size or len(matching_rows) <= 1:
                    continue
                member_rows = set(matching_rows)
                for row_idx in range(GRID_SIZE):
                    if row_idx in member_rows:
                        continue
                    cell = markup[row_idx, col_idx]
                    if isinstance(cell, set):
                        before = len(cell)
                        cell.difference_update(subset_set)
                        if len(cell) != before:
                            changed = True
    return changed

# MOVED TO CORE/SOLVER.PY
def _apply_naked_subsets_blocks(markup: npt.NDArray[np.object_]) -> bool:
    """
    Detects naked subsets within each block and removes their digits from peers.

    Parameters:
    - markup: np.ndarray, the candidate grid aligned with the Sudoku grid.

    Returns:
    - bool, True if any candidates were removed, False otherwise.
    """
    changed = False
    for start_row in range(0, GRID_SIZE, BLOCK_SIZE):
        for start_col in range(0, GRID_SIZE, BLOCK_SIZE):
            candidate_cells: list[tuple[tuple[int, int], set[int]]] = []
            union_vals: set[int] = set()
            for row_idx in range(start_row, start_row + BLOCK_SIZE):
                for col_idx in range(start_col, start_col + BLOCK_SIZE):
                    cell = markup[row_idx, col_idx]
                    if isinstance(cell, set) and 1 < len(cell) < GRID_SIZE:
                        candidate_cells.append(((row_idx, col_idx), cell))
                        union_vals.update(cell)
            if len(candidate_cells) < 2:
                continue
            max_subset = min(len(union_vals), GRID_SIZE - 1)
            for subset_size in range(2, max_subset + 1):
                for subset in combinations(union_vals, subset_size):
                    subset_set = set(subset)
                    matching_coords = [coord for coord, cell in candidate_cells if cell.issubset(subset_set)]
                    if len(matching_coords) != subset_size or len(matching_coords) <= 1:
                        continue
                    members = set(matching_coords)
                    for row_idx in range(start_row, start_row + BLOCK_SIZE):
                        for col_idx in range(start_col, start_col + BLOCK_SIZE):
                            if (row_idx, col_idx) in members:
                                continue
                            cell = markup[row_idx, col_idx]
                            if isinstance(cell, set):
                                before = len(cell)
                                cell.difference_update(subset_set)
                                if len(cell) != before:
                                    changed = True
    return changed
