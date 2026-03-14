import numpy as np
import pytest

from sudoku_omakase.core.sudoku import Sudoku

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


@pytest.mark.parametrize(
    "grid",
    [
        EXAMPLE_GRID_HOLES,
        EXAMPLE_GRID_SHORTZ,
        EXAMPLE_GRID_X_WING,
    ],
    ids=["holes", "shortz", "x-wing"],
)
def test_solver_fills_examples(grid: np.ndarray) -> None:
    sudoku = Sudoku(grid.copy())
    assert sudoku.solve() is True
    assert sudoku.solved is True
    assert not np.any(sudoku.board == 0)


def test_solver_keeps_completed_grid() -> None:
    sudoku = Sudoku(EXAMPLE_GRID.copy())
    assert sudoku.is_solved_sudoku() is True
    assert sudoku.solve() is True
    np.testing.assert_array_equal(sudoku.board, EXAMPLE_GRID)