from sudoku_omakase import SudokuImage
import pytest
import pytest


SUDOKU_PATHS = [
    "tests/test_images/IMG_0120.jpg",
    "tests/test_images/sudoku_easy.png",
    "tests/test_images/sudoku_hard.png",
    "tests/test_images/sudoku.png",
    "tests/test_images/sudoku1.png",
    "tests/test_images/test-sudoku.jpg",]

CORRECT_BOARDS = [
    [
        [0, 6, 0, 0, 0, 7, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 8, 2, 0],
        [0, 0, 8, 0, 0, 0, 4, 0, 0],
        [0, 0, 6, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 8, 0, 6, 5, 0, 0],
        [0, 2, 3, 0, 1, 5, 0, 0, 0],
        [2, 0, 0, 7, 0, 0, 0, 0, 0],
        [0, 7, 0, 9, 0, 8, 0, 0, 3],
        [0, 3, 0, 0, 0, 2, 0, 1, 7]
    ],
    [
        [0, 0, 7, 0, 0, 2, 0, 4, 8],
        [1, 2, 9, 3, 0, 0, 0, 0, 0],
        [0, 0, 8, 6, 3, 0, 7, 0, 1],
        [9, 7, 0, 2, 1, 0, 0, 0, 0],
        [4, 1, 0, 0, 0, 8, 0, 0, 3],
        [0, 0, 0, 0, 0, 3, 1, 5, 7],
        [3, 0, 0, 0, 8, 6, 4, 0, 0],
        [0, 5, 4, 0, 7, 0, 6, 0, 0],
        [7, 8, 0, 0, 0, 4, 3, 1, 0]
    ],
    [
        [0, 5, 0, 9, 1, 0, 0, 0, 0],
        [0, 0, 6, 0, 3, 0, 0, 0, 0],
        [0, 0, 1, 7, 0, 0, 3, 0, 4],
        [0, 0, 3, 0, 0, 8, 1, 0, 0],
        [2, 0, 0, 0, 9, 3, 0, 0, 5],
        [0, 4, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 9],
        [0, 0, 0, 0, 5, 0, 4, 2, 1],
        [0, 0, 0, 0, 0, 2, 0, 7, 0]
    ],
    [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ],
    [
        [0, 0, 5, 8, 7, 0, 0, 0, 1],
        [0, 0, 7, 2, 0, 0, 0, 6, 5],
        [0, 3, 6, 0, 0, 0, 0, 4, 0],
        [1, 8, 9, 5, 4, 0, 2, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 3, 0, 6, 8, 7, 9, 4],
        [0, 7, 0, 0, 0, 0, 5, 8, 0],
        [6, 9, 0, 0, 0, 3, 4, 0, 0],
        [3, 0, 0, 0, 8, 2, 6, 0, 0]
    ],
    [
        [8, 0, 0, 0, 1, 0, 0, 0, 9],
        [0, 5, 0, 8, 0, 7, 0, 1, 0],
        [0, 0, 4, 0, 9, 0, 7, 0, 0],
        [0, 6, 0, 7, 0, 1, 0, 2, 0],
        [5, 0, 8, 0, 6, 0, 1, 0, 7],
        [0, 1, 0, 5, 0, 2, 0, 9, 0],
        [0, 0, 7, 0, 4, 0, 6, 0, 0],
        [0, 8, 0, 3, 0, 9, 0, 4, 0],
        [3, 0, 0, 0, 5, 0, 0, 0, 8]
    ]
]

CASES = list(zip(SUDOKU_PATHS, CORRECT_BOARDS))

MODEL_TYPE = "BIG"

@pytest.mark.parametrize(("path", "correct"), CASES)
def test_extraction(path, correct) -> None:
    sudoku = SudokuImage(path, model_type=MODEL_TYPE)
    extracted = sudoku.board
    wrong = calculate_wrong_cells(extracted, correct)
    accuracy = 1 - wrong / 81
    assert accuracy >= 0.95, f"{path}: {wrong} mismatched cells"
    

def calculate_wrong_cells(extracted, correct):
    wrong_cells = 0
    for i in range(9):
        for j in range(9):
            if extracted[i][j] != correct[i][j]:
                wrong_cells += 1
    return wrong_cells

