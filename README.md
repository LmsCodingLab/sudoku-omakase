This project is a python package for solving sudokus from images. It uses computer vision to extract the sudoku grid and digits, and then uses a combined algorithm with first utilizing
a simple implementation of crooks algorithm and backtracking if needed.

## 1. Installation
**Virtual environment**

Not exactly necessary but highly recommended.

```sh
python -m venv .venv
```
```sh
source .venv/bin/activate
```

**PyTorch note (CPU vs CUDA)**
> If you have an NVIDIA GPU and want CUDA acceleration, install a PyTorch build that matches your CUDA version.
> Use the official selector to get the correct command for your system:
> https://pytorch.org/get-started/locally/
>
> CPU-only users can typically install the default wheels.
```sh
pip install torch torchvision
```

**Build**
```sh
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sudoku-omakase==0.1.0
```
!DANGER Packages on testPyPI are temporary, it maybe already deleted. Please contact if so.

## 2. Quickstart

```python
from sudoku_omakase import SudokuImage

sudoku = SudokuImage(source="path/to/sudoku/image.jpg", model_type="BIG")
print("Original:")
print(sudoku)
sudoku.solve()
print("Solved:")
print(sudoku)
```

Or if you already have the grid and just want to solve it:

```python
from sudoku_omakase import Sudoku

grid = np.array([
        [0, 0, 0, 2, 0, 0, 0, 6, 3],
        [3, 0, 0, 0, 0, 5, 4, 0, 1],
        [0, 0, 1, 0, 0, 3, 9, 8, 0],
        [0, 0, 0, 0, 0, 0, 0, 9, 0],
        [0, 0, 0, 5, 3, 8, 0, 0, 0],
        [0, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 6, 3, 0, 0, 5, 0, 0],
        [5, 0, 3, 7, 0, 0, 0, 0, 8],
        [4, 7, 0, 0, 0, 1, 0, 0, 0]
    ], dtype=np.int8)

sudoku = Sudoku(grid)
print("Original:")
print(sudoku)
sudoku.solve()
print("Solved:")
print(sudoku)
```

## 3. Usage

1. **Prepare the image:** Capture a high-contrast, rectangular photo of the puzzle page. Crop borders or rotate the file so the grid is almost front-facing; blurry edges usually cause extraction glitches.
2. **Instantiate `SudokuImage`:** Load the image with `SudokuImage(source=..., model_type=...)`. `model_type` selects the OCR backbone (`SMALL`, `NORMAL`, `BIG`). The bigger the model, the slower but more accurate. 
3. **Call `solve()`:** `sudoku.solve()` runs digit classification and the backtracking solver. If solving fails, inspect the errors — most issues stem from misread digits or a poorly detected grid.
4. **Read the result:** Use `print(sudoku)` or `sudoku.board` to access the solved 9×9 array.

### Common parameters

- `model_type`: `"SMALL"` (fast, good lighting), `"NORMAL"` (balanced default), `"BIG"` (handles noisy phones scans).
- `source`: File path, numpy array, or bytes; make sure relative paths exist within your working directory.


## Models
Currently, there are 3 pretrained models for sudoku recognition. The model will be automatically installed, when needed.

- `"SMALL"` (179 KB)
- `"NORMAL"` (42 MB)
- `"BIG"` (332 MB)

## 4. How it works

### `SudokuImage`

The high-level entry point that turns a photo into a solvable grid lives in [src/sudoku_omakase/core/sudoku_image.py](src/sudoku_omakase/core/sudoku_image.py). The class inherits from `Sudoku`, so every extracted board immediately gains solving capabilities.

- **Preprocessing:** `prepare_image()` normalizes contrast, `detect_grid()` finds the outer contour, and `warp_image()` produces a top-down crop so downstream steps see a perfect square.
- **Cell extraction:** `extract_fields()` slices the warped grid into 81 tiles, and `resize_fields()` brings each tile to the expected CNN input size.
- **Digit prediction:** `extract_numbers()` runs the configured OCR model (`ModelType` cast from the `model_type` string) and returns a `9x9` NumPy array with zeros for uncertain cells.
- Because it subclasses `Sudoku`, calling `SudokuImage.solve()` immediately reuses the same solver logic described below.

### `Sudoku` solver core

The solving logic is implemented in [src/sudoku_omakase/core/solver.py](src/sudoku_omakase/core/solver.py) and wrapped by [src/sudoku_omakase/core/sudoku.py](src/sudoku_omakase/core/sudoku.py).

- **Crook-style passes:** `run_passes()` repeatedly applies human-style deductions such as naked singles, hidden singles, and naked subsets. Each technique mutates the board in place and recalculates candidate markup when progress is made.
- **Backtracking fallback:** If deterministic passes stall, `dfs()` performs a depth-first search with heuristics (choose the cell with the fewest candidates first) to resolve remaining ambiguity.
- **Board validity:** `Sudoku.is_solved_sudoku()` ensures every row, column, and 3x3 block contains digits 1–9 without repetition; this guard runs after solving and after manual mutations.
- **Mutations and inspection:** Helper methods like `mutate_one_field()` or `mutate_fields()` support editing puzzles programmatically, while `__str__()` produces a readable grid with block separators for debugging.

