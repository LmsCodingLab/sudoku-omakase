This project is a Python package for solving sudokus from images. It uses computer vision to extract the sudoku grid and digits, and then uses a combined algorithm with first utilizing
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
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sudoku-omakase==0.1.2
```
> **⚠️ WARNING:** Packages on TestPyPI are temporary and may already be deleted. Please contact the maintainer if this occurs.

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
import numpy as np

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


## 4. Models
Currently, there are 3 pretrained models for sudoku recognition. The model will be **automatically installed**, when needed.

- `"SMALL"` (179 KB)
- `"NORMAL"` (42 MB)
- `"BIG"` (332 MB)

> **⚠️ WARNING:** Automatic installation of the models on the first use.

## 5. Public API overview

### `SudokuImage`

Use this class when you have a picture. Point it to any JPEG/PNG path, pick a `model_type`, and call `solve()`. Afterwards you can:

- Inspect the recognized puzzle via `sudoku.board` or `print(sudoku)`.
- Export the solved grid as NumPy `to_numpy()`, Python lists `to_list()`, or mutate cells with `mutate_one_field()` if something looks off.
- Inspect the processed sudoku image via `sudoku.get_extracted_sudoku_image()`.

### `Sudoku`

Use this class when you already own the 9×9 array. Initialize it with a NumPy grid, then:

- Run `solve()` to apply the built-in deduction plus backtracking routine.
- Check completion state via `sudoku.solved` or revalidate with `is_solved_sudoku()`.
- Adjust entries using `mutate_fields()` or inspect them cell-by-cell; printing the instance shows a formatted board with block separators.

