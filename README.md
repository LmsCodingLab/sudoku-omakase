This project is a python package for solving sudokus from images. It uses computer vision to extract the sudoku grid and digits, and then uses a backtracking algorithm to solve the sudoku.

## 1. Installation

> **PyTorch note (CPU vs CUDA):**
> If you have an NVIDIA GPU and want CUDA acceleration, install a PyTorch build that matches your CUDA version.
> Use the official selector to get the correct command for your system:
> https://pytorch.org/get-started/locally/
>
> CPU-only users can typically install the default wheels.
```sh
pip install torch torchvision
```

```sh
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sudoku-omakase==0.1.0
```
!DANGER Packages on testPyPI are temporary, it maybe already deleted. Please contact if so.

## 2. Example Usage

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


