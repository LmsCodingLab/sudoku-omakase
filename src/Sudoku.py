from pathlib import Path
import numpy as np
import numpy.typing as npt 
from src.photo_processing import extract_fields, extract_numbers, extract_sudoku, resize_fields
from src.solver import solve_sudoku

#TODO implement dev_mode
class Sudoku:
    _GRID_SIZE = 9
    _BLOCK_SIZE = 3
    def __init__(self):
        self.source: str = ""
        self.solved: bool = False
        self.grid: npt.NDArray[np.int8] = np.zeros((self._GRID_SIZE, self._GRID_SIZE), dtype=np.int8)
        self.original_grid: npt.NDArray[np.int8] = np.zeros((self._GRID_SIZE, self._GRID_SIZE), dtype=np.int8)

    def __str__(self) -> str:
        def printer(grid: npt.NDArray[np.int8]) -> str:
            to_output = ""
            for row_idx in range(self._GRID_SIZE):
                if row_idx % self._BLOCK_SIZE == 0 and row_idx != 0:
                    to_output += "— " * (self._GRID_SIZE + self._BLOCK_SIZE - 1) + '\n' # separator between blocks
                row = ""
                for col_idx in range(self._GRID_SIZE):
                    if col_idx % self._BLOCK_SIZE == 0 and col_idx != 0:
                        row += "| " # separator between blocks
                    row += f"{grid[row_idx, col_idx]} "
                to_output += row + '\n'
            return to_output
        
        output = 'Sudoku From: ' + str(self.source) + '\n'
        output += 'Original Grid:\n'
        output += printer(self.original_grid)
        output += 'Current Grid:\n'
        output += printer(self.grid)
        output += 'Solved: ' + str(self.solved)
        return output

    def get_grid(self) -> npt.NDArray[np.int8]:
        return self.grid.copy()
    
    def is_solved(self) -> bool:
        return self.solved
    
    def extract_grid(self, source: str) -> npt.NDArray[np.int8]:
        p = Path(source).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {source}")
        self.source = source
        self.solved = False

        clean_sudoku = extract_sudoku(source, dev_mode=False)
        fields = extract_fields(clean_sudoku, dev_mode=False)
        ready_fields = resize_fields(fields, dev_mode=False)
        self.grid = extract_numbers(ready_fields, model_type="resnext", dev_mode=False)

        self.original_grid = self.grid.copy()

        return self.grid.copy()
    
    def set_grid(self, grid: npt.NDArray[np.int8]) -> None:
        if grid.shape != (self._GRID_SIZE, self._GRID_SIZE):
            raise ValueError(f"Grid must be of shape ({self._GRID_SIZE}, {self._GRID_SIZE})")
        if not np.all((grid >= 0) & (grid <= self._GRID_SIZE)):
            raise ValueError(f"Grid values must be between 0 and {self._GRID_SIZE} (inclusive)")
        
        self.solved = False
        self.source = ""
        self.grid = grid
        self.original_grid = grid.copy()

    def solve(self) -> bool:
        self.solved = solve_sudoku(self.grid)
        return self.solved
    

# Example usage
if __name__ == "__main__":
    sudoku = Sudoku()
    sudoku.extract_grid("src/test/IMG_0120.jpg")
    sudoku.solve()
    print(sudoku)

