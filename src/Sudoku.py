from pathlib import Path
import numpy as np
import numpy.typing as npt 
from src.photo_processing import extract_fields, extract_numbers, extract_sudoku, resize_fields

#TODO implement dev_mode
class Sudoku:
    def __init__(self):
        self.source: str = ""
        self.solved: bool = False
        self.grid: npt.NDArray[np.int8] = np.zeros((9, 9), dtype=np.int8)

        self.__GRID_SIZE = 9
        self.__BLOCK_SIZE = 3

    def get_grid(self) -> npt.NDArray[np.int8]:
        return self.grid
    
    def is_solved(self) -> bool:
        return self.solved
    
    def extract_grid(self, source: str) -> npt.NDArray[np.int8]:
        p = Path(source).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {source}")
        self.source = source

        clean_sudoku = extract_sudoku(source, dev_mode=False)
        fields = extract_fields(clean_sudoku, dev_mode=False)
        ready_fields = resize_fields(fields, dev_mode=False)
        self.grid = extract_numbers(ready_fields, model_type="resnext", dev_mode=False)

        return self.grid
    
    def set_grid(self, grid: npt.NDArray[np.int8]) -> None:
        if grid.shape != (self.__GRID_SIZE, self.__GRID_SIZE):
            raise ValueError(f"Grid must be of shape ({self.__GRID_SIZE}, {self.__GRID_SIZE})")
        if not np.all((grid >= 0) & (grid <= 9)):
            raise ValueError("Grid values must be between 0 and 9 (inclusive)")
        self.grid = grid
    
    def __str__(self) -> str:
        output = 'Sudoku From: ' + str(self.source) + '\n'

        for row_idx in range(self.__GRID_SIZE):
            if row_idx % self.__BLOCK_SIZE == 0 and row_idx != 0:
                output += "— " * (self.__GRID_SIZE + self.__BLOCK_SIZE - 1) + '\n' # separator between blocks
            row = ""
            for col_idx in range(self.__GRID_SIZE):
                if col_idx % self.__BLOCK_SIZE == 0 and col_idx != 0:
                    row += "| " # separator between blocks
                row += f"{self.grid[row_idx, col_idx]} "
            output += row + '\n'

        output += 'Solved: ' + str(self.solved)
        return output

# Example usage
if __name__ == "__main__":
    sudoku = Sudoku()
    sudoku.extract_grid("src/test/IMG_0120.jpg")
    print(sudoku)

