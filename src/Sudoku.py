from pathlib import Path
import numpy as np
import numpy.typing as npt 
from src.photo_processing import extract_fields, extract_numbers, extract_sudoku, resize_fields
from src.solver import solve_sudoku

class Sudoku:
    """
        A class representing a Sudoku puzzle, with methods to extract the grid from a photo, solve the puzzle, and display the current state of the grid.
    """
    _GRID_SIZE = 9
    _BLOCK_SIZE = 3
    def __init__(self, dev_mode: bool = False) -> None:
        """
            Initializes a new Sudoku instance with an empty grid and default values for source, solved status, and model type.

            Parameters:
            - dev_mode (bool): A flag to indicate whether to enable development mode for debugging purposes. Default is False.

            Returns:
            - None
        """
        self.source: str = ""
        self.solved: bool = False
        self.grid: npt.NDArray[np.int8] = np.zeros((self._GRID_SIZE, self._GRID_SIZE), dtype=np.int8)
        self.original_grid: npt.NDArray[np.int8] = np.zeros((self._GRID_SIZE, self._GRID_SIZE), dtype=np.int8)
        self.dev_mode: bool = dev_mode
        self.model_type: str = "resnet"

    def __str__(self) -> str:
        output = 'Sudoku From: ' + str(self.source) + '\n'
        output += 'Original Grid:\n'
        output += self._format_grid(self.original_grid)
        output += 'Current Grid:\n'
        output += self._format_grid(self.grid)
        output += 'Solved: ' + str(self.solved)
        return output

    def get_grid(self) -> npt.NDArray[np.int8]:
        return self.grid.copy()
    
    def is_solved(self) -> bool:
        return self.solved
    
    def from_photo(self, source: str, model_type: str = "resnet") -> npt.NDArray[np.int8]:
        """
            Extracts a Sudoku grid from a photo, processes it, and updates the instance's grid and original_grid attributes.

            Parameters:
            - source (str): The file path to the photo containing the Sudoku puzzle.
            - model_type (str): The type of model to use for number extraction. Default is "resnet".
        """
        p = Path(source).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {source}")
        self.source = source
        self.solved = False
        self.model_type = model_type

        clean_sudoku = extract_sudoku(source, dev_mode=self.dev_mode)
        fields = extract_fields(clean_sudoku, dev_mode=self.dev_mode)
        ready_fields = resize_fields(fields, dev_mode=self.dev_mode)
        self.grid = extract_numbers(ready_fields, model_type=self.model_type, dev_mode=self.dev_mode)

        self.original_grid = self.grid.copy()

        return self.grid.copy()
    
    def from_grid(self, grid: npt.NDArray[np.int8]) -> None:
        """
            Initializes the Sudoku instance with a given grid, validating the input for correct shape and value range.

            Parameters:
            - grid (npt.NDArray[np.int8]): A 9x9 numpy array representing the Sudoku grid, where empty cells are represented by 0 and filled cells contain values from 1 to 9.

            Returns:
            - None
        """
        if grid.shape != (self._GRID_SIZE, self._GRID_SIZE):
            raise ValueError(f"Grid must be of shape ({self._GRID_SIZE}, {self._GRID_SIZE})")
        if not np.all((grid >= 0) & (grid <= self._GRID_SIZE)):
            raise ValueError(f"Grid values must be between 0 and {self._GRID_SIZE} (inclusive)")
        
        self.solved = False
        self.source = ""
        self.grid = grid
        self.original_grid = grid.copy()

    def solve(self) -> bool:
        """
            Solves the Sudoku puzzle using a backtracking algorithm. Updates the grid with the solved state and sets the solved attribute to True if a solution is found.

            Returns:
            - bool: True if the puzzle was solved successfully, False otherwise.

        """
        self.solved = solve_sudoku(self.grid)
        return self.solved
    
    @staticmethod
    def _format_grid( grid: npt.NDArray[np.int8]) -> str:
        to_output = ""
        for row_idx in range(Sudoku._GRID_SIZE):
            if row_idx % Sudoku._BLOCK_SIZE == 0 and row_idx != 0:
                to_output += "— " * (Sudoku._GRID_SIZE + Sudoku._BLOCK_SIZE - 1) + '\n' # separator between blocks
            row = ""
            for col_idx in range(Sudoku._GRID_SIZE):
                if col_idx % Sudoku._BLOCK_SIZE == 0 and col_idx != 0:
                    row += "| " # separator between blocks
                row += f"{grid[row_idx, col_idx]} "
            to_output += row + '\n'
        return to_output
    

