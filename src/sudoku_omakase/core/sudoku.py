import numpy.typing as npt
import numpy as np
from typing import Final

from sudoku_omakase.core.solver import dfs, run_passes

class Sudoku:
	"""
	A class representing the Sudoku board
	"""
	GRID_SIZE: Final[int] = 9
	BLOCK_SIZE: Final[int] = 3
	VALID_VALUES: Final[set[int]] = set(range(1, GRID_SIZE + 1))

	def __init__(self, board: npt.NDArray[np.int8]):
		if board.shape != (self.GRID_SIZE, self.GRID_SIZE):
			raise ValueError(f"Board must be a {self.GRID_SIZE}x{self.GRID_SIZE} grid, got {board.shape}")
		self.board = board
		self.solved = self.is_solved_sudoku()
	
	def __str__(self) -> str:
		output = 'Original Grid:\n'
		output += self._print_helper()
		output += 'Solved: ' + str(self.solved)
		return output

	def is_solved_sudoku(self) -> bool:
		"""
		Checks if a given Sudoku grid is solved by ensuring that it contains only valid numbers and that each row, column, and 3x3 block contains unique values. 

		Parameters:
		- self: Sudoku, the instance of the Sudoku class containing the board to check.

		Returns:
		- bool, True if the grid is a solved Sudoku grid, False otherwise.
		"""
		
		if not self._contains_only_valid_numbers():
			return False
		return self._rows_are_unique() and self._columns_are_unique() and self._blocks_are_unique()
	
	def solve(self) -> bool:
		"""
		Runs Crook-style passes until no deduction rule makes further progress.

		Parameters:
		- sudoku: Sudoku, the instance of the Sudoku class containing the board to solve.

		Returns:
		- bool, True if the resulting grid is a valid Sudoku, False otherwise.
		"""
		run_passes(self.board)
		has_zero = np.any(self.board == 0)

		if has_zero:
			if not dfs(self.board):
				return False            
		
		self.solved = self.is_solved_sudoku()
		return self.solved
		
	def _contains_only_valid_numbers(self) -> bool:
		"""
		Checks whether every entry falls within the allowed digit range.

		Parameters:
		- self: Sudoku, the instance of the Sudoku class containing the board to check.

		Returns:
		- bool, True if all values are between 1 and 9, False otherwise.
		"""
		return bool(np.all(np.isin(self.board, list(self.VALID_VALUES))))
	
	def _rows_are_unique(self) -> bool:
		"""
		Checks that each row contains unique values.

		Parameters:
		- self: Sudoku, the instance of the Sudoku class containing the board to check.

		Returns:
		- bool, True if all rows contain unique values, False otherwise.
		"""
		return all(len(np.unique(self.board[row_idx, :])) == self.GRID_SIZE for row_idx in range(self.GRID_SIZE))

	def _columns_are_unique(self) -> bool:
		"""
		Checks whether each column contains nine distinct digits.

		Parameters:
		- self: Sudoku, the instance of the Sudoku class containing the board to check.

		Returns:
		- bool, True if every column has unique values, False otherwise.
		"""
		return all(len(np.unique(self.board[:, col_idx])) == self.GRID_SIZE for col_idx in range(self.GRID_SIZE))

	def _blocks_are_unique(self) -> bool:
		"""
		Checks whether each 3x3 block holds nine distinct digits.

		Parameters:
		- self: Sudoku, the instance of the Sudoku class containing the board to check.

		Returns:
		- bool, True if every block has unique values, False otherwise.
		"""
		for start_row in range(0, self.GRID_SIZE, self.BLOCK_SIZE):
			for start_col in range(0, self.GRID_SIZE, self.BLOCK_SIZE):
				block = self.board[start_row:start_row + self.BLOCK_SIZE, 
							start_col:start_col + self.BLOCK_SIZE]
				if len(np.unique(block)) != self.GRID_SIZE:
					return False
		return True
	
	def _print_helper(self) -> str:
		"""
		Helper function to format the Sudoku grid as a string for display purposes.

		Parameters:
		- self: Sudoku, the instance of the Sudoku class containing the board to format.

		Returns:
		- str, a formatted string representation of the Sudoku grid.
		"""
		to_output = ""
		for row_idx in range(self.GRID_SIZE):
			if row_idx % self.BLOCK_SIZE == 0 and row_idx != 0:
				to_output += "— " * (self.GRID_SIZE + self.BLOCK_SIZE - 1) + '\n' # separator between blocks
			row = ""
			for col_idx in range(self.GRID_SIZE):
				if col_idx % self.BLOCK_SIZE == 0 and col_idx != 0:
					row += "| " # separator between blocks
				row += f"{self.board[row_idx, col_idx]} "
			to_output += row + '\n'
		return to_output + '\n'