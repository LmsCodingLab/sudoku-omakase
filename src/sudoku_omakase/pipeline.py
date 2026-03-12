from pathlib import Path
from typing import Literal

from sudoku_omakase.model.models import ModelType
from sudoku_omakase.vision.sudoku_image import SudokuImage
from sudoku_omakase.core.sudoku import Sudoku


class SudokuImageSolver:
    def __init__(self, image_path: str, model_type: Literal["BAD", "NORMAL", "GOOD"]) -> None:
        self.image_path = Path(image_path)
        self.model_type = self._convert_to_model_type(model_type)
        self.sudoku=None

    
    def get_sudoku_from_image(self) -> Sudoku:
        """
        Processes the input image to extract the Sudoku grid, fields, and predicted numbers, and creates a Sudoku instance.

        Parameters:
        - self: SudokuImageSolver, the instance of the SudokuImageSolver class containing the image

        Returns:
        - Sudoku - The Sudoku instance containing the predicted numbers extracted from the image.
        """
        # Create a SudokuImage instance with the provided image path and model type
        sudoku_image = SudokuImage(self.image_path, model_type=self.model_type)
        
        # Preprocess the image to get the warped version
        _ = sudoku_image.preprocess()
        
        # Extract the fields from the warped image
        _ = sudoku_image.extract_fields()
        
        # Extract the predicted numbers from the fields
        predicted_numbers = sudoku_image.extract_numbers()

        # Create a Sudoku instance with the predicted numbers
        self.sudoku = Sudoku(predicted_numbers)

        return self.sudoku
    
    def solve(self):
        """
        Solves the Sudoku puzzle contained in the self.sudoku attribute.

        Parameters:
        - self: SudokuImageSolver, the instance of the SudokuImageSolver class containing the Sudoku puzzle.

        Returns:
        - Sudoku - The solved Sudoku instance.
        """
        if self.sudoku is None:
            raise ValueError("No Sudoku puzzle found. Please run get_sudoku_from_image() first to extract the puzzle from the image.")
        
        solved = self.sudoku.solve()
        if not solved:
            raise ValueError("The Sudoku puzzle could not be solved.")

    def _convert_to_model_type(self, model_type: str) -> ModelType:
        """
        Converts a string representation of the model type to the corresponding ModelType enum.

        Parameters:
        - model_type: str - The string representation of the model type.

        Returns:
        - ModelType - The corresponding ModelType enum value.
        """
        try:
            return ModelType[model_type.upper()]
        except KeyError:
            raise ValueError(f"Invalid model type: {model_type}. Valid options are: {[model.name for model in ModelType]}")
