from pathlib import Path
from sudoku_omakase.model.models import ModelType
from sudoku_omakase.vision.sudoku_image import SudokuImage
from sudoku_omakase.core.sudoku import Sudoku

class SudokuImageSolver:
    def __init__(self, image_path: str, model_type: str) -> None:
        """
        Initializes the SudokuImageSolver with the path to the image and the model type.
        Parameters:
        - image_path: str - The file path to the Sudoku image.
        - model_type: str - The type of model to use for solving (e.g., "BAD", "NORMAL", "GOOD").
        """
        self.image_path = Path(image_path)
        self.model_type = self._convert_to_model_type(model_type)
        self.sudoku=None

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
    
    def get_sudoku_from_image(self):
        # Create a SudokuImage instance with the provided image path and model type
        sudoku_image = SudokuImage(self.image_path, model_type=self.model_type)
        
        # Preprocess the image to get the warped version
        warped_image = sudoku_image.preprocess()
        
        # Extract the fields from the warped image
        fields = sudoku_image.extract_fields()
        
        # Extract the predicted numbers from the fields
        predicted_numbers = sudoku_image.extract_numbers()

        # Create a Sudoku instance with the predicted numbers
        self.sudoku = Sudoku(predicted_numbers)

    def get_sudoku(self) -> Sudoku:
        """
        Returns the Sudoku instance created from the image.

        Returns:
        - Sudoku - The Sudoku instance containing the predicted numbers.
        """
        if self.sudoku is None:
            raise ValueError("Sudoku has not been generated yet. Call get_sudoku_from_image() first.")
        return self.sudoku
    
    def solve(self) -> None:
        """
        Solves the Sudoku puzzle using a backtracking algorithm.

        Parameters:
        - self: SudokuImageSolver, the instance of the SudokuImageSolver class containing the Sudoku
        """
        if self.sudoku is None:
            raise ValueError("Sudoku has not been generated yet. Call get_sudoku_from_image() first.")
        
        self.sudoku.solve()