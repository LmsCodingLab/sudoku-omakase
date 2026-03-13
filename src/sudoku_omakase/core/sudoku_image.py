from pathlib import Path
import numpy as np
import numpy.typing as npt
from typing import Final, Literal

from sudoku_omakase.model.models import ModelType 
from sudoku_omakase.core.sudoku import Sudoku
from sudoku_omakase.vision.preprocess import prepare_image, detect_grid, warp_image
from sudoku_omakase.vision.cell_extraction import extract_fields, resize_fields, extract_numbers

Image = npt.NDArray[np.uint8]

class SudokuImage(Sudoku):
    """
    Represents a sudoku image and provides methods for preprocessing, extracting fields, and predicting numbers.
    """
    def __init__(self, source: str, model_type: Literal["BAD", "NORMAL", "BIG"] = "NORMAL") -> None:
        p = Path(source).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {source}")
        
        self.source = p
        self.model_type = ModelType[model_type]

        board = self.get_sudoku_from_image()
        super().__init__(board)
        
        self.image: Image | None = None
        self.corners: npt.NDArray[np.float32] | None = None
        self.warped: Image | None = None
        self.fields: list[Image] | None = None
        self.numbers: npt.NDArray[np.int8] | None = None

    def get_sudoku_from_image(self) -> npt.NDArray[np.int8]:
        """
        Processes the input image to extract the sudoku grid as a 9x9 array of integers.

        Parameters:
        - self: SudokuImage, the instance of the SudokuImage class containing the source image.

        Returns:
        - np.ndarray, a 9x9 array representing the predicted numbers in the sudoku grid (0 for empty/uncertain fields).
        """
        self._preprocess()
        self._get_fields()
        return self._get_numbers()
    
    def get_extracted_sudoku_image(self) -> Image:
        """
        Returns the preprocessed (warped) image of the sudoku grid.

        Parameters:
        - self: SudokuImage, the instance of the SudokuImage class containing the warped image.

        Returns:
        - np.ndarray, the preprocessed (warped) image of the sudoku grid.
        """
        if self.warped is None:
            raise RuntimeError("Image must be preprocessed before accessing the warped image. Call _preprocess() method first.")
        
        return self.warped.copy()

    def _preprocess(self) -> Image:
        """
        Preprocesses the input image by preparing it, detecting the grid, and warping it to a top-down view.

        Parameters:
        - self: SudokuImage, the instance of the SudokuImage class containing the source image.

        Returns:
        - np.ndarray, the preprocessed (warped) image of the sudoku grid.
        """
        
        self.image = prepare_image(str(self.source))
        self.corners = detect_grid(self.image)
        self.warped = warp_image(self.image, self.corners)

        return self.warped.copy()
    
    def _get_fields(self) -> list[Image]:
        """
        Extracts the individual field images from the warped sudoku grid.

        Parameters:
        - self: SudokuImage, the instance of the SudokuImage class containing the warped image.

        Returns:
        - list of np.ndarray, the extracted field images.
        """
        if self.warped is None:
            raise RuntimeError("Image must be preprocessed before extracting fields. Call _preprocess() method first.")
        
        fields = extract_fields(self.warped)
        self.fields = resize_fields(fields)

        return [field.copy() for field in self.fields]
    
    def _get_numbers(self) -> npt.NDArray[np.int8]:
        """
        Extracts the predicted numbers from the extracted field images using the specified model.

        Parameters:
        - self: SudokuImage, the instance of the SudokuImage class containing the extracted field images.

        Returns:
        - np.ndarray, a 9x9 array representing the predicted numbers in the sudoku grid (0 for empty/uncertain fields).
        """
        if self.fields is None:
            raise RuntimeError("Fields must be extracted before predicting numbers. Call _get_fields() method first.")
        
        numbers = extract_numbers(self.fields, self.model_type)
        self.numbers = numbers

        return self.numbers.copy()
