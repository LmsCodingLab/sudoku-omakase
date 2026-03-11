from pathlib import Path
import numpy as np
import numpy.typing as npt
from sudoku_omakase.model.models import ModelType 

Image = npt.NDArray[np.uint8]

class SudokuImage:
    """
    Represents a sudoku image and provides methods for preprocessing, extracting fields, and predicting numbers.
    """
    def __init__(self, source: Path, model_type: ModelType = ModelType.NORMAL) -> None:
        p = Path(source).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {source}")
        
        self.source = p
        self.model_type = model_type
        self.image = None
        self.size = 450
        self.corners = None
        self.warped = None
        self.fields = None
        self.numbers = None

    def preprocess(self) -> Image:
        from src.sudoku_omakase.vision.preprocess import prepare_image, detect_grid, warp_image
        """
        Preprocesses the input image by preparing it, detecting the grid, and warping it to a top-down view.

        Parameters:
        - self: SudokuImage, the instance of the SudokuImage class containing the source image.

        Returns:
        - np.ndarray, the preprocessed (warped) image of the sudoku grid.
        """
        self.image = prepare_image(self)
        self.corners = detect_grid(self.image)
        self.warped = warp_image(self.image, self.corners, self.size)

        return self.warped.copy()
    
    def extract_fields(self) -> list[Image]:
        from src.sudoku_omakase.vision.cell_extraction import extract_fields, resize_fields
        """
        Extracts the individual field images from the warped sudoku grid.

        Parameters:
        - self: SudokuImage, the instance of the SudokuImage class containing the warped image.

        Returns:
        - list of np.ndarray, the extracted field images.
        """
        fields = extract_fields(self)
        self.fields = resize_fields(fields)

        return [field.copy() for field in self.fields]
    
    def extract_numbers(self) -> npt.NDArray[np.int8]:
        from sudoku_omakase.vision.cell_extraction import extract_numbers
        """
        Extracts the predicted numbers from the extracted field images using the specified model.

        Parameters:
        - self: SudokuImage, the instance of the SudokuImage class containing the extracted field images.

        Returns:
        - np.ndarray, a 9x9 array representing the predicted numbers in the sudoku grid (0 for empty/uncertain fields).
        """
        if self.fields is None:
            raise RuntimeError("Fields must be extracted before predicting numbers. Call extract_fields() method first.")
        
        numbers = extract_numbers(self.fields, self.model_type)
        self.numbers = numbers

        return self.numbers.copy()
