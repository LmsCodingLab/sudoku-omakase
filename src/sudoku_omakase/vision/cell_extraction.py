import cv2
import numpy as np
import numpy.typing as npt

from sudoku_omakase.model.predictor import guess_num 
from sudoku_omakase.model.models import ModelType

Image = npt.NDArray[np.uint8]
SIZE = 450

def extract_fields(sudoku_image: Image) -> list[Image]:
    """
    Extracts the 81 individual fields from the warped sudoku grid image.

    Parameters:
    - sudoku_image: SudokuImage, the warped sudoku grid image.

    Returns:
    - list of np.ndarray, the extracted field images.
    """
    if sudoku_image is None:
        raise RuntimeError("Image must be preprocessed before extracting fields. Call preprocess() method first.")
    if sudoku_image.shape[0] != SIZE or sudoku_image.shape[1] != SIZE:
        raise ValueError(f"Invalid image size: {sudoku_image.shape}. Expected ({SIZE}, {SIZE}). Ensure the image is warped to the correct size before extracting fields.")

    fields = []
    field_size = SIZE // 9
    for row in range(9):
        for col in range(9):
            x_start, y_start = col * field_size, row * field_size
            field = sudoku_image[y_start:y_start + field_size, x_start:x_start + field_size]
            fields.append(field)

    return fields

def resize_fields(fields: list[Image]) -> list[Image]:
    """
        Resizes the extracted field images to 32x32 pixels, which is the input size expected by the models.

        Parameters:
        - fields: list of np.ndarray, the extracted field images.

        Returns:
        - list of np.ndarray, the resized field images.
    """
    if not fields:
        return []
    
    CROP_MARGIN = 5
    resized_fields = []
    for field in fields:
        cropped_field = field[CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN] # Crop the margin from each side to remove grid lines
        resized_field = cv2.resize(cropped_field, (32, 32), interpolation=cv2.INTER_AREA)
        resized_fields.append(resized_field)

    return resized_fields

def extract_numbers(sudoku: list[Image], model_type: ModelType) -> npt.NDArray[np.int8]:
    """
        Uses the specified model to predict the number in each field of the sudoku grid.

        Parameters:
        - sudoku: list of np.ndarray, the list of field images.
        - model_type: ModelType, the type of model to use for prediction.

        Returns:
        - np.ndarray, a 9x9 array representing the predicted numbers in the sudoku grid (0 for empty/uncertain fields).
    """
    if len(sudoku) != 81:
        raise ValueError(f"Invalid number of fields: {len(sudoku)}. Expected 81 fields for a 9x9 sudoku grid.")
    
    numbers = []
    batch = []
    for field in sudoku:
        num = guess_num(data=field, model_type=model_type)
        batch.append(num)
        if len(batch) == 9:
            numbers.append(batch)
            batch = []
    result = np.array(numbers, dtype=np.int8)

    return result