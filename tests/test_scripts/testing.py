from src.sudoku_omakase.vision.sudoku_image import SudokuImage
from src.sudoku_omakase.model.models import ModelType
from pathlib import Path

def test_guess_num():
    source = Path("tests/test_images/IMG_0120.jpg")
    # Create a SudokuImage instance with a sample image and model type
    sudoku_image = SudokuImage(source, model_type=ModelType.BAD)
    
    # Preprocess the image to get the warped version
    warped_image = sudoku_image.preprocess()
    
    # Extract the fields from the warped image
    fields = sudoku_image.extract_fields()
    
    # Extract the predicted numbers from the fields
    predicted_numbers = sudoku_image.extract_numbers()
    
    # Print the predicted numbers for verification
    print(predicted_numbers)


if "__main__" == __name__:
    test_guess_num()