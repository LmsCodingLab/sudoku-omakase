from sudoku_omakase.pipeline import SudokuGridSolver, SudokuImageSolver
from sudoku_omakase.vision.sudoku_image import SudokuImage
from sudoku_omakase.core.sudoku import Sudoku
from sudoku_omakase.model.models import ModelType
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

    sudoku = Sudoku(predicted_numbers)
    print(sudoku)
    

def test_pipelines():
    source = "tests/test_images/IMG_0120.jpg"
    solver = SudokuImageSolver(source, model_type="BAD")
    solver.get_sudoku_from_image()
    sudoku = solver.get_sudoku()
    gird = sudoku.board.copy()
    print(sudoku)
    solver.solve()
    print(sudoku)

    grid_solver = SudokuGridSolver(gird)
    print(grid_solver.sudoku)
    grid_solver.solve()
    print(grid_solver.sudoku)


if "__main__" == __name__:
    test_pipelines()