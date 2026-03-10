from src.sudoku_omakase.Sudoku import Sudoku

sudoku = Sudoku()
sudoku.from_photo("tests/test_images/IMG_0120.jpg", model_type="resnext")
sudoku.solve()
print(sudoku)