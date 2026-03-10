from src.Sudoku import Sudoku

sudoku = Sudoku()
sudoku.from_photo("src/test/IMG_0120.jpg")
sudoku.solve()
print(sudoku)