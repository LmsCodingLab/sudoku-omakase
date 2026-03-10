from src.Sudoku import Sudoku

sudoku = Sudoku()
sudoku.extract_grid("src/test/IMG_0120.jpg")
sudoku.solve()
print(sudoku)