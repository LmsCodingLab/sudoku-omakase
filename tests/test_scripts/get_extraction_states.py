from sudoku_omakase import SudokuImage
import os


SUDOKU_PATHS = [
    "tests/test_images/IMG_0120.jpg",
    "tests/test_images/sudoku_easy.png",
    "tests/test_images/sudoku_hard.png",
    "tests/test_images/sudoku.png",
    "tests/test_images/sudoku1.png",
    "tests/test_images/test-sudoku.jpg",]

MODEL = "BIG"
def save_grid(i, sudoku: SudokuImage, solved: bool) -> None:
    if not solved:
        with open("pommer/got_boards.txt", 'a') as f:
            f.write(f"{i:04d}\n")
            f.write(f"{sudoku.board}\n")
    else:
        with open("pommer/solved_boards.txt", 'a') as f:
            f.write(f"{i:04d}\n")
            f.write(f"{sudoku.board}\n")




def get_states():
    for i, sp in enumerate(SUDOKU_PATHS):
        sudoku = SudokuImage(sp)
        save_grid(i, sudoku, solved=False)
        sudoku.solve()
        save_grid(i, sudoku, solved=True)

if __name__ == "__main__":
    get_states()