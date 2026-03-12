from sudoku_omakase.core.sudoku_image import SudokuImage

def test_sudoku_image():
    image_path = "tests/test_images/IMG_0120.jpg"
    sudoku_image = SudokuImage(image_path, "NORMAL")
    print(sudoku_image)
    sudoku_image.solve()
    print(sudoku_image)
    



if "__main__" == __name__:
    test_sudoku_image()