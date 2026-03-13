This project is a python package for solving sudokus from images. It uses computer vision to extract the sudoku grid and digits, and then uses a backtracking algorithm to solve the sudoku.

## 1. Installation

> **PyTorch note (CPU vs CUDA):**
> If you have an NVIDIA GPU and want CUDA acceleration, install a PyTorch build that matches your CUDA version.
> Use the official selector to get the correct command for your system:
> https://pytorch.org/get-started/locally/
>
> CPU-only users can typically install the default wheels.
```sh
pip install torch torchvision
```

```sh
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sudoku-omakase==0.1.0
```
!DANGER Packages on testPyPI are temporary, it maybe already deleted. Please contact if so.

## 2. Quickstart

```python
from sudoku_omakase import SudokuImage

sudoku = SudokuImage(source="path/to/sudoku/image.jpg", model_type="BIG")
print("Original:")
print(sudoku)
sudoku.solve()
print("Solved:")
print(sudoku)
```

Or if you already have the grid and just want to solve it:

```python
from sudoku_omakase import Sudoku

grid = np.array([
        [0, 0, 0, 2, 0, 0, 0, 6, 3],
        [3, 0, 0, 0, 0, 5, 4, 0, 1],
        [0, 0, 1, 0, 0, 3, 9, 8, 0],
        [0, 0, 0, 0, 0, 0, 0, 9, 0],
        [0, 0, 0, 5, 3, 8, 0, 0, 0],
        [0, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 6, 3, 0, 0, 5, 0, 0],
        [5, 0, 3, 7, 0, 0, 0, 0, 8],
        [4, 7, 0, 0, 0, 1, 0, 0, 0]
    ], dtype=np.int8)

sudoku = Sudoku(grid)
print("Original:")
print(sudoku)
sudoku.solve()
print("Solved:")
print(sudoku)
```

## 3. Usage

1. **Bild vorbereiten:** Sorge für ein kontrastreiches, rechteckiges Foto der Sudoku-Zeitungsseite. Entferne Randbereiche oder rotiere die Datei so, dass das Gitter nahezu frontal sichtbar ist; unscharfe Kanten führen sonst zu Fehlern.
2. **`SudokuImage` erzeugen:** Lade das Bild über `SudokuImage(source=..., model_type=...)`. `model_type` wählt den OCR-Backbone (`BAD`, `NORMAL`, `BIG` – langsam aber am genauesten). 
3. **`solve()` aufrufen:** `sudoku.solve()` führt die Zell-Klassifikation und den Backtracking-Solver aus. Falls keine Lösung gefunden wird, prüfe zuerst das Log – meist liegt es an falsch erkannten Ziffern oder daran, dass das Raster nicht sauber extrahiert werden konnte.
4. **Ergebnis interpretieren:** Über `print(sudoku)` oder `sudoku.board` erhältst du das vollständig gelöste 9x9-Array. 

### Häufig genutzte Parameter

- `model_type`: `"SMALL"` (schnell, genügt für klare Scans), `"NORMAL"` (Standard), `"BIG"` (robust bei schlechten Fotos).
- `source`: Dateipfad (String); stelle sicher, dass Pfade relativ zum Arbeitsverzeichnis existieren.


## Models
Currently, there are 3 pretrained Models for sudoku recognition.

- `"SMALL"` (179 KB)
- `"NORMAL"` (42 MB)
- `"BIG"` (332 MB)


