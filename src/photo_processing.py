import cv2
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike
from helpers.dev_info import dev_show_image, dev_show_message, dev_draw_image

SIZE = 450

# TODO1: Add more rigid controls over sudoku extraction (e.g. aspect ratio, size, etc.) Issue #15
def extract_sudoku(image_path: str, dev_mode: bool = False) -> MatLike:
    """
    Detects a sudoku grid in the given image.

    Parameters:
    - image_path: str, path to the input image.

    Returns:
    - np.ndarray, the extracted (warped) sudoku grid image.
    """

    # Load the image convert to grayscale and blur it
    grey_image = cv2.imread(filename=image_path, flags=0) 
    if grey_image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}. File may be missing or have an unsupported format.")
    dev_show_image(dev_mode, "Greyscale Image", grey_image)
    
    smooth = cv2.GaussianBlur(src=grey_image, ksize=(7, 7), sigmaX=0) 
    dev_show_image(dev_mode, "Smoothed Image", smooth)


    # Apply adaptive thresholding to get a binary image
    thresh = cv2.adaptiveThreshold(
        src=smooth, 
        maxValue=255, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY_INV, 
        blockSize=13, 
        C=8
    )
    dev_show_image(dev_mode, "Thresholded Image", thresh)


    # Connect broken grid lines / border so contours become closed
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    dev_show_image(dev_mode, "Closed (morphology)", closed)


    # Find contours in the edged image
    contours, _ = cv2.findContours(image=closed, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    dev_show_message(dev_mode, f"Found {len(contours)} contours")
    dev_draw_image(dev_mode, "Contours", grey_image, contours, color=(0, 0, 255))


    # Find the largest contour which should be the sudoku grid
    # ! Potential bug here if the largest contour is not the sudoku grid (it will take alot of time to fix so leaving as is for now)
    sudoku = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        perimeter = cv2.arcLength(curve=contour, closed=True)
        approx = cv2.approxPolyDP(curve=contour, epsilon=0.02 * perimeter, closed=True) # Douglas-Peucker algorithm

        # approximated contour has 4 points, we can assume we found the sudoku grid
        if len(approx) == 4:
            sudoku = approx
            dev_draw_image(dev_mode, "Sudoku Contour", grey_image, [sudoku], color=(0, 255, 0))
            break
    if sudoku is None:
        raise RuntimeError("Couldn't find sudoku grid in the image")
    

    # Get the edge points and apply perspective transform
    pts1 = _order_points(sudoku)
    pts2 = np.array([[0, 0], [SIZE -1, 0], [0, SIZE -1], [SIZE -1, SIZE -1]], dtype=np.float32) # square shape

    dev_show_message(dev_mode, f"Source Points: {pts1}")
    dev_show_message(dev_mode, f"Destination Points: {pts2}")

    matrix = cv2.getPerspectiveTransform(src=pts1, dst=pts2)
    warped = cv2.warpPerspective(src=thresh, M=matrix, dsize=(SIZE, SIZE))

    dev_show_image(dev_mode, "Warped Sudoku", warped)

    return warped

def _order_points(pts:  MatLike) -> npt.NDArray[np.float32]:
    """
    Orders points in the order: top-left, top-right, bottom-left, bottom-right, with coordinates starting at the top-left corner.

    Parameters:
    - pts: np.ndarray, array of points to be ordered.

    Returns:
    - np.ndarray, ordered points.
    """
    pts = pts.reshape(4, 2)
    ordered_pts = np.array([
        pts[np.argmin(pts.sum(axis=1))], # top-left (smallest sum of coordinates)
        pts[np.argmin(np.diff(pts, axis=1))], # top-right (smallest difference)
        pts[np.argmax(np.diff(pts, axis=1))], # bottom-left (largest difference)
        pts[np.argmax(pts.sum(axis=1))] # bottom-right (largest sum of coordinates)
    ], dtype=np.float32)

    return ordered_pts 

if __name__ == "__main__":
    extract_sudoku("test/sudoku_hard.png", dev_mode=True)
