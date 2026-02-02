import cv2
import numpy as np
from typing import Annotated

SIZE = 450

# TODO1: Add more rigid controls over sudoku extraction (e.g. aspect ratio, size, etc.) Issue #15
# TODO2: Abstract out the dev mode code into a decorator to reduce clutter. Issue #14
def extract_sudoku(image_path: str, dev_mode: bool = False) -> Annotated[np.ndarray, (SIZE, SIZE), np.uint8]:
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
    
    smooth = cv2.GaussianBlur(src=grey_image, ksize=(5, 5), sigmaX=0) 
    if dev_mode:
        cv2.imshow("Smoothed Image", smooth)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Apply adaptive thresholding to get a binary image
    thresh = cv2.adaptiveThreshold(
        src=smooth, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=13, C=4
    )
    if dev_mode:
        cv2.imshow("Thresholded Image", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Apply edge detection
    edges = cv2.Canny(image=thresh, threshold1=50, threshold2=150, apertureSize=5, L2gradient=True)
    if dev_mode:
        cv2.imshow("Edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Find contours in the edged image
    contours, _ = cv2.findContours(image=edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    if dev_mode:
        print(f"Found {len(contours)} contours")
        tmp = cv2.cvtColor(grey_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image=tmp, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=2)
        cv2.imshow("Contours", tmp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Find the largest contour which should be the sudoku grid
    # ! Potential bug here if the largest contour is not the sudoku grid (it will take alot of time to fix so leaving as is for now)
    sudoku = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        perimeter = cv2.arcLength(curve=contour, closed=True)
        approx = cv2.approxPolyDP(curve=contour, epsilon=0.02 * perimeter, closed=True) # Douglas-Peucker algorithm

        # approximated contour has 4 points, we can assume we found the sudoku grid
        if len(approx) == 4:
            sudoku = approx
            if dev_mode:
                tmp = cv2.cvtColor(grey_image, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(image=tmp, contours=[sudoku], contourIdx=-1, color=(0, 255, 0), thickness=2)
                cv2.imshow("Sudoku Contour", tmp)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            break
    if sudoku is None:
        raise RuntimeError("Couldn't find sudoku grid in the image")
    

    # Get the edge points and apply perspective transform
    pts1 = _order_points(sudoku)
    pts2 = np.array([[0, 0], [SIZE, 0], [0, SIZE], [SIZE, SIZE]], dtype=np.float32) # square shape

    if dev_mode:
        print(f"Sudoku Points: {pts1}")

    matrix = cv2.getPerspectiveTransform(src=pts1, dst=pts2)
    warped = cv2.warpPerspective(src=grey_image, M=matrix, dsize=(SIZE, SIZE))

    if dev_mode:
        cv2.imshow("Warped Sudoku", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return warped

def _order_points(pts: Annotated[np.ndarray, np.float32]) -> Annotated[np.ndarray, (4,2), np.float32]:
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
    extract_sudoku("test/test-sudoku.jpg", dev_mode=True)