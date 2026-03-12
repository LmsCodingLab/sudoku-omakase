import cv2
import numpy as np
import numpy.typing as npt
from typing import cast

Image = npt.NDArray[np.uint8]

def prepare_image(source: str) -> Image:
	"""
		Reads the input image, converts it to grayscale, applies a Gaussian blur, and then applies adaptive thresholding to prepare it for contour detection.

		Parameters:
		- source: str, the path to the input image.

		Returns:
		- np.ndarray, the processed image.
	"""

	# Load the image convert to grayscale and blur it
	image = cv2.imread(source, flags=0)
	if image is None:
		raise FileNotFoundError(f"Image not found at path: {source}")

	image = cv2.GaussianBlur(src=image, ksize=(7, 7), sigmaX=0)

	# Apply adaptive thresholding to get a binary image
	image = cv2.adaptiveThreshold(
		src=image, 
		maxValue=255, 
		adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
		thresholdType=cv2.THRESH_BINARY_INV, 
		blockSize=13, 
		C=8
	)

	return cast(Image, image)

def detect_grid(image: Image) -> npt.NDArray[np.float32]:
	"""
		Detects the largest contour in the image, which is assumed to be the sudoku grid, and returns its corners.

		Parameters:
		- image: np.ndarray, the processed image from which to detect the grid.

		Returns:
		- np.ndarray, the corners of the detected grid.
	"""

	# Connect broken grid lines / border so contours become closed
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	closed_image = cast(Image, cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2))

	# Find contours in the edged image
	contours, _ = cv2.findContours(image=closed_image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

	# Find the largest contour which should be the sudoku grid
	# ! Potential bug here if the largest contour is not the sudoku grid (it will take alot of time to fix so leaving as is for now)
	for contour in sorted(contours, key=cv2.contourArea, reverse=True):
		peri = cv2.arcLength(curve=contour, closed=True)
		corners = cv2.approxPolyDP(curve=contour, epsilon=0.02 * peri, closed=True)

		if len(corners) == 4:
			return corners.reshape(4, 2).astype(np.float32)

	raise ValueError("No valid grid found in the image.")

def warp_image(image: Image, corners: npt.NDArray[np.float32], size: int = 450) -> Image:
	"""
		Warps the input image to a top-down view of the sudoku grid based on the detected corners.

		Parameters:
		- image: np.ndarray, the original input image.
		- corners: np.ndarray, the corners of the detected grid.
		- size: int, the desired size of the output warped image (default is 450x450).

		Returns:
		- np.ndarray, the warped image of the sudoku grid.
	"""

	# Define the destination points for the perspective transform
	corners = _order_corners(corners)
	dest_corners = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)

	# Compute the perspective transform matrix and apply it
	transform_matrix = cv2.getPerspectiveTransform(src=corners, dst=dest_corners)
	warped_image = cv2.warpPerspective(src=image, M=transform_matrix, dsize=(size, size))

	return cast(Image, warped_image)

def _order_corners(corners: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
	"""
		Orders the corners in the order: top-left, top-right, bottom-right, bottom-left.

		Parameters:
		- corners: np.ndarray, the corners to be ordered.

		Returns:
		- np.ndarray, the ordered corners.
	"""

	sorted_corners = np.zeros_like(corners)
	sorted_corners[0] = corners[np.argmin(corners.sum(axis=1))]  # Top-left has the smallest sum
	sorted_corners[2] = corners[np.argmax(corners.sum(axis=1))]  # Bottom-right has the largest sum
	sorted_corners[1] = corners[np.argmin(np.diff(corners, axis=1))]  # Top-right has the smallest difference
	sorted_corners[3] = corners[np.argmax(np.diff(corners, axis=1))]  # Bottom-left has the largest difference

	return sorted_corners