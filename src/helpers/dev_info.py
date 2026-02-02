import cv2
from typing import Sequence
import numpy as np

def dev_show_image(dev_mode: bool, title: str, image: np.ndarray) -> None:
    """
    Displays an image in a window if dev_mode is enabled.

    Parameters:
    - dev_mode: bool, flag to enable/disable developer mode.
    - title: str, title of the window.
    - image: np.ndarray, image to be displayed.
    """
    if dev_mode:
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def dev_draw_image(dev_mode: bool, title: str, image: np.ndarray, contours: Sequence[np.ndarray], color: tuple[int, int, int], thickness: int = 2) -> None:
    """
    Draws contours on an image and displays it in a window if dev_mode is enabled.

    Parameters:
    - dev_mode: bool, flag to enable/disable developer mode.
    - title: str, title of the window.
    - image: np.ndarray, image on which contours will be drawn.
    - contours: list of np.ndarray, contours to be drawn.
    - color: tuple of int, color of the contours in BGR format.
    - thickness: int, thickness of the contour lines.
    """
    if dev_mode:
        if len(image.shape) == 2:
          tmp = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            tmp = image.copy()
        cv2.drawContours(image=tmp, contours=contours, contourIdx=-1, color=color, thickness=thickness)
        cv2.imshow(title, tmp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def dev_show_message(dev_mode: bool, message: str) -> None:
    """
    Prints a message to the console if dev_mode is enabled.

    Parameters:
    - dev_mode: bool, flag to enable/disable developer mode.
    - message: str, message to be printed.
    """
    if dev_mode:
        print(message)