import tkinter as tk
import numpy as np
from skimage import transform
from scipy.stats import scoreatpercentile


def open_window():
    """Creates a Tkinter root window"""
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    root.lift()
    root.focus_force()


def rescale_array(arr: np.array, new_min: float, new_max: float) -> np.array:
    """Rescales the input array between the new_min and new_max values provided

    Args:
        arr (np.ndarray): Array to be rescaled
        new_min (float): smallest value in the new array
        new_max (float): largest value in the new array

    Returns:
        np.ndarray: rescaled version of the input array
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    scaled_arr = (np.copy(arr) - min_val) / (
        max_val - min_val
    )  # Scale values between new_min and new_max
    rescaled_arr = (
        scaled_arr * (new_max - new_min) + new_min
    )  # Rescale to the new range
    return rescaled_arr


def clip_image(img: np.array) -> np.array:
    """Takes the input image and clips it between 0.05% and 99.95% of it origonal values.
    The clipped image is then rescaled between 0 and 1

    Args:
        img (np.ndarray): The 2D array that makes up the SEM image

    Returns:
        np.array: clipped and rescale image
    """
    xlow = scoreatpercentile(img, 0.05)
    xhigh = scoreatpercentile(img, 99.95)
    return rescale_array(np.clip(np.copy(img), xlow, xhigh), 0, 1)


def rotate_image(image: np.array, angle: float) -> np.array:
    """Takes the input image and rotates it by the given angle

    Args:
        image (np.array): 2D PSD image
        angle (float): angle of needed rotation given by def rotate_angle()

    Returns:
        np.array: rotated 2D PSD image
    """
    # Define the transformation function for rotation
    transform_matrix = transform.AffineTransform(rotation=np.deg2rad(angle))

    # Perform the image transformation
    transformed_image = transform.warp(np.copy(image), inverse_map=transform_matrix)

    return transformed_image
