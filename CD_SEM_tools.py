import tkinter as tk
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile
from scipy.ndimage import rotate


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
    """Takes the input image and rotates it by the given angle around the center pixel

    Args:
        image (np.array): image that needs rotated
        angle (float): angle in radians of needed rotation

    Returns:
        np.array: rotated image
    """
    # Perform the rotation around the center
    rotated_image = rotate(
        image, np.degrees(angle), reshape=False, order=1, mode="nearest"
    )

    return rotated_image


def pad_vector_to_match_length(vector: np.array, target_vector: np.array) -> np.array:
    """Takes a vector and evenly added zeros to either end until it is the same
        length as the 'taget_vector'

    Args:
        vector (np.array): vector that needs padded
        target_vector (np.array): longer vector with the target length

    Returns:
        np.array: input vector padded with zeros
    """
    target_length = len(target_vector)
    current_length = len(vector)

    difference = target_length - current_length
    pad_before = difference // 2
    pad_after = difference - pad_before

    return np.pad(vector, (pad_before, pad_after), mode="constant", constant_values=0)


def simple_image_display(img: np.array, title: str) -> None:
    """Simple display of an np.array with a title

    Args:
        img (np.array): Array to display
        title (str): Title of image
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.colorbar()
    plt.axis("off")
    plt.show()
