import tkinter as tk
import numpy as np


def open_window():
    """Creates a Tkinter root window"""
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    root.lift()
    root.focus_force()


def rescale_array(arr: np.ndarray, new_min: float, new_max: float) -> np.ndarray:
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
    scaled_arr = (arr - min_val) / (max_val - min_val)  # Scale values between 0 and 1
    rescaled_arr = (
        scaled_arr * (new_max - new_min) + new_min
    )  # Rescale to the new range
    return rescaled_arr


def threashold_mask(img: np.ndarray, threshold: float, new_value: float = 0) -> np.ndarray:
    """Sets all values in an array below the threashold to the new_value

    Args:
        img (np.ndarray): imgage array to have the mask applied to
        threshold (float): The lowest value being kept
        new_value (float): The value assigned to all pixels below threshold: Default zero

    Returns:
        np.ndarray: Output image array
    """
    new_img = np.copy(img)
    new_img[img < threshold] = new_value
    return new_img
