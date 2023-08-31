import tkinter as tk
import numpy as np
from skimage import filters
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


def rescale_array(
    arr: np.ndarray, new_min: float = 0, new_max: float = 1
) -> np.ndarray:
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


def blackwhite_image(img: np.ndarray, midlevel: float) -> np.ndarray:
    """Takes and image and turns it into an array of 0's and 1's based on the input threshold

    Args:
        img (np.ndarray): Flattend image that need binarized
        midlevel (float): The cut off for deciding if the pixel should be a 0 or 1

    Returns:
        np.ndarray: Binarized version of the input image
    """
    # Threshold the image using a midlevel value
    binary_image = (np.where(img >= midlevel, 1, 0)).astype(np.uint8)

    # Apply Gaussian filter to the binary image
    sigma = (8, 2)
    smoothed_image = filters.gaussian(
        binary_image, sigma=sigma, mode="constant", cval=0
    )
    smoothed_image = rescale_array(smoothed_image)
    # Threshold the smoothed image using another threshold value
    bw_image = (np.where(smoothed_image >= 0.5, 1, 0)).astype(np.uint8)

    return bw_image


def clip_image(img: np.ndarray) -> np.ndarray:
    """Takes the input image and clips it between 0.05% and 99.95% of it origonal values.
    The clipped image is then rescaled between 0 and 1

    Args:
        img (np.ndarray): The 2D array that makes up the SEM image

    Returns:
        np.ndarray: clipped and rescale image
    """
    xlow = scoreatpercentile(img, 0.05)
    xhigh = scoreatpercentile(img, 99.95)
    return rescale_array(np.clip(np.copy(img), xlow, xhigh))


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Takes the input image and rotates it by the given angle around the center pixel

    Args:
        image (np.ndarray): image that needs rotated
        angle (float): angle in radians of needed rotation

    Returns:
        np.ndarray: rotated image
    """
    # Perform the rotation around the center
    rotated_image = rotate(
        image, np.degrees(angle), reshape=False, order=1, mode="nearest"
    )

    return rotated_image


def pad_vector_to_match_length(
    vector: np.ndarray, target_vector: np.ndarray
) -> np.ndarray:
    """Takes a vector and evenly added zeros to either end until it is the same
        length as the 'taget_vector'

    Args:
        vector (np.ndarray): vector that needs padded
        target_vector (np.ndarray): longer vector with the target length

    Returns:
        np.ndarray: input vector padded with zeros
    """
    target_length = len(target_vector)
    current_length = len(vector)

    difference = target_length - current_length
    pad_before = difference // 2
    pad_after = difference - pad_before

    return np.pad(vector, (pad_before, pad_after), mode="constant", constant_values=0)


def simple_image_display(img: np.ndarray, title: str) -> None:
    """Simple display of an np.ndarray with a title

    Args:
        img (np.ndarray): Array to display
        title (str): Title of image
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.colorbar()
    plt.axis("off")
    plt.show()


def extract_repeating_dict_entries(
    dictionary: dict, start: int = 0, repeat: int = 2
) -> dict:
    """Extracts some repeated number at a given interval of dictionary entries

    Args:
        dictionary (dict): Full Dictionary
        start (int, optional): First entry to extract. Defaults to 0.
        repeat (int, optional): The interval to extract entries. Defaults to 2.

    Returns:
        dict: Dictionary with the extracted entries
    """
    keys = list(dictionary.keys())[start::repeat]
    values = [dictionary[key] for key in keys]
    extracted_entries = dict(zip(keys, values))
    return extracted_entries


def linear_fit(points: list) -> tuple[float, float]:
    """Calculates the slope and intercept of an input line of coordinates

    Args:
        points (list): All (row, column) points that make up the boundary edge line

    Returns:
        tuple[slope: float, intercept: float]: slope, interecpt of linear fit line
    """
    # Convert the points list to a NumPy array
    points_array = np.array(points)

    # Extract x and y coordinates from the points array
    x = points_array[:, 1]
    y = points_array[:, 0]

    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c


def list_barplot(list: list | np.ndarray) -> None:
    """Displays a barplot of the provided list

    Args:
        list (list): list or np.ndarray to plot
    """
    # Create a bar plot to visualize the column sums
    plt.figure(figsize=(15, 15))
    plt.bar(
        range(len(list)),
        list,
        tick_label=[f"Column {i+1}" for i in range(len(list))],
    )
    plt.xlabel("Columns")
    plt.ylabel("Sum")
    plt.title("Sum of Image Lines")
    plt.show()


def plot_peaks(peaks: list, scale: float) -> float:
    """Displays the values of the peaks 'list' against the index of the value shifted to start at 1.

    Args:
        peaks (list): Pixel location of the peak of the column sum funtion
        scale (float): size per pixel

    Returns:
        float: fitpitch to compare to previous values
    """
    indices = np.arange(1, len(peaks) + 1)  # Creating an array of indices
    slope, intercept = np.polyfit(
        indices, peaks, 1
    )  # Fit a first-degree polynomial (line)

    fitpitch = slope * scale

    # Create a plot
    plt.scatter(indices, peaks, label="List Values", color="gray")
    plt.plot(indices, slope * indices + intercept, label="Fit")
    plt.xlabel("Line No.")
    plt.ylabel("Centroid, px")
    plt.title(f"L0 = {round(fitpitch, 2)} nm, {len(peaks)} lines")
    plt.legend()
    plt.show()

    return fitpitch
