import CD_SEM_tools as tools
import numpy as np
from scipy.fftpack import fftshift, fft2  # , ifft2

# from scipy.ndimage import gaussian_filter, morphology
from scipy.stats import scoreatpercentile

# from scipy.ndimage.measurements import label
# from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt


def lmax(height: int, pixel_scale: float, pixel_size: float) -> tuple[int, int, float]:
    """This function calculates and return imax, lmax, and kscale which are needed throughout later functions.

    Args:
        height (float): # Number of pixels in the vertical direction of the original image
        pixel_scale (float): multiplicative factor from recorded pixel size to um
        pixel_size (float): pixel size in recorded units

    Returns:
        imax (int): # Number of pixels
        lmax (int): size of the image for Fourier Filtering. Ideally it will be 40 pixels larger than imax to later be able to cut 20 pixels on each side to trim the edges containing issues from size-effects.
        kscale (float): Reciprocal space for lmax in inverse nanometers
    """
    imax = 2 ** np.floor(np.log2(height))
    lmax = (imax + 40) if height >= (imax + 40) else imax
    kscale = 2 * np.pi / (lmax * pixel_size * pixel_scale * 10**3)
    return imax, lmax, kscale


def ExtractCenterPart(img: np.ndarray, size: int) -> np.ndarray:
    """Extracts a square submatrix of size "size" from the center part of the larger matrix "img"
    roi_w & roi_h = region of interest

    Args:
        img (np.ndarray): input image as array
        size (int): size of submatrix

    Returns:
        np.arry: subarray defined by the indices
    """
    height, width = img.shape
    roi_h = int(np.maximum(np.floor((height - size) / 2), 0))
    roi_w = int(np.maximum(np.floor((width - size) / 2), 0))
    # roi = 0  when there is a data zone below zero
    return img[roi_h : int(roi_h + size), roi_w : int(roi_w + size)]


def fourier_img(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Calculates the magnitude squared of the Fourier Transform of a square image "img" of size "lmax".
    It recenters it so that the zero frequency is at {lmax/2+1, lmax/2+1}. It saves the magnitude square
    of the zero frequency in a variable called "ctrval". The zero frequncy in the image is replaced by "1" to help
    visualizing.

    Args:
        img (np.ndarray): Clipped and Rescaled image that need processed
        lmax (int): size of the image for Fourier Filtering

    Returns:
        tuple[np.ndarray, float]: FFT image, Magnitude square of the zero frequency
    """
    center = np.array(img.shape) // 2
    fimg = np.abs(fftshift(fft2(img))) ** 2
    ctrval = fimg[center, center]
    fimg[center, center] = 1
    return tools.rescale_array(np.log(fimg), 0, 1), ctrval


def clip_image(img: np.ndarray) -> np.ndarray:
    """Takes the input image and clips it between 0.05% and 99.95% of it origonal values.
    The clipped image is then rescaled between 0 and 1

    Args:
        img (np.ndarray): The 2D array that makes up the SEM image

    Returns:
        np.array: clipped and rescale image
    """
    xlow = scoreatpercentile(img, 0.05)
    xhigh = scoreatpercentile(img, 99.95)
    return tools.rescale_array(np.clip(img, xlow, xhigh), 0.0, 1.0)
