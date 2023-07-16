import CD_SEM_tools as tools
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, fft2  # , ifft2
from scipy.ndimage import median_filter
from skimage.draw import line

# from scipy.ndimage import gaussian_filter, morphology
from scipy.stats import scoreatpercentile

# from scipy.ndimage.measurements import label
# from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt


def image_size(
    height: int, pixel_scale: float, pixel_size: float
) -> tuple[int, int, float]:
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
    imax = int(2 ** np.floor(np.log2(height)))
    lmax = (imax + 40) if height >= (imax + 40) else imax
    kscale = 2 * np.pi / (lmax * pixel_size * pixel_scale * 10**3)
    return imax, (lmax-1)|1, kscale # Guarantees lmax is ODD insuring a pixel at the very center of the image


def extract_center_part(img: np.ndarray, size: int) -> np.ndarray:
    """Extracts a square submatrix of size "size" from the center part of the larger matrix "img"
    roi_w & roi_h = region of interest

    Args:
        img (np.ndarray): input image as array
        size (int): size of submatrix

    Returns:
        np.arry: subarray defined by the indices
    """
    height, width = img.shape
    dimension = min(height, width)
    roi_h = int(max(0,np.floor((dimension - size) / 2)))
    roi_w = int(max(0,np.floor((dimension - size) / 2)))
    # roi = 0  when there is a data zone below zero
    return img[roi_h : int(roi_h + size), roi_w : int(roi_w + size)]


def fourier_img(img: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Calculates the magnitude squared of the Fourier Transform of a square image "img" of size "lmax".
    It recenters it so that the zero frequency is at {lmax/2+1, lmax/2+1}. It saves the magnitude square
    of the zero frequency in a variable called "ctrval". The zero frequncy in the image is replaced by "1" to help
    visualizing.

    Args:
        img (np.ndarray): Clipped and Rescaled image that need processed

    Returns:
        tuple[np.ndarray, tuple]: FFT image, Magnitude square of the zero frequency
    """
    fimg = np.abs(fftshift(fft2(np.copy(img)))) ** 2
    center = np.array(fimg.shape) // 2
    rescale_values = (fimg[center, center], np.min(fimg), np.max(fimg))
    fimg = tools.rescale_array(np.log(fimg), 0, 1)
    return fimg, rescale_values


def rotated_angle(angle: float, img: np.ndarray, lmax: int) -> float:
    """Integrates the I(qx, qy = 0) line (the horizontal center line).
    It also probes different rotation values by rotating the image one pixel at a time up to "probe" pixels.
    It then plots the integral values vs rotated amount. It picks the maximum value to be the one that indicates the "true" horizontal position.
    It returns the rotated angle in degrees. "probe needs to be an ODD number"

    Args:
        probe (int): Maximum number of pixels to rotate by
        img (np.ndarray): FFT image being analyized
        lmax (int): Size of FFT image

    Returns:
        float: angle the image needs rotated
    """
    def line_sum(a, img):
        dy = int(np.ceil((lmax/2)*np.arctan(np.radians(a))))
        rr, cc = line(0, int(lmax//2 - dy), lmax-1, int(lmax//2 + dy))
        return np.sum(img[rr, cc])

    probe = int(np.ceil((lmax/2)*np.arctan(np.radians(angle))))
    totals = [line_sum(n, img) for n in range(-angle, angle + 1)]
    totals2 = median_filter(totals, size=3, mode='mirror')

    max_index = np.argmax(totals2)
    ra =0
    # if totals2[max_index] / totals[probe] > 1.05:
    #     ra = np.arctan((max_index - (probe - 3 + 1)) / (lmax / 2 - 1)) * 180 / np.pi
    # else:
    #     ra = 0
    print("Angle of rotation:", ra)

    angle_range = np.linspace(-probe, probe, len(totals))
    angle_range_plusminus3 = np.linspace(-probe + 3, probe - 3, len(totals2))

    plt.plot(angle_range, totals, marker="o", linestyle="-", label="Totals")
    plt.plot(angle_range_plusminus3, totals2, label="Moving Median")

    plt.xlabel("angle (deg)")
    plt.ylabel("Intensity")
    plt.title("Finding Image Rotation")
    plt.legend()
    plt.axvline(
        x=ra, color="red", linestyle="--", label="RA"
    )  # Add vertical line at ra

    plt.show()

    return ra


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
