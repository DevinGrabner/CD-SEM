import CD_SEM_tools as tools
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.fftpack import fftshift, fft2, ifft2, ifftshift


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
    kscale = 2 * np.pi / (lmax * pixel_size * pixel_scale)
    return imax, lmax, kscale


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
    roi_h = int(np.maximum(np.floor((height - size) / 2), 0))
    roi_w = int(np.maximum(np.floor((width - size) / 2), 0))
    # roi = 0  when there is a data zone below zero
    return img[roi_h : int(roi_h + size), roi_w : int(roi_w + size)]


def PDS_img(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Calculates the magnitude squared of the Fourier Transform of a square image "img" of size "lmax".
    It recenters it so that the zero frequency is at {lmax/2+1, lmax/2+1}. It saves the magnitude square
    of the zero frequency in a variable called "ctrval". The zero frequncy in the image is replaced by "1" to help
    visualizing.

    Args:
        img (np.ndarray): Clipped and Rescaled image that need processed

    Returns:
        tuple[np.ndarray, float]: FFT image, Magnitude square of the zero frequency
    """
    fimg = fftshift((np.abs(fft2(img))) ** 2)
    center = np.array(fimg.shape) // 2
    ctrval = fimg[center, center]
    fimg = np.log(fimg)
    fimg = tools.rescale_array(fimg)
    fimg[center, center] = 1
    return fimg, ctrval


def rotated_angle(probe: int, img: np.ndarray, lmax: int) -> float:
    """_summary_

    Args:
        probe (int): Maximum number of pixels to rotate by
        img (np.ndarray): FFT image being analyized
        lmax (int): Size of FFT image

    Returns:
        float: angle the image needs rotated
    """

    def calculatetotals(probe: int, img: np.ndarray, lmax: int) -> np.ndarray:
        totals = []
        for l in range(-probe, probe + 1):
            indices = np.array(
                [
                    (round((j - lmax / 2 - 1) * (l / (lmax / 2 - 1)) + lmax / 2 + 1), j)
                    for j in range(lmax)
                ]
            )
            values = img[indices[:, 0], indices[:, 1]]
            total = np.sum(values)
            totals.append(total)
        totals = np.array(totals)
        return totals

    def movingmedian(data: np.ndarray, window_size: int) -> float:
        padded_data = np.pad(data, (window_size - 1) // 2, mode="edge")
        windowed_data = np.lib.stride_tricks.sliding_window_view(
            padded_data, window_size
        )
        medians = np.apply_along_axis(lambda x: np.median(x), 1, windowed_data)
        return medians

    totals = calculatetotals(probe, np.copy(img), lmax)
    totals2 = movingmedian(totals, 7)

    max_index = np.argmax(totals2)
    if totals2[max_index] / totals[probe] > 1.05:
        ra = np.arctan((max_index - (probe - 3 + 1)) / (lmax / 2 - 1)) * 180 / np.pi
    else:
        ra = 0
    #    print("Angle of rotation:", ra)

    """
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
    """
    return ra


def fourier_pitch(img: object) -> float:
    """is a function that takes the 2 D PSD image of an array of straight lines. It is assumed that the diffreaction pattern corresponds to
    that of an array of straight, mostly vertical lines but that they may be slightly rotated by an angle theta. The function straightens the diffraction patern by rotating it by
    "-theta" . Then integrates it along qy at every qx pixel. Next it subtracts the background from the I(qx).It then finds the peaks and plots them as qx vs peak order and fits
    them to a line. The slope is the average qo. From that it calculates a pitch.

    Args:
        img (object): 2D PSD image

    Returns:
        tuple[float, list]: pitch of the grating, FFT peak positions with intensity
    """
    lmax = img.lmax
    imax = img.imax

    # At this point, it is assumed that the Fourier peaks are all vertical at fixed qx along qy.
    # The function adds up all of the columns -- integrates along qy for each qx
    intqwave = np.sum(img.image_PDS, axis=0)

    # background estimates the background in intqxave so that we can subtract it before identifying peaks
    background = gaussian_filter1d(intqwave, 6)
    newintqwave = intqwave - background

    # peaks will find the peaks in intqxave after background subtraction. Note that the coordinates are in pixel units
    peaks = find_peaks(
        newintqwave,
        prominence=0.15 * np.max(newintqwave),
        distance=abs(np.argmax(newintqwave) - len(newintqwave // 2)) // 50,
    )
    """
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(intqwave, label="data")
    plt.plot(background, label="background")
    plt.plot(newintqwave, label="subtracted")
    plt.plot(peaks[0], newintqwave[peaks[0]], "ro", label="peaks")
    plt.xlabel("pixel")
    plt.ylabel("Intensity")
    plt.legend()
    plt.title("Frequency Spectrum")
    plt.show()
    """

    # This could be tricky for DSA samples where the ebeam guiding pattern may show at a lower frequency.
    # Identify the 1st order peak as the peak with the maximum amplitude above the zero frequency and below imax/2.
    # Assuming 'peakpositions' contains the positions of all peaks.
    valid_peaks = ((lmax / 2) < peaks[0]) & (peaks[0] < imax)

    # Check if any valid peaks were found
    if np.any(valid_peaks):
        zero_order = int(np.median(peaks[0]))
        # To identify the correct peak order number, we first need to identify the 1st order peak.
        maxposition = np.argmax(intqwave[zero_order : zero_order + lmax // 3])
        # get the peak positions in pixel units centered at lmax/2+1
        peakpositions = peaks[0] - int(np.median(peaks[0]))
        kx = img.kscale * peakpositions

        # Then we normalize all peak relative to the maximum rounded to 0.01 this would leave us room for rounding the peak order numbers
        # when the density multiplication is greater than 2, even though it may seem strange to have peak order numbers in decimal fractions.
        peakpositions = np.round(peakpositions / maxposition, decimals=2)
        # rewrite peakpositions in format: {"peak order No., peak position in nm^-1"}
        peakpositions = np.transpose([peakpositions, kx])

        slope, intercept = np.polyfit(peakpositions[:, 0], peakpositions[:, 1], deg=1)

        # fitpitch = 2 Pi/m /. fitslope
        fitpitch = 2 * np.pi / slope

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.plot(
            peakpositions[:, 0],
            peakpositions[:, 1],
            "o",
            label="peakpositions",
            color="gray",
        )
        plt.plot(
            peakpositions[:, 0], slope * peakpositions[:, 0] + intercept, label="fit"
        )
        plt.xlabel("Peak order")
        plt.ylabel("kx (nm^-1)")
        plt.title(f"L0 = {round(fitpitch, 2)} nm")
        plt.legend()
        plt.show()

        return fitpitch
    else:
        raise ValueError(
            "No valid peaks were found or incorrect data for peak detection."
        )


def disk_filter(r1: float, r2: float, imsize: float) -> np.ndarray:
    """DiskFilter makes a bandpass filter in a square matrix of size "imsize" . The filter is an array of 1' s and 0' s.
    The position where there is a "1" mean that those frequencies will "pass" and where there is a "zero" frequencies will be "blocked".
    In the disk filter the array of 1' s and 0' s follows the radial assignment r1 and r2.

    Args:
        r1 (float): radial assingment for Low frequency filter
        r2 (float): radial assingment for High frequency filter
        imsize (float): _description_

    Returns:
        np.ndarray: _description_
    """
    # Create a mesh grid to represent the Cartesian coordinates
    x, y = np.meshgrid(range(imsize), range(imsize))
    xo = imsize // 2 + 1
    yo = imsize // 2 + 1

    # Convert the Cartesian coordinates to polar coordinates
    r = np.sqrt((x - xo) ** 2 + (y - yo) ** 2)

    # Create the disk filter
    filter = np.zeros((imsize, imsize), dtype=int)

    # Mark regions as pass or block based on radial assignments
    filter[(r <= r1)] = 0  # Low frequencies (block)
    filter[(r1 < r) & (r < r2)] = 1  # Mid frequencies (pass)
    filter[r >= r2] = 0  # High frequencies (block)

    # Ensure the center (zero frequency) remains as pass
    filter[xo, yo] = 1

    return filter


def filter_img(img: object) -> np.ndarray:
    """_summary_

    Args:
        img (object): _description_

    Returns:
        np.ndarray: _description_
    """
    # Sets the low and high frequency cutoffs
    fLow = 0.3 * 2 * np.pi / img.fitpitch / img.kscale
    fHigh = 1 + (0.9 * img.lmax / 2)

    # Applies a frequency filter to the FFT
    filteredImage = img.image_FFT * disk_filter(fLow, fHigh, img.lmax)
    # Preforms the inverse Fourier transform to the filtered FFT
    filteredImage = ifft2(ifftshift(filteredImage)).real

    return tools.clip_image(filteredImage)
