import CD_SEM_tools as tools
import CD_SEM_FFT as FFTcalc
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.spatial import distance
from skimage import filters, segmentation


def threshold_level(image: np.array, thresh: float) -> float:
    """Take input of a grayscale image and a threshold weight. Using an intensity histogram
        a weighted cutlevel for binarizing the image can be established.

    Args:
        image (np.array): Image that need a threashold found for
        thresh (float): Allows for an adjustment to weighting between the black and white lines

    Returns:
        float: The weighted midpoint between the black and white levels
    """
    # Histogram of flattend input image
    histdata, bins = np.histogram(image.flatten(), bins=256, range=(0, 1))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Smooth the histogram using a moving average
    hist_smoothed = np.convolve(histdata, np.ones(10) / 10, mode="valid")

    # Calculates the second derivative of the smoothed histogram
    der_2nd = np.gradient(np.gradient(hist_smoothed)) * 100
    der_2nd_smoothed = np.convolve(der_2nd, np.ones(10) / 10, mode="valid")

    # Makes the two histogram vectors the same length as 'bin_centers' for plotting pourposes
    hist_smoothed = tools.pad_vector_to_match_length(hist_smoothed, bin_centers)
    der_2nd_smoothed = tools.pad_vector_to_match_length(der_2nd_smoothed, hist_smoothed)

    # Blackwhite gets the black level from the minumum value of the 2nd derivative in the first 50% of the histogram.
    # It gets the white level from the minumum value in the upper 60% of the histogram.
    black_idx = np.argmin(der_2nd_smoothed[: len(der_2nd_smoothed) // 2])
    index = np.floor(len(der_2nd_smoothed) * 0.4).astype(int)
    white_idx = np.argmin(der_2nd_smoothed[index:]) + index

    # cutlevel is defined as the weighted midpoint between the black and white levels
    cutlevel = (1 - thresh) * bin_centers[black_idx] + thresh * bin_centers[white_idx]

    print("blackwhite =", [bin_centers[black_idx], bin_centers[white_idx]])
    print("Threshold Level =", cutlevel)

    """
    # Plot the histogram and the 2nd derivative with the cutlevel line
    plt.plot(bin_centers, hist_smoothed, label="histogram")
    plt.plot(bin_centers, der_2nd_smoothed, label="2nd derivative")
    plt.axvline(x=cutlevel, color="red", linestyle="--", label="cutlevel")
    plt.xlabel("Gray Level")
    plt.ylabel("Counts")
    plt.legend()
    plt.show()
    """

    return cutlevel


def blackwhite_image(img: np.array, midlevel: float) -> np.array:
    """Takes and image and turns it into an array of 0's and 1's based on the input threshold

    Args:
        img (np.array): Flattend image that need binarized
        midlevel (float): The cut off for deciding if the pixel should be a 0 or 1

    Returns:
        np.array: Binarized version of the input image
    """
    # Threshold the image using a midlevel value
    binary_image = (np.where(img >= midlevel, 1, 0)).astype(np.uint8)

    # Apply Gaussian filter to the binary image
    sigma = (8, 2)
    smoothed_image = filters.gaussian(
        binary_image, sigma=sigma, mode="constant", cval=0
    )
    smoothed_image = tools.rescale_array(smoothed_image, 0, 1)
    # Threshold the smoothed image using another threshold value
    bw_image = (np.where(smoothed_image >= 0.5, 1, 0)).astype(np.uint8)

    # Introduce a broken gap defect artificially (uncomment the following lines to apply the defect)
    # bw_image[20:31, 550:601] = 1

    # Display the result
    #tools.simple_image_display(bw_image, "Binary SEM Image")

    return bw_image


def boundary_image(img: np.array) -> np.array:
    """Taken a binary image and returns single pixel width lines that correspond
        to the outside edges of the outside edge of the white lines.

    Args:
        img (np.array): Binary image to find line edges on

    Returns:
        np.array: array of single pixel width boundary lines
    """
    # Apply Canny edge detector to find edges
    edges = segmentation.find_boundaries(img, mode="outer", background=0).astype(
        np.uint8
    )
    edges = (np.where(tools.rescale_array(edges, 0, 1) >= 0.5, 1, 0)).astype(np.uint8)
    return edges


def boundary_lines(boundary_img: np.array) -> dict:
    """Label each line the self.image_boundaries array and returs a dictionary of the corresponding pixel coordinates making up the line

    Args:
        binary_array (np.array): Boundary line image to label

    Returns:
        dict: Dictionary containing a mapping of each labeled line (connected component) to a list of its corresponding pixel coordinates
    """
    labeled_array, num_labels = label(boundary_img, connectivity=2, return_num=True)
    lines = {f"Line {i + 1}": [] for i in range(num_labels)}

    for region in regionprops(labeled_array):
        for pixel in region.coords:
            line_label = labeled_array[pixel[0], pixel[1]]
            lines[f"Line {line_label}"].append((pixel[0], pixel[1]))
    
    # Zero out lines with pixels at column 0 or column image width
    image_width = boundary_img.shape[1]
    for line_label, pixels in lines.items():
        first_pixel_col = min(pixels, key=lambda pixel: pixel[1])[1]
        last_pixel_col = max(pixels, key=lambda pixel: pixel[1])[1]

        if first_pixel_col == 0 or last_pixel_col == image_width - 1:
            for pixel in pixels:
                boundary_img[pixel[0], pixel[1]] = 0

    tools.simple_image_display(boundary_img, "Trimmed Line Boundaries")
    return lines


def calculate_rotation_angle(points):
    # # Fit a line to the points using linear regression
    # x, y = zip(*(point[:2] for point in points))
    # x = np.array(x)  # Convert x to a flat NumPy array
    # y = np.array(y)  # Convert y to a flat NumPy array

    # Convert the points list to a NumPy array
    points_array = np.array(points)

    # Extract x and y coordinates from the points array
    x = points_array[:, 0]
    y = points_array[:, 1]

    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    # Calculate the angle between the line and the horizontal axis
    angle_rad = -np.arctan(m)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def apply_rotation(image, pixels, angle_deg):
    # Apply rotation to the image and pixel coordinates
    rows, cols = image.shape
    rotated_image = np.zeros_like(image)
    for pixel in pixels:
        x, y = pixel
        x_rot = int(x * np.cos(np.radians(angle_deg)) - y * np.sin(np.radians(angle_deg)))
        y_rot = int(x * np.sin(np.radians(angle_deg)) + y * np.cos(np.radians(angle_deg)))
        if 0 <= x_rot < rows and 0 <= y_rot < cols:
            rotated_image[x_rot, y_rot] = image[x, y]

    return rotated_image

def boundary_edges_rotate(boundary_edges, lines):

    # Calculate the average rotation angle for each line
    rotation_angles = [calculate_rotation_angle(lines[f"Line {line_label}"]) for line_label in range(1, len(lines)+1)]
        

    # Calculate the average rotation angle for all lines
    avg_rotation_angle = np.mean(rotation_angles)

    # Apply rotation to the binary_array and all the lines' pixel coordinates
    binary_array_rotated = apply_rotation(boundary_edges, [(x, y) for x, y in np.argwhere(boundary_edges)], avg_rotation_angle)

    tools.simple_image_display(binary_array_rotated, "rotated")

    # Apply rotation to all the lines
    rotated_lines = {line_label: apply_rotation(boundary_edges, pixels, avg_rotation_angle) for line_label, pixels in lines.items()}

    if check_matching_lines(binary_array_rotated, lines, rotated_lines):
        return binary_array_rotated, rotated_lines
    else:
        raise ValueError(
            "Some pixels in the lines were modified during the rotation."
        )


def check_matching_lines(rotated_image, lines, lines_rotated):
        # Subtract all the rotated lines from the original image
        subtracted_image = rotated_image.copy()
        for line_label, pixels in lines_rotated.items():
            # Convert pixels list to a NumPy array and reshape it to match the shape of lines[line_label]
            pixels_array = np.array(pixels)
            pixels_array = pixels_array.reshape((-1,2))

            # Perform the subtraction (assuming pixels_array contains (x, y) coordinates)
            subtracted_image[pixels_array[:, 0], pixels_array[:, 1]] -= 1

        # Check if the result is an image of all zeros
        all_zeros = np.all(subtracted_image == 0)

        return all_zeros