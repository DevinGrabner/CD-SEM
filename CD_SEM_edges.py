import CD_SEM_tools as tools
import CD_SEM_FFT as FFTcalc
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, segmentation, morphology, util
from skimage.measure import label, regionprops


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

    #    print("blackwhite =", [bin_centers[black_idx], bin_centers[white_idx]])
    #    print("Threshold Level =", cutlevel)

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
    smoothed_image = tools.rescale_array(smoothed_image)
    # Threshold the smoothed image using another threshold value
    bw_image = (np.where(smoothed_image >= 0.5, 1, 0)).astype(np.uint8)

    # Introduce a broken gap defect artificially (uncomment the following lines to apply the defect)
    # bw_image[20:31, 550:601] = 1

    # Display the result
    # tools.simple_image_display(bw_image, "Binary SEM Image")

    return bw_image


def remove_defects(binary_image, crop=5):
    img = util.crop(np.copy(binary_image), ((crop, crop), (crop, crop)))

    # Label all White Lines of binary image
    labeled_img = label(img, connectivity=2)

    # Removes all lines which touch the parrallel edges and do not extend all the way accross
    # Get region properties
    regions = regionprops(labeled_img)

    # Get image dimensions
    image_height, image_width = labeled_img.shape

    # Create an array of zeros with the same shape as the original image
    cleaned_image = np.zeros_like(labeled_img)

    # Compute median solidity value (Convexity of a shape)
    solidities = np.array([region.solidity for region in regionprops(labeled_img)])
    median_solidity = np.median(solidities)
    solidity_range = 0.5

    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox

        # Check if the region touches the left or right margin
        # Check if the region doesn't extend the full height of the image
        if (
            min_col != 0
            and max_col != image_width
            and min_row == 0
            and max_row == image_height
        ):
            if (
                (1 - solidity_range) * median_solidity
                < region.solidity
                < (1 + solidity_range) * median_solidity
            ):
                for coord in region.coords:
                    cleaned_image[coord[0], coord[1]] = 1

    cleaned_image = label(cleaned_image, connectivity=2)
    return cleaned_image


def boundary_image(img: np.array) -> np.array:
    """Taken a binary image and returns single pixel width lines that correspond
        to the outside edges of the edges of the white lines.

    Args:
        img (np.array): Binary image to find line edges on

    Returns:
        np.array: array of single pixel width boundary lines
    """
    # Apply Canny edge detector to find edges
    edges = segmentation.find_boundaries(img, mode="outer", background=0).astype(
        np.uint8
    )
    edges = (np.where(tools.rescale_array(edges) >= 0.5, 1, 0)).astype(np.uint8)
    return edges


def avg_rotation(boundary_img: np.array) -> float:
    """Caclulates the average tilt of all the lines in the SEM image

    Args:
        boundary_img (np.array): binarized SEM image that need rotated

    Returns:
        float: The average angle in radian that the lines are tilted
    """
    # Calculate the rotation angle for each line
    rotation_angles = [
        calculate_rotation_angle(region.coords) for region in regionprops(boundary_img)
    ]

    # Calculate the average rotation angle for all lines
    avg_rotation_angle = np.mean(rotation_angles)
    print(f"Average Rotation Angle: {round(np.degrees(avg_rotation_angle), 2)} Degrees")
    return avg_rotation_angle


def calculate_rotation_angle(points: list) -> float:
    """Fits a linear line the input boundary edge and calculates the tilt angle

    Args:
        points (list): All (row, column) points that make up the boundary edge line

    Returns:
        float: angle in degree the line need to rotate
    """
    slope = tools.linear_fit(points)[0]

    # Calculate the angle between the line and the horizontal axis
    angle_rad = -np.arctan(1 / slope)

    return angle_rad


def trim_rotation(image: np.array, angle_rad: float) -> np.array:
    """Trims up the rotated image so that the lines exent the full length. They need to all overlap so that further statistical analysis can be done.

    Args:
        image (np.array): Image that needs trimmed
        angle_rad (float): The angle it was rotated at. Used to calculate how much to trim

    Returns:
        np.array: trimmed array
    """
    width = image.shape[1]
    pix = int(width * np.tan(abs(angle_rad)))
    trimmed_img = np.copy(image)
    # I have no good reason to trim of 1.5*pix instead of just pix. Except that there was not enough getting trimmed. Maybe something different than 1.5 would be better?
    trimmed_img = trimmed_img[(int(1.5 * pix)) : -(int(1.5 * pix)), :]

    return trimmed_img


def boundary_edges(boundary_img: np.array) -> dict:
    """Takes the cleaned edge boundary image and returns a dictionary with with the coordinates
      of all the points making up a line is made and returned.

    Args:
        boundary_img (np.array): Cleaned edge boundary image

    Returns:
        dict: dictionary of the labeled boundary edge lines. Keyword "Line #": Value(List of points that make up the line)
    """
    num_labels = np.amax(boundary_img)
    lines = {f"Line {i + 1}": [] for i in range(num_labels)}

    for region in regionprops(boundary_img):
        for pixel in region.coords:
            line_label = boundary_img[pixel[0], pixel[1]]
            lines[f"Line {line_label}"].append((pixel[0], pixel[1]))
            # [row], [column]

    return lines


def edge_boundary_order(binary_img: np.array, lines: dict) -> str:
    """Returns a string defining whether it is a white are black space between the first two lines, defining the begining of the alternating pattern.

    Args:
        binary_img (np.array): The black and white rotated binary image
        lines (dict): Dictionary of coordinate position of boundary edges

    Returns:
        str: "white" or "black"
    """
    # Line coords for Line 1 and Line 2
    line1_coords = lines["Line 1"]
    line2_coords = lines["Line 2"]

    # Gets the x coordinate for the the line at about the center of the image
    middle_pixel_line1 = (line1_coords[10])[1]
    middle_pixel_line2 = (line2_coords[10])[1]

    # Calculate the pixel halfway between the lines
    midpoint_x = int((middle_pixel_line1 + middle_pixel_line2) // 2)

    # Get the value at the pixel located at midpoint_coordinates from the binary_img
    value_at_midpoint = binary_img[(line1_coords[10])[0], midpoint_x]

    if value_at_midpoint == 1:
        print("The space between lines 1 and 2 is a 'white' space")
        return "white"
    elif value_at_midpoint == 0:
        print("The space between lines 1 and 2 is a 'black' space")
        return "black"


def display_overlay(
    top_image: np.array, bottom_image: np.array, title: str, size: int
) -> None:
    """Displays and overlay of the two input images

    Args:
        top_image (np.array): The image of the boundary lines
        bottom_image (np.array): The Black and White image
        title (str): Title of the image
        size (int): Size of the window to be produced
    """
    # Set the custom figure size
    plt.figure(figsize=(size, size))

    # Plot the bottom image with the default 'gray' colormap (black for 0, white for 1)
    plt.imshow(bottom_image, cmap="gray", vmin=0, vmax=1, aspect="auto")

    # Plot the top image on top of the bottom image as bright green for non-zero values
    plt.imshow(top_image, cmap="Greens", alpha=0.7, aspect="auto")

    # Show the plot
    plt.axis("off")  # Turn off axis ticks and labels
    plt.colorbar()
    plt.title(title)
    plt.show()


def pitch_fit(lines: dict, scale: float) -> float:
    """Caclulates and returns the pitch of the grating lines

    Args:
        lines (dict): Dictionary of the line edge boundary coordiates
        scale (float): nanometers per pixel

    Returns:
        pitch: float: pitch of the grating lines in nanometers
    """
    leading_edge = tools.extract_repeating_dict_entries(lines)
    centroids = [
        sum(col for _, col in points) / len(points) for points in leading_edge.values()
    ]

    line_num = [i for i in range(1, len(centroids) + 1)]

    fit = tools.linear_fit(list(zip(centroids, line_num)))
    pitch = scale * fit[0]

    plt.plot(line_num, centroids, "bo", label="Data")
    plt.plot(line_num, (fit[0] * np.array(line_num) + fit[1]), "r-", label="Fit")
    plt.xlabel("Line Number")
    plt.ylabel("Centroid, (px)")
    plt.title(f"Pitch = {round(pitch, 3)} nm")
    plt.legend()
    plt.grid(True)
    plt.show()

    return pitch
