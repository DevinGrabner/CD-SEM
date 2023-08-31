import CD_SEM_tools as tools
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from skimage import filters, segmentation, morphology, util
from skimage.measure import label, regionprops


def threshold_level(image: np.ndarray, thresh: float) -> float:
    """Take input of a grayscale image and a threshold weight. Using an intensity histogram
        a weighted cutlevel for binarizing the image can be established.

    Args:
        image (np.ndarray): Image that need a threashold found for
        thresh (float): Allows for an adjustment to weighting between the black and white lines

    Returns:
        float: The weighted midpoint between the black and white levels
    """
    # Histogram of flattend input image
    histdata, bins = np.histogram(image.flatten(), bins=128, range=(0, 1))
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


def column_sums(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sums all the columns of the rotated and trimmed binary image so that the line even with breaks can be properly numbered

    Args:
        img (np.ndarray): binarized, rotated, and trimmed SEM image

    Returns:
        np.ndarray: Sum of all the columns in the image
    """
    # Sum the columns along axis 0
    column_sums = img.sum(axis=0)
    peaks = find_peaks(column_sums)[0]

    return (column_sums, peaks)


def remove_defects(binary_image: np.ndarray, crop: int = 5) -> np.ndarray:
    """Removes all lines that are invloved with a defect in the SEM grating image

    Args:
        binary_image (np.ndarray): binary image of the grating that needs processed
        crop (int, optional): Number of pixels cropped from the edge to midigate edge effects from the FFT. Defaults to 5.

    Returns:
        np.ndarray: labeled boundary image
    """
    img = util.crop(np.copy(binary_image), ((crop, crop), (crop, crop)))

    # Label all White Lines of binary image
    boundaries = boundary_image(img)
    cleaned_image = label(boundaries, connectivity=2)

    regions = regionprops(cleaned_image)

    # Get image dimensions
    image_height, image_width = cleaned_image.shape

    for region in regions:
        label_value = region.label
        coords = region.coords
        min_row, min_col, max_row, max_col = region.bbox

        col_coords = coords[:, 1]
        row_coords = coords[:, 0]

        if (
            (0 in col_coords or image_width - 1 in col_coords)
            or (
                np.sum(row_coords == 0) > 1
                or np.sum(row_coords == image_height - 1) > 1
            )
            or min_row != 0
            or max_row != image_height
            or check_continuous(coords)
        ):
            cleaned_image[cleaned_image == label_value] = 0

    cleaned_image = label(cleaned_image, connectivity=2)
    return cleaned_image


def check_continuous(coordinates: list[tuple]) -> bool:
    """Checks if a list of coordinates are continuous

    Args:
        coordinates (list): List of coordinates making up the line to check

    Returns:
        bool: True if the line is not continuous
    """
    x_coords = np.array(coordinates)[:, 0]
    diffs = np.diff(x_coords)
    return np.all(diffs != 1)


def check_lines_continuous(line_coordinates_dict: dict[str, list[tuple]]) -> None:
    """Checks all the labeled lines in a dictionary for continuity

    Args:
        line_coordinates_dict (dict): The dictionary with the line label as the key and the value a list of coordinates
    """
    for label, coordinates in line_coordinates_dict.items():
        is_continuous = check_continuous(coordinates)
        if is_continuous:
            print(f"Line {label} is not continuous.")


def boundary_image(img: np.ndarray) -> np.ndarray:
    """Taken a binary image and returns single pixel width lines that correspond
        to the outside edges of the edges of the white lines.

    Args:
        img (np.ndarray): Binary image to find line edges on

    Returns:
        np.ndarray: array of single pixel width boundary lines
    """
    # Apply segmentation edge detector to find edges
    edges = segmentation.find_boundaries(img, mode="outer", background=0).astype(
        np.uint8
    )
    return edges


def avg_rotation(boundary_img: np.ndarray) -> float:
    """Caclulates the average tilt of all the lines in the SEM image

    Args:
        boundary_img (np.ndarray): binarized SEM image that need rotated

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


def calculate_rotation_angle(points: list[tuple]) -> float:
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


def trim_rotation(image: np.ndarray, angle_rad: float) -> np.ndarray:
    """Trims up the rotated image so that the lines exent the full length. They need to all overlap so that further statistical analysis can be done.

    Args:
        image (np.ndarray): Image that needs trimmed
        angle_rad (float): The angle it was rotated at. Used to calculate how much to trim

    Returns:
        np.ndarray: trimmed array
    """
    width = image.shape[1]
    pix = int(width * np.tan(abs(angle_rad)))
    trimmed_img = np.copy(image)
    # I have no good reason to trim of 1.5*pix instead of just pix. Except that there was not enough getting trimmed. Maybe something different than 1.5 would be better?
    trimmed_img = trimmed_img[(int(1.5 * pix)) : -(int(1.5 * pix)), :]

    return trimmed_img


def boundary_coords(boundary_img: np.ndarray) -> dict[str, list[tuple]]:
    """Takes the cleaned edge boundary image and returns a dictionary with with the coordinates
      of all the points making up a line is made and returned.

    Args:
        boundary_img (np.ndarray): Cleaned edge boundary image

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


def rotate_edges(
    line_coordinates_dict: dict[str, list[tuple]], angle: float, center: tuple
) -> dict[str, list[tuple]]:
    """Rotates all the coordinates for each boundary line about the center of the boundaries SEM image by a specified angle.
        After rotation the boundary lines are compared and trimmed so that they are the same length and begin and end at the
        same row index.

    Args:
        line_coordinates_dict (dict[str, list[tuple]]): Dictionary of the line number and the coordinates of the line
        angle (float): Angle in radians the line needs rotated
        center (tuple): Center of the boundaries SEM image

    Returns:
        dict[str, list[tuple]]: Dictionary of the line number and coordinates of the rotated lines
    """
    center = (center[0] / 2 - 0.5, center[1] / 2 - 0.5)  # (row, column)

    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    def apply_rotation(coordinates: list[tuple]) -> list[tuple]:
        """Rotates all the coordinates in a list of coordinates by theta

        Args:
            coordinates (list[tuple]): list of (y,x) coordinate pairs

        Returns:
            list[tuple]: list of fractional (y,x) coordinate pairs
        """
        return [
            (
                ((x - center[1]) * sin_theta + (y - center[0]) * cos_theta) + center[0],
                ((x - center[1]) * cos_theta - (y - center[0]) * sin_theta) + center[1],
            )
            for y, x in coordinates
        ]

    rotated_dict = {
        key: apply_rotation(coordinates)
        for key, coordinates in line_coordinates_dict.items()
    }

    # Find the minimum and maximum y values among the rotated coordinates
    min_y = max([coords[0][0] for coords in rotated_dict.values()])

    # Trim the rotated coordinates to have the same length and starting row coordinate
    for key in rotated_dict:
        rotated_coords = rotated_dict[key]
        filtered_coords = [coord for coord in rotated_coords if min_y <= coord[0]]
        rotated_dict[key] = filtered_coords

    # Calculate the minimum length among all lists
    min_length = min(len(coords) for coords in rotated_dict.values())

    # Trim all lists to the minimum length
    for key in rotated_dict:
        rotated_dict[key] = rotated_dict[key][:min_length]

    return rotated_dict


def edge_boundary_order(
    binary_img: np.ndarray, lines: dict[str, list[tuple]]
) -> dict[str, list[tuple]]:
    """Returns list of characters 'b' for black and 'w' for white of what is two the right of each boundary line.

    Args:
        binary_img (np.ndarray): The black and white rotated binary image
        lines (dict): Dictionary of coordinate position of boundary edges

    Returns:
        dict: Key is the line edge label and the value is whether the space to the right is black or white
    """
    bw_label = {label: [] for label in lines}

    # # Iterate through each line
    # for key, coords in lines.items():
    #     coord = coords[10]  # 10th coordinate pair,

    #     # Get the value at the pixel located at midpoint_coordinates from the binary_img
    #     value_at_coord = binary_img[
    #         coord[0], coord[1] + 5
    #     ]  # column coordinate shifted right 2 pixels

    #     if value_at_coord == 1:
    #         value = "w"
    #     elif value_at_coord == 0:
    #         value = "b"
    #     bw_label[key].append(value)

    labels = list(lines.keys())
    # Iterate through each line pair
    for label_1, label_2 in zip(labels, labels[1:]):
        # Line coords for Line 1 and Line 2
        line1_coords = lines[label_1]
        line2_coords = lines[label_2]

        a = len(line1_coords)//2
        # Gets the x coordinate between the two lines
        middle_pixel_line1 = (line1_coords[a])[1]
        middle_pixel_line2 = (line2_coords[a])[1]

        # Calculate the pixel halfway between the lines
        midpoint_x = (middle_pixel_line1 + middle_pixel_line2) // 2

        print(middle_pixel_line1, middle_pixel_line2, midpoint_x)

        # Extract the pixel values from a point in the binary image
        row_values = binary_img[a, (middle_pixel_line1+1):(middle_pixel_line2-1)]
        print(row_values)

        # Calculate the average pixel value
        average_pixel_value = np.mean(row_values)
        print(average_pixel_value)

        # Get the value at the pixel located at midpoint_coordinates from the binary_img
        value_at_midpoint = binary_img[(line1_coords[a])[0], midpoint_x]

        if value_at_midpoint > 0:
            value = "w"
        elif value_at_midpoint == 0:
            value = "b"
        bw_label[label_1].append(value)

    return bw_label


def display_overlay(
    top_image: np.ndarray, bottom_image: np.ndarray, title: str, size: int
) -> None:
    """Displays and overlay of the two input images

    Args:
        top_image (np.ndarray): The image of the boundary lines
        bottom_image (np.ndarray): The Black and White image
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


def pitch_fit(lines: dict[str, list[tuple]], scale: float) -> float:
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
