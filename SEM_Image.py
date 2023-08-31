import CD_SEM_tools as tools
import CD_SEM_FFT as FFTcalc
import CD_SEM_edges as edges
import CD_SEM_analysis as anly

# import CD_SEM_ruffness as ruff
# import CD_SEM_analysis as anlys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from scipy.fftpack import fftshift, fft2
from skimage import util
import numpy as np
import tifffile
from tkinter import filedialog
from typing import Final

TAG_NAMES: Final[list[str]] = ["ImageWidth", "ImageLength", "CZ_SEM"]
PIX_SIZE: Final[str] = "ap_image_pixel_size"
IMAGE_SCALE_UM: Final[float] = 0.2  # Image scale bar length in micrometers

######### This object holds ?? properties from CD-SEM analyis. It only auto initializes values pulled directly from the header file of the '.fit' SEM image.
######### The __call__ function will run all the necassary calculations to assign values to all object properties


class SEMImageDetails:
    def __init__(self):
        self.path = self._sem_image_selector()
        self.height = self._sem_image_tag(self.path, TAG_NAMES[0])
        self.width = self._sem_image_tag(self.path, TAG_NAMES[1])
        self.pix_scale, self.pix_size, self.pix_dimen = self._pix_data(
            self.path, TAG_NAMES[2]
        )
        self.imax: int | None = None  # See ImgFlat.lmax for details on variables
        self.lmax: int | None = None
        self.kscale: float | None = None
        self.fitpitch: float | None = None
        self.image_PDS_center: None | float = (
            None  # PDS of the zero frequency of the FFT image
        )
        self.rotate_angle: None | float = None

        self.peakposition: np.ndarray | None = None
        self.midlevel: float | None = None
        self.grid_boundaries: dict | None = None
        self.rotated_boundaries: dict | None = (
            None  # Fractional coordinates of rotated image
        )
        self.bw_order: dict | None = None

        # Images
        self.image = tifffile.imread(self.path)  # Original
        self.image_clipped = None  # Clipped image used for FFT analysis
        self.image_FFT = None  # Fourier Transform of clipped image
        self.image_PDS = None  # log of the power spectral density of the Fourier image
        self.image_flat = None  # Flattened image
        self.image_binary = None  # Black and White Binary image
        self.image_boundaries = None  # Binary Boundary image

    def __call__(self):
        # These operations have to deal with extracting information from the original SEM image, rotating it if it is tilted, and applying a frequency filter
        self.imax, self.lmax, self.kscale = FFTcalc.image_size(
            self.height, self.pix_scale, self.pix_size
        )
        self.image_clipped = tools.clip_image(
            FFTcalc.extract_center_part(self.image, self.lmax)
        )
        self.display_SEM_image(self.image_clipped, bar=True, title="SEM Image")

        self.image_PDS, self.image_PDS_center = FFTcalc.PDS_img(self.image_clipped)
        ###        self.display_fft_image(self.image_PDS, title="Power Spectral Density")

        self.rotate_angle = FFTcalc.rotated_angle(25, self.image_PDS, self.lmax)

        if self.rotate_angle > 0:
            self.image_clipped = tools.rotate_image(
                self.image_clipped, self.image_rotate
            )
            self.image_PDS = tools.rotate_image(self.image_PDS, self.image_rotate)

        self.image_FFT = fftshift(fft2(self.image_clipped))
        self.fitpitch = FFTcalc.fourier_pitch(self)

        self.image_flat = FFTcalc.filter_img(self)

        ###         self.display_SEM_image(self.image_flat, bar=True, title="Filtered SEM Image")

        # These operations have to deal with threasholding, binary filter
        self.midlevel = edges.threshold_level(self.image_flat, 0.6)
        self.image_binary = tools.blackwhite_image(self.image_flat, self.midlevel)
        # tools.simple_image_display(self.image_binary, "Original Binary Image")

        # Now removing defects and finding line edges
        self.image_boundaries = edges.remove_defects(np.copy(self.image_binary))

        # Clean and straighten the image with the edge boundaries
        self.rotate_angle = edges.avg_rotation(np.copy(self.image_boundaries))
        self.image_binary = edges.trim_rotation(
            tools.rotate_image(self.image_binary, self.rotate_angle), self.rotate_angle
        )

        # Make boundary coordinate dictionary, rotate to fractional coordinates, trim to same begin and end row coordinates
        self.rotated_boundaries = edges.boundary_coords(np.copy(self.image_boundaries))
        self.rotated_boundaries = edges.rotate_edges(
            self.rotated_boundaries, self.rotate_angle, (self.image_boundaries).shape
        )

        # Now that the binary image has been straightened we have to redo the line detection and line defect removal because the rotation of sigle pixel wide lines doesn't map well for visulization
        self.image_boundaries = edges.remove_defects(np.copy(self.image_binary))

        linesum, peaks = edges.column_sums(
            util.crop(np.copy(self.image_binary), ((5, 5), (5, 5)))
        )

        fitpitch_ceck = tools.plot_peaks(peaks, self.pix_scale * self.pix_size)
        if abs(fitpitch_ceck - self.fitpitch) / self.fitpitch > 0.1:
            raise ValueError(
                f"Inconsistent Fitpitch after rotation and defect removal ({fitpitch_ceck} vs. {self.fitpitch}). Needs manually checked"
            )

        edges.display_overlay(
            tools.blackwhite_image(np.copy(self.image_boundaries), 0.5),
            np.copy(self.image_binary),
            "Boundary Overlay",
            10,
        )

        self.grid_boundaries = edges.boundary_coords(np.copy(self.image_boundaries))
        self.bw_order = edges.edge_boundary_order(
            self.image_binary, self.grid_boundaries
        )

        # tools.list_barplot(linesum)

        # self.column_sum = edges.column_sums(edges.boundary_image(self.image_binary))

        # print(self.bw_order)

        ## NEEDS FIXED ###self.fitpitch = edges.pitch_fit(self.boundaries, (self.pix_size * self.pix_scale))

        # These operations have to deal with LER, LWR, LPR

        # anly.LER(self.boundaries, (self.pix_scale * self.pix_size))
        # anly.LWR(self.boundaries, (self.pix_scale * self.pix_size))

        # These operations have to deal with statistical line analysis

        # These operations have to deal with frequency domain analysis

    def _sem_image_selector(self) -> str:
        """Lets you select the image file for the object

        Returns:
            file_path (str): file path to the image
        """
        tools.open_window()
        file_path = filedialog.askopenfilename(
            filetypes=[("TIFF Files", "*.tif"), ("All Files", "*.*")]
        )
        return file_path

    def _sem_image_tag(self, file_path: str, tag_name: str) -> any:
        """Fetches information from the .tif file header

        Args:
            file_path (str): file path to the image
            tag_name (str): Name of the tag in the header we want the associated value for.

        Returns:
            any: the value for the input tag
        """
        metadata = tifffile.TiffFile(file_path)
        return metadata.pages[0].tags[tag_name].value

    def _pix_data(self, file_path: str, tag_name: str) -> tuple:
        """Fetches the data associated with the image pixel dimensions from the library in the header. The multiplication of pixel scale and pixel size puts the pixel size in nanometers

        Args:
            file_path (str): file path to the image
            tag_name (str): Name of the tag in the header we want the associated value for.

        Returns:
            tuple: [pixel_scale, pixel_size, pixel_dimension]
        """
        global PIX_SIZE

        ImagTagDict = self._sem_image_tag(file_path, "CZ_SEM")
        # List containing the proper name, value, and dimension
        PixelList = list(ImagTagDict.get("ap_image_pixel_size"))

        # Checks the pixel dimesion and assigns the appropriate scale so the dimensions are in nm
        unitConversion = {"pm": 10**-3, "nm": 1, "um": 10**3}
        if PixelList[2] not in unitConversion:
            while True:
                PixelList[1] = float(
                    input(
                        "Pixel dimension not defined in image file. Enter scale in nm/pixel: "
                    )
                )
                if PixelList[1] > 0:
                    PixelList[0] = 1
                    PixelList[2] = "nm"
                    break
        PixelList[0] = unitConversion.get(PixelList[2])
        return PixelList

    def display_SEM_image(
        self: object, image: tifffile, title: str = None, bar: bool = False
    ) -> None:
        """Displays an image with scale bar (if wanted) based on pixel size from the image

        Args:
            image (np.ndarray): image that you want displayed
            title (str): Title of the image if wanted
            bar (bool, optional): Option to display scale bar on image. Defaults to False.
        """

        # Plot the image
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")

        if bar:
            global IMAGE_SCALE_UM
            # calculate the dimensions of the scale bar
            image_height = image.shape[0]
            scale_bar_length_pixels = IMAGE_SCALE_UM / (
                self.pix_size * self.pix_scale / 1000
            )

            # calculate the position of the scale bar
            scale_bar_x = image.shape[1] - scale_bar_length_pixels - 100
            scale_bar_y = image_height - 50

            # Add the scale bar to the plot
            scale_bar = patches.Rectangle(
                (scale_bar_x, scale_bar_y),
                scale_bar_length_pixels,
                5,
                edgecolor="white",
                facecolor="white",
                linewidth=2,
            )
            plt.gca().add_patch(scale_bar)

            # Add text for the scale bar length
            scale_bar_text = f"{IMAGE_SCALE_UM} µm"
            text_props = dict(facecolor="white", edgecolor="white", linewidth=1)
            plt.text(
                scale_bar_x + scale_bar_length_pixels / 2,
                scale_bar_y - 20,
                scale_bar_text,
                color="white",
                ha="center",
            )

        # Show the plot
        plt.show()

    def display_fft_image(self: object, fimg: np.ndarray, title: str = None) -> None:
        """Displays the scaled FFT image on a colorblind friendly colorbar

        Args:
            fimg (np.ndarray): FFT image getting displayed
            title (str): Title of the image if wanted
        """
        # Define a colorblind-friendly colormap
        cmap = LinearSegmentedColormap.from_list(
            "colorblind_cmap", ["#000000", "#377eb8", "#ff7f00", "#4daf4a"], N=256
        )

        # Plot the FFT image
        plt.imshow(fimg, cmap=cmap)
        plt.colorbar(label="Intensity")
        plt.title(title)
        plt.axis("off")
        plt.show()
