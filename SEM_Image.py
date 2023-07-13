import CD_SEM_tools as tools
import CD_SEM_Calc as calc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import tifffile
from tkinter import filedialog
from typing import Final

TAG_NAMES: Final[list[str]] = ["ImageWidth", "ImageLength", "CZ_SEM"]
PIX_SIZE: Final[str] = "ap_image_pixel_size"
IMAGE_SCALE_UM: Final[float] = 0.2  # Image scale bar length in micrometers

######### This object holds 54 properties from CD-SEM analyis. It only auto initializes values pulled directly from the header file of the '.fit' SEM image.
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
        self.image_FFT_center: None | float = (
            None  # Magnitude square of the zero frequency of the FFT image
        )
        self.image_rotate: None | float = (
            None  # The angle the image needs rotated to make sure the FFT is horizontal
        )

        # Images
        self.image = tifffile.imread(self.path)  # Original
        self.image_clipped = None  # Clipped image used for FFT analysis
        self.image_FFT = None  # log of Fourier image
        self.image_flat = None  # Flattened image
        self.image_binary = None  # Black and White Binary image
        self.image_color = None  # Colorized image
        self.image_pintched = None  # Pintched Color image

        # These are all of the variable that we need at the end of the analysis
        # Line Edge Roughness  -  LER
        self.LER_edges: None | int = None  # Number of measured edges
        self.LER_wave_low: None | float = None  # nm # Cutoff Wavelengths
        self.LER_wave_high: None | float = None
        self.LER_median: None | float = None  # nm # Median LER 3*sigma
        self.LER_range_low: None | float = None  # nm # LER 3*sigma range
        self.LER_range_high: None | float = None

        # White Line Width Roughness  -  WLWR
        self.WLWR_lines: None | int = None  # Number of measured lines
        self.WLWR_avg_width: None | float = None  # nm # Average Line Width
        self.WLWR_LDC: None | float = None  # Line Duty Cycle
        self.WLWR_median: None | float = None  # nm # Median LWR 3*sigma
        self.WLWR_lin_corr: None | float = None  # Median Lin. corr. coeff. (c_white)
        self.WLWR_range_low: None | float = None  # c_white range
        self.WLWR_range_high: None | float = None

        # White Line Placement Accuracy
        self.WLPA_lines: None | int = None  # Number of measured lines
        self.WLPA_place: None | float = None  # nm # Placement Roughness 3*sigma
        self.WLPA_place_low: None | float = None  # nm # Placement 3*sigma range
        self.WLPA_place_high: None | float = None
        self.WLPA_crossline_L: None | float = None  # nm # Cross Line L_o
        self.WLPA_crossline_A: None | float = None  # nm # Cross Line A_o
        self.WLPA_inline: None | float = None  # nm # In Line
        self.WLPA_pitch: None | float = None  # nm # Pitch Lo
        self.WLPA_pitch_walk: None | float = None  # nm # Pitch Walking * Pitch Lo

        # Black Line Width Roughness
        self.BLWR_lines: None | int = None  # Number of measured lines
        self.BLWR_avg_width: None | float = None  # nm # Average Line Width
        self.BLWR_LDC: None | float = None  # Line Duty Cycle
        self.BLWR_median: None | float = None  # nm # Median LWR 3*sigma
        self.BLWR_lin_corr: None | float = None  # Median Lin. corr. coeff. (c_black)
        self.BLWR_range_low: None | float = None  # c_black range
        self.BLWR_range_high: None | float = None

        # Black Line Placement Accuracy
        self.BLPA_lines: None | int = None  # Number of measured lines
        self.BLPA_place: None | float = None  # nm # Placement Roughness 3*sigma
        self.BLPA_place_low: None | float = None  # nm # Placement 3*sigma range
        self.BLPA_place_high: None | float = None
        self.BLPA_crossline_L: None | float = None  # nm # Cross Line L_o
        self.BLPA_crossline_A: None | float = None  # nm # Cross Line A_o
        self.BLPA_inline: None | float = None  # nm # In Line

    def __call__(self):
        self.imax, self.lmax, self.kscale = calc.image_size(
            self.height, self.pix_scale, self.pix_size
        )
        self.image = tools.rescale_array(
            calc.extract_center_part(self.image, self.lmax), 0, 1
        )
        self.image_FFT, self.image_FFT_center = calc.fourier_img(self.image)
        self.image_rotate = calc.rotated_angle(25, self.image_FFT, self.lmax)

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
        """Fetches the data associated with the image pixel dimensions from the library in the header

        Args:
            file_path (str): file path to the image
            tag_name (str): Name of the tag in the header we want the associated value for.

        Returns:
            tuple: [pixel_name, pixel_size, pixel_dimension]
        """
        global PIX_SIZE

        ImagTagDict = self._sem_image_tag(file_path, "CZ_SEM")
        # List containing the proper name, value, and dimension
        PixelList = list(ImagTagDict.get("ap_image_pixel_size"))

        # Checks the pixel dimesion and assigns the appropriate scale so the dimensions are in um
        unitConversion = {"pm": 10**-6, "nm": 10**-3, "um": 1}
        if PixelList[2] not in unitConversion:
            while True:
                PixelList[1] = float(
                    input(
                        "Pixel dimension not defined in image file. Enter scale in nm/pixel: "
                    )
                )
                if PixelList[1] > 0:
                    PixelList[0] = 10**-3
                    PixelList[2] = "nm"
                    break
        PixelList[0] = unitConversion.get(PixelList[2])
        return PixelList

    def display_SEM_image(self: object, image: tifffile, bar=False) -> None:
        """Displays an image with scale bar (if wanted) based on pixel size from the image

        Args:
            image (np.ndarray): image that you want displayed
            bar (bool, optional): Option to display scale bar on image. Defaults to False.
        """
        global IMAGE_SCALE_UM

        # Plot the image
        plt.imshow(image, cmap="gray")
        plt.axis("off")

        if bar:
            # Calculate the dimensions of the scale bar
            image_height = image.shape[0]
            scale_bar_length_pixels = (IMAGE_SCALE_UM * self.pix_scale) / self.pix_size

            # Calculate the position of the scale bar
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

    def display_fft_image(self: object, fimg: np.ndarray) -> None:
        """Displays the scaled FFT image on a colorblind friendly colorbar

        Args:
            fimg (np.ndarray): FFT image getting displayed
        """
        # Define a colorblind-friendly colormap
        cmap = LinearSegmentedColormap.from_list(
            "colorblind_cmap", ["#000000", "#377eb8", "#ff7f00", "#4daf4a"], N=256
        )

        # Plot the FFT image
        plt.imshow(fimg, cmap=cmap)
        plt.colorbar(label="Intensity")
        plt.title("FFT Image")
        plt.axis("off")
        plt.show()
