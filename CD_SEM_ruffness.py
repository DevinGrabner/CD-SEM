import CD_SEM_tools as tools
import CD_SEM_FFT as FFTcalc
import CD_SEM_edges as edges
import CD_SEM_ruffness as ruff
import CD_SEM_analysis as anlys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

######### This object holds ?? properties from CD-SEM analyis. It only auto initializes values pulled directly from the header file of the '.fit' SEM image.
######### The __call__ function will run all the necassary calculations to assign values to all object properties


class SEMImageDetails:
    def __init__(self):
        
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
        