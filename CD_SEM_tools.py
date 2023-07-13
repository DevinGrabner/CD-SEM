import tkinter as tk
import numpy as np

def open_window():
    # Create a Tkinter root window
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    root.lift()
    root.focus_force()

def rescale_array(arr, new_min, new_max) -> np.ndarray:
    min_val = np.min(arr)
    max_val = np.max(arr)
    scaled_arr = (arr - min_val) / (max_val - min_val)  # Scale values between 0 and 1
    rescaled_arr = scaled_arr * (new_max - new_min) + new_min  # Rescale to the new range
    return rescaled_arr