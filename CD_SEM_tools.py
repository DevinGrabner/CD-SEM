import tkinter as tk

def open_window():
    # Create a Tkinter root window
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    root.lift()
    root.focus_force()

