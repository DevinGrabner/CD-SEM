import numpy as np
from PIL import Image
from scipy.fftpack import fftshift, fft2, ifft2
from scipy.ndimage import gaussian_filter, morphology
from scipy.stats import scoreatpercentile
from scipy.ndimage.measurements import label
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog
import re

# Define the export file names
def export_filename(extension):
    return re.sub(r'\.tif$', extension, figurefile)

export_results = export_filename("_Results.png")
export_figure = export_filename("_edges.png")
export_psd_data = export_filename("_PSD-data.dat")
export_white_line_stats = export_filename("_WhiteLineStats.dat")
export_black_line_stats = export_filename("_BlackLineStats.dat")
export_odd_even_psds = export_filename("_OddEvenPSDs.dat")
export_odd_even_plots = export_filename("_OddEvenPlots.png")

#----------------------------------------------------------------------------------------------------------------------------

def fourier_pitch(img, theta):
    intqxave = np.sum(np.rot90(img, k=-1, axes=(0, 1)), axis=1)
    background = np.median(intqxave)
    peaks = find_peaks(intqxave - background, distance=3)
    peakpositions = peaks[0]
    peakpositions = np.column_stack((np.round(peakpositions / ((peaks[1][0] - (lmax/2 + 1))) * (lmax/2 - 1), 2), peakpositions))
    fitslope, _ = curve_fit(lambda x, m: m * x, peakpositions[:, 0], peakpositions[:, 1])
    fitpitch = 2 * np.pi / fitslope[0]
    return fitpitch, peaks[0]

def disk_filter(r1, r2, imsize):
    xo, yo = imsize // 2 + 1, imsize // 2 + 1
    transform = lambda r, theta: (r * np.cos(theta), r * np.sin(theta))
    pts = np.round(np.array([transform(r, theta) for r in np.arange(r1, r2, 1 / r1) for theta in np.arange(0, 2 * np.pi, 1 / r1)])) + (xo, yo)
    filter = np.zeros((imsize, imsize))
    filter[pts[:, 0], pts[:, 1]] = 1
    filter[xo, yo] = 1
    return filter

def threshold_level(image, thresh):
    histdata, _ = np.histogram(image.flatten(), bins=256, range=(0, 1))
    blackwhite = np.where((histdata[:-1] == np.min(histdata[:len(histdata)//2])) | (histdata[1:] == np.min(histdata[len(histdata)//2:])))
    offset = (256 - len(histdata)) // 2
    cutlevel = (blackwhite[0] + offset).mean() / 256
    return cutlevel

def rescale(image, new_min=0, new_max=1):
    old_min, old_max = np.min(image), np.max(image)
    return (image - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

def resampling_data(list, xo, xf):
    f = interpolate.interp1d(list[:, 0], list[:, 1], kind='linear')
    return np.column_stack((np.arange(xo, xf+1), f(np.arange(xo, xf+1))))

def check_pinched_edges(edges_matrix):
    pinched_edges = label(edges_matrix > 1.9 * imax)[0]
    if not pinched_edges.any():
        return edges_matrix
    output_matrix = np.where(np.isin(edges_matrix, pinched_edges, invert=True), edges_matrix, 0)
    for i, label_val in enumerate(pinched_edges):
        onepinchedlinebyrow = np.array([np.array([(i, j)]) for j in np.where(edges_matrix == label_val)])
        concaverows = np.where(np.any(np.max(onepinchedlinebyrow, axis=2) > 2, axis=1))
        pinchedrows = np.where(np.any(np.max(onepinchedlinebyrow, axis=2) == 1, axis=1))
        onepinchedlinebyrow[concaverows] = np.clip(np.hstack(onepinchedlinebyrow[concaverows]), 0, 2)
        onepinchedlinebyrow[pinchedrows] = np.where(onepinchedlinebyrow[pinchedrows] == 1, 1, 2)
        output_matrix = np.where(np.isin(output_matrix, [label_val], invert=True), output_matrix, onepinchedlinebyrow)

def pitchfit():
    data = np.array([[(label, y) for label, y in zip(goodlineslabels, line_data)] for line_data in np.transpose(np.mean(resamplededgelinesdata[:, :, 1], axis=1)[::2])])
    pfit, _ = curve_fit(lambda x, m, b: m * x + b, data[:, 0], data[:, 1])
    
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], label='Data', alpha=0.5)
    plt.plot(data[:, 0], pfit[0] * data[:, 0] + pfit[1], 'r', label='Fit')
    plt.xlabel('Line No')
    plt.ylabel('Centroid, (px)')
    plt.title(f'L_o = {round(pfit[0] * scale, 0.01)} nm')
    plt.legend()
    plt.show()
    
    return pfit[0]

pitch = pitchfit()
scale = np.median(2 * np.mean(np.diff(resamplededgelinesdata[:, :, 1], axis=1), axis=1)[::2])

edgeVariances = np.var(resamplededgelinesdata[:, :, 1], axis=1)
edgeCovariances = np.cov(resamplededgelinesdata[:, :, 1].T).diagonal()
edgeCorrelations = np.correlate(resamplededgelinesdata[:-1, :, 1], resamplededgelinesdata[1:, :, 1], mode='valid')[:, 0]
whitec = edgeCorrelations[::2]
blackc = edgeCorrelations[1::2][np.where(np.diff(goodlineslabels) != 1)[0]]

print("White Line c =", np.median(whitec))
print("Black Line c =", np.median(blackc))

linewidthsdata = np.diff(resamplededgelinesdata[:, :, 1], axis=1)
whiteWidthMeans = np.mean(linewidthsdata[::2] + 1, axis=1)
blackWidthMeans = np.mean(linewidthsdata[1::2][np.where(np.diff(goodlineslabels) != 1)[0]] - 1, axis=1)
whiteLW = np.median(whiteWidthMeans)
blackLW = np.median(blackWidthMeans)
dutycycle = whiteLW / (whiteLW + blackLW)
whiteWidthVariances = np.var(linewidthsdata[::2] + 1, axis=1)
blackWidthVariances = np.var(linewidthsdata[1::2][np.where(np.diff(goodlineslabels) != 1)[0]] - 1, axis=1)

lineplacementdata = np.mean(np.diff(resamplededgelinesdata[:, :, 1], axis=1), axis=1)
placementresiduals = lineplacementdata - np.mean(lineplacementdata)
whitePlacementMeans = np.mean(lineplacementdata[::2], axis=1)
blackPlacementMeans = np.mean(lineplacementdata[1::2][np.where(np.diff(goodlineslabels) != 1)[0]], axis=1)
whitePlacementVariances = np.var(lineplacementdata[::2], axis=1)
blackPlacementVariances = np.var(lineplacementdata[1::2][np.where(np.diff(goodlineslabels) != 1)[0]], axis=1)

def CentroidPitch(plcdata):
    meancentroids = np.mean(plcdata, axis=1)
    wcts = np.array([(label, centroid) for label, centroid in zip(goodlineslabels[::2], meancentroids[::2])])
    bcts = np.array([(label + 0.5, centroid) for label, centroid in zip(goodlineslabels[1::2][np.where(np.diff(goodlineslabels) != 1)[0]], meancentroids[1::2][np.where(np.diff(goodlineslabels) != 1)[0]])])
    periodfit, _ = curve_fit(lambda x, m, b: m * x + b, np.concatenate((wcts[:, 0], bcts[:, 0])), np.concatenate((wcts[:, 1], bcts[:, 1])))
    
    plt.figure()
    plt.scatter(wcts[:, 0], wcts[:, 1], label='White Lines', alpha=0.5)
    plt.scatter(bcts[:, 0], bcts[:, 1], label='Black Lines', alpha=0.5)
    plt.plot(np.concatenate((wcts[:, 0], bcts[:, 0])), periodfit[0] * np.concatenate((wcts[:, 0], bcts[:, 0])) + periodfit[1], 'r', label='Fit')
    plt.xlabel('Line No')
    plt.ylabel('Position y (nm)')
    plt.title(f'L_o = {round(periodfit[0] * scale, 0.01)} nm')
    plt.legend()
    plt.show()
    
    mo = periodfit[0]
    bOdd = np.mean(plcdata[::2][np.where(np.diff(goodlineslabels) != 1)[0]])
    bEven = np.mean(plcdata[1::2][np.where(np.diff(goodlineslabels) != 1)[0]])
    pwalk = round(abs(bEven - bOdd) / mo, 0.01)
    
    return {'m': periodfit[0], 'pw': pwalk}

centroidPitchandWalk = CentroidPitch(lineplacementdata)

def PlacementCorrelations(residata):
    whtplcmtcorr = np.correlate(residata[::2], residata[::2], mode='valid')[:, 0] / np.max(np.correlate(residata[::2], residata[::2], mode='valid')[:, 0])
    blkplcmtcorr = np.correlate(residata[1::2][np.where(np.diff(goodlineslabels) != 1)[0]], residata[1::2][np.where(np.diff(goodlineslabels) != 1)[0]], mode='valid')[:, 0] / np.max(np.correlate(residata[1::2][np.where(np.diff(goodlineslabels) != 1)[0]], residata[1::2][np.where(np.diff(goodlineslabels) != 1)[0]], mode='valid')[:, 0])
    
    def model(x, A, xi):
        return (1 - A) * np.exp(-x / xi) + A
    
    x = np.arange(0, len(whtplcmtcorr))
    
    whtdataperp = np.column_stack((x, whtplcmtcorr[:len(x)]))
    whtfitperp, _ = curve_fit(model, whtdataperp[:, 0], whtdataperp[:, 1], bounds=([-0.5, 0], [0.5, 15]))
    
    plt.figure()
    plt.scatter(whtdataperp[:, 0], whtdataperp[:, 1], label='Data', alpha=0.5)
    plt.plot(whtdataperp[:, 0], model(whtdataperp[:, 0], *whtfitperp), 'r', label='Fit')
    plt.xlabel('Line No')
    plt.ylabel('Autocorrelation')
    plt.title('Perpendicular Correlation')
    plt.legend()
    plt.show()
    
    blkdataperp = np.column_stack((x, blkplcmtcorr[:len(x)]))
    blkfitperp, _ = curve_fit(model, blkdataperp[:, 0], blkdataperp[:, 1], bounds=([-0.5, 0], [0.5, 15]))
    
    plt.figure()
    plt.scatter(blkdataperp[:, 0], blkdataperp[:, 1], label='Data', alpha=0.5)
    plt.plot(blkdataperp[:, 0], model(blkdataperp[:, 0], *blkfitperp), 'r', label='Fit')
    plt.xlabel('Line No')
    plt.ylabel('Autocorrelation')
    plt.title('Perpendicular Correlation')
    plt.legend()
    plt.show()
    
    xpts = int((xmax - xmin + 1) / 2)
    
    whtdatapar = np.column_stack((np.arange(0, xpts), whtplcmtcorr[:xpts]))
    whtfitpar, _ = curve_fit(model, whtdatapar[:, 0], whtdatapar[:, 1])
    
    plt.figure()
    plt.scatter(whtdatapar[:, 0] * scale, whtdatapar[:, 1], label='Data', alpha=0.5)
    plt.plot(whtdatapar[:, 0] * scale, model(whtdatapar[:, 0], *whtfitpar) * scale, 'r', label='Fit')
    plt.xlabel('distance (nm)')
    plt.ylabel('Autocorrelation')
    plt.title('Parallel Correlation')
    plt.legend()
    plt.show()
    
    blkdatapar = np.column_stack((np.arange(0, xpts), blkplcmtcorr[:xpts]))
    blkfitpar, _ = curve_fit(model, blkdatapar[:, 0], blkdatapar[:, 1])
    
    plt.figure()
    plt.scatter(blkdatapar[:, 0] * scale, blkdatapar[:, 1], label='Data', alpha=0.5)
    plt.plot(blkdatapar[:, 0] * scale, model(blkdatapar[:, 0], *blkfitpar) * scale, 'r', label='Fit')
    plt.xlabel('distance (nm)')
    plt.ylabel('Autocorrelation')
    plt.title('Parallel Correlation')
    plt.legend()
    plt.show()
    
    return {'whtfitperp': whtfitperp, 'blkfitperp': blkfitperp, 'whtfitpar': whtfitpar, 'blkfitpar': blkfitpar}

correlationPlots = PlacementCorrelations(placementresiduals)

def round_value(value, precision):
    return round(value / precision) * precision

# LPR Section
lineplacementdata = np.diff(resamplededgelinesdata[:, :, 1], axis=1)

linewidthsdata = lineplacementdata
whiteWidthMeans = np.mean(linewidthsdata[::2], axis=1)
blackWidthMeans = np.mean(linewidthsdata[1::2][np.where(np.diff(goodlineslabels) != 1)[0]], axis=1)
whiteWidthVariances = np.var(linewidthsdata[::2], axis=1)
blackWidthVariances = np.var(linewidthsdata[1::2][np.where(np.diff(goodlineslabels) != 1)[0]], axis=1)
whiteLW = np.median(whiteWidthMeans)
blackLW = np.median(blackWidthMeans)
dutycycle = whiteLW / (whiteLW + blackLW)

def CentroidPitch(plcdata):
    meancentroids = np.mean(plcdata, axis=1)
    wcts = np.array([(label, centroid) for label, centroid in zip(goodlineslabels[::2], meancentroids[::2])])
    bcts = np.array([(label + 0.5, centroid) for label, centroid in zip(goodlineslabels[1::2][np.where(np.diff(goodlineslabels) != 1)[0]], meancentroids[1::2][np.where(np.diff(goodlineslabels) != 1)[0]])])
    periodfit, _ = curve_fit(lambda x, m, b: m * x + b, np.concatenate((wcts[:, 0], bcts[:, 0])), np.concatenate((wcts[:, 1], bcts[:, 1])))
    
    plt.figure()
    plt.scatter(wcts[:, 0], wcts[:, 1], label='White Lines', alpha=0.5)
    plt.scatter(bcts[:, 0], bcts[:, 1], label='Black Lines', alpha=0.5)
    plt.plot(np.concatenate((wcts[:, 0], bcts[:, 0])), periodfit[0] * np.concatenate((wcts[:, 0], bcts[:, 0])) + periodfit[1], 'r', label='Fit')
    plt.xlabel('Line No')
    plt.ylabel('Position y (nm)')
    plt.title(f'L_o = {round(periodfit[0] * scale, 0.01)} nm')
    plt.legend()
    plt.show()
    
    mo = periodfit[0]
    bOdd = np.mean(plcdata[::2][np.where(np.diff(goodlineslabels) != 1)[0]])
    bEven = np.mean(plcdata[1::2][np.where(np.diff(goodlineslabels) != 1)[0]])
    pwalk = round(abs(bEven - bOdd) / mo, 0.01)
    
    return {'m': periodfit[0], 'pw': pwalk}

centroidPitchandWalk = CentroidPitch(lineplacementdata)

def PlacementCorrelations(residata):
    whtplcmtcorr = np.correlate(residata[::2], residata[::2], mode='valid')[:, 0] / np.max(np.correlate(residata[::2], residata[::2], mode='valid')[:, 0])
    blkplcmtcorr = np.correlate(residata[1::2][np.where(np.diff(goodlineslabels) != 1)[0]], residata[1::2][np.where(np.diff(goodlineslabels) != 1)[0]], mode='valid')[:, 0] / np.max(np.correlate(residata[1::2][np.where(np.diff(goodlineslabels) != 1)[0]], residata[1::2][np.where(np.diff(goodlineslabels) != 1)[0]], mode='valid')[:, 0])
    
    def model(x, A, xi):
        return (1 - A) * np.exp(-x / xi) + A
    
    x = np.arange(0, len(whtplcmtcorr))
    
    whtdataperp = np.column_stack((x, whtplcmtcorr[:len(x)]))
    whtfitperp, _ = curve_fit(model, whtdataperp[:, 0], whtdataperp[:, 1], bounds=([-0.5, 0], [0.5, 15]))
    
    plt.figure()
    plt.scatter(whtdataperp[:, 0], whtdataperp[:, 1], label='Data', alpha=0.5)
    plt.plot(whtdataperp[:, 0], model(whtdataperp[:, 0], *whtfitperp), 'r', label='Fit')
    plt.xlabel('Line No')
    plt.ylabel('Autocorrelation')
    plt.title('Perpendicular Correlation')
    plt.legend()
    plt.show()
    
    blkdataperp = np.column_stack((x, blkplcmtcorr[:len(x)]))
    blkfitperp, _ = curve_fit(model, blkdataperp[:, 0], blkdataperp[:, 1], bounds=([-0.5, 0], [0.5, 15]))
    
    plt.figure()
    plt.scatter(blkdataperp[:, 0], blkdataperp[:, 1], label='Data', alpha=0.5)
    plt.plot(blkdataperp[:, 0], model(blkdataperp[:, 0], *blkfitperp), 'r', label='Fit')
    plt.xlabel('Line No')
    plt.ylabel('Autocorrelation')
    plt.title('Perpendicular Correlation')
    plt.legend()
    plt.show()
    
    xpts = int((xmax - xmin + 1) / 2)
    
    whtdatapar = np.column_stack((np.arange(0, xpts), whtplcmtcorr[:xpts]))
    whtfitpar, _ = curve_fit(model, whtdatapar[:, 0], whtdatapar[:, 1])
    
    plt.figure()
    plt.scatter(whtdatapar[:, 0] * scale, whtdatapar[:, 1], label='Data', alpha=0.5)
    plt.plot(whtdatapar[:, 0] * scale, model(whtdatapar[:, 0], *whtfitpar) * scale, 'r', label='Fit')
    plt.xlabel('distance (nm)')
    plt.ylabel('Autocorrelation')
    plt.title('Parallel Correlation')
    plt.legend()
    plt.show()
    
    blkdatapar = np.column_stack((np.arange(0, xpts), blkplcmtcorr[:xpts]))
    blkfitpar, _ = curve_fit(model, blkdatapar[:, 0], blkdatapar[:, 1])
    
    plt.figure()
    plt.scatter(blkdatapar[:, 0] * scale, blkdatapar[:, 1], label='Data', alpha=0.5)
    plt.plot(blkdatapar[:, 0] * scale, model(blkdatapar[:, 0], *blkfitpar) * scale, 'r', label='Fit')
    plt.xlabel('distance (nm)')
    plt.ylabel('Autocorrelation')
    plt.title('Parallel Correlation')
    plt.legend()
    plt.show()
    
    return {'whtfitperp': whtfitperp, 'blkfitperp': blkfitperp, 'whtfitpar': whtfitpar, 'blkfitpar': blkfitpar}

correlationPlots = PlacementCorrelations(placementresiduals)

def round_with_precision(value, precision):
    return round(value / precision) * precision

def add_conf_data(stats_array):
    conf = [
        stats_array[:, 2] + stats_array[:, 3] - 2 * stats_array[:, -1] * np.sqrt(stats_array[:, 2] * stats_array[:, 3]),
        (stats_array[:, 2] / 4 + stats_array[:, 3] / 4 + 1/2 * stats_array[:, -1] * np.sqrt(stats_array[:, 2] * stats_array[:, 3]))
    ]
    output_array = np.insert(stats_array, 7, conf[0], axis=1)
    output_array = np.insert(output_array, -2, conf[1], axis=1)
    odd_indices = np.where(np.mod(stats_array[:, 0], 2) == 1)
    even_indices = np.where(np.mod(stats_array[:, 0], 2) == 0)
    odd_array = output_array[odd_indices]
    even_array = output_array[even_indices]
    odd_tots = np.round(np.mean(odd_array, axis=0), 0.01)
    odd_tots[0] = "Mean from odd lines"
    even_tots = np.round(np.mean(even_array, axis=0), 0.01)
    even_tots[0] = "Mean from even lines"
    tots = np.round(np.mean(output_array, axis=0), 0.01)
    tots[0] = "Mean values"
    return np.concatenate((np.round(output_array, 0.01), odd_tots[:, None], even_tots[:, None], tots[:, None]), axis=1)

def extr_odd_pair(color, y):
    color_list = white_line_stats if color == "white" else black_line_stats
    odd_positions = np.where(np.mod(color_list[:, 0], 2) == 1)
    return np.column_stack((color_list[odd_positions][:, 0], color_list[odd_positions][:, y]))

def extr_even_pair(color, y):
    color_list = white_line_stats if color == "white" else black_line_stats
    even_positions = np.where(np.mod(color_list[:, 0], 2) == 0)
    return np.column_stack((color_list[even_positions][:, 0], color_list[even_positions][:, y]))

# Assuming the values are already defined
white_placement_variances = np.array([...])
black_placement_variances = np.array([...])
black_perp_corr = np.array([...])
black_par_corr = np.array([...])
scale = 1.0  # Scale factor

white_placement_means = np.array([...])
white_width_means = np.array([...])
white_width_variances = np.array([...])
white_c = np.array([...])

black_placement_means = np.array([...])
black_width_means = np.array([...])
black_width_variances = np.array([...])
black_c = np.array([...])

white_line_stats = np.column_stack((
    good_lines_labels,
    white_placement_means * scale,
    white_width_means * scale,
    white_width_variances * scale**2,
    white_placement_variances * scale**2,
    white_c
))

black_line_stats = np.column_stack((
    np.delete(most(good_lines_labels) + 0.5, bad_black_lines_labels),
    black_placement_means * scale,
    np.delete(black_width_means, bad_black_lines_labels) * scale,
    np.delete(black_width_variances, bad_black_lines_labels) * scale**2,
    np.delete(black_placement_variances, bad_black_lines_labels) * scale**2,
    black_c
))

white_line_stats_table = np.concatenate((
    np.array([
        ["Line", "Position", "E1 σ₁²", "E2 σ₂²", "LW", "LW σ_w²",
         "σ₁²+σ₂²-2c_wσ₁σ₂", "LPR σ_p²", "(σ₁²/4)+(σ₂²/4)+((c_wσ₁σ₂)/2)", "c_w"],
        ["No.", "nm", "nm²", "nm²", "nm", "nm²", "nm²", "nm²", "nm²", ""]
    ]),
    add_conf_data(white_line_stats)
), axis=0)

black_line_stats_table = np.concatenate((
    np.array([
        ["Line", "Position", "E1 σ₁²", "E2 σ₂²", "LW", "LW σ_w²",
         "σ₁²+σ₂²-2c_wσ₁σ₂", "LPR σ_p²", "(σ₁²/4)+(σ₂²/4)+((c_wσ₁σ₂)/2)", "c_w"],
        ["No.", "nm", "nm²", "nm²", "nm", "nm²", "nm²", "nm²", "nm²", ""]
    ]),
    add_conf_data(black_line_stats)
), axis=0)

print("White Line Statistics")
print(white_line_stats_table)
print("Black Line Statistics")
print(black_line_stats_table)

# Helper function to calculate PSD
def psd_calculate(linedata, title):
    residata = [line - np.mean(line) for line in linedata]
    arrayPSD = [(scale ** 3) * 2 * np.abs(np.fft.fft(line * window)) ** 2
                for line in residata]

    meanPSD = np.mean(arrayPSD, axis=0)
    plot1 = plt.figure()
    for psd in arrayPSD:
        plt.loglog(range(Npts // 2), psd[:Npts // 2])
    plt.xlabel("f (nm^-1)")
    plt.ylabel("PSD (nm^3)")
    plt.title(title)
    plt.grid(True)

    plot2 = plt.figure()
    plt.loglog(range(Npts // 2), meanPSD[:Npts // 2], 'k-', linewidth=2)
    plt.xlabel("f (nm^-1)")
    plt.ylabel("PSD (nm^3)")
    plt.title(title + " (Mean)")
    plt.grid(True)

    plt.show()

    return meanPSD, arrayPSD


# Plotting functions for PSD pairs
def plot_psd_pair(psd_array, color):
    if color == "white":
        odd_positions = np.where(whiteLineStats[0].astype(int) % 2 == 1)[0]
        even_positions = np.where(whiteLineStats[0].astype(int) % 2 == 0)[0]
    elif color == "black":
        odd_positions = np.where(blackLineStats[0].astype(int) % 2 == 1)[0]
        even_positions = np.where(blackLineStats[0].astype(int) % 2 == 0)[0]

    odd_psd_array = psd_array[odd_positions]
    even_psd_array = psd_array[even_positions]

    odd_psd_mean = np.mean(odd_psd_array, axis=0)
    even_psd_mean = np.mean(even_psd_array, axis=0)

    odd_plot = plt.figure()
    for psd in odd_psd_array:
        plt.loglog(range(Npts // 2), psd[:Npts // 2], color='b', alpha=0.25)
    plt.xlabel("f (nm^-1)")
    plt.ylabel("PSD (nm^3)")
    plt.title("Odd Lines " + color + " PSD")
    plt.grid(True)

    odd_plot_ave = plt.figure()
    plt.loglog(range(Npts // 2), odd_psd_mean[:Npts // 2], 'b-', linewidth=2)
    plt.xlabel("f (nm^-1)")
    plt.ylabel("PSD (nm^3)")
    plt.title("Average Odd Lines " + color + " PSD")
    plt.grid(True)

    even_plot = plt.figure()
    for psd in even_psd_array:
        plt.loglog(range(Npts // 2), psd[:Npts // 2], color='r', alpha=0.25)
    plt.xlabel("f (nm^-1)")
    plt.ylabel("PSD (nm^3)")
    plt.title("Even Lines " + color + " PSD")
    plt.grid(True)

    even_plot_ave = plt.figure()
    plt.loglog(range(Npts // 2), even_psd_mean[:Npts // 2], 'r-', linewidth=2)
    plt.xlabel("f (nm^-1)")
    plt.ylabel("PSD (nm^3)")
    plt.title("Average Even Lines " + color + " PSD")
    plt.grid(True)

    plt.show()

    return odd_psd_mean, even_psd_mean


# Parameters and data
PerLineGrid = ...

# Define variables and constants
Npts = xmax - xmin + 1
fscale = 1 / (scale * Npts)
window = np.hanning(Npts + 1)[:-1]  # Hanning window
window = window / np.sqrt(np.mean(window ** 2))

# Call PSD calculation function for different datasets
edgePSDave, edgePSDarray = psd_calculate(resamplededgelinesdata[:, :, 1], "LER PSD")
whiteWiPSDave, whiteWiPSDarray = psd_calculate(linewidthsdata[::2], "White-line LWR PSD")
blackWiPSDave, blackWiPSDarray = psd_calculate(linewidthsdata[1::2], "Black-line LWR PSD")
whitePlPSDave, whitePlPSDarray = psd_calculate(lineplacementdata[::2], "White-line LPR PSD")
blackPlPSDave, blackPlPSDarray = psd_calculate(lineplacementdata[1::2], "Black-line PSD")

# Extract even and odd PSD pairs and plot them
whiteWiPSDodd, whiteWiPSDeven = plot_psd_pair(whiteWiPSDarray, "white")
blackWiPSDodd, blackWiPSDeven = plot_psd_pair(blackWiPSDarray, "black")
whitePlPSDodd, whitePlPSDeven = plot_psd_pair(whitePlPSDarray, "white")
blackPlPSDodd, blackPlPSDeven = plot_psd_pair(blackPlPSDarray, "black")

# Other plotting and export operations
# ...

