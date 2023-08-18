import numpy as np
import pandas as pd


def LER(lines: dict[str, list[tuple]], scale: float):
    goodlineslabels = [i != 1 for i in range(1, len(lines) + 1)]

    # LER section
    edgeVariances = [np.var([coord[1] for coord in line]) for line in lines.values()]
    edgeCorrelations = [
        np.corrcoef(
            [coord[1] for coord in line[:-1]],
            [coord[1] for coord in line[1:]],
            rowvar=False,
        )[0, 1]
        for line in lines.values()
    ]

    # Separate whitec and blackc values explicitly
    whitec = edgeCorrelations[1::2]
    blackc = [
        edgeCorrelations[i]
        for i in range(1, len(edgeCorrelations), 2)
        if goodlineslabels[i // 2]
    ]

    print("White Line c =", np.median(whitec))
    print("Black Line c =", np.median(blackc))

    # Calculate xmax and xmin from the first and last rows of the lists in the dictionary
    xmax = max(line[-1][0] for line in lines.values())
    xmin = min(line[0][0] for line in lines.values())

    LERPanel = """
    Line Edge Roughness
    No. of measured edges: {}
    Cutoff wavelengths: \u03BBmin = {} nm, \u03BBmax = {} nm
    Median LER: 3\u03C3\u03B5 = {} nm
    LER 3\u03C3e range => {} - {} nm
    """.format(
        len(lines),
        round(2 * scale, 2),
        round(scale * (xmax - xmin + 1), 2),
        round(3 * scale * np.sqrt(np.median(edgeVariances)), 2),
        round(3 * scale * np.sqrt(np.min(edgeVariances)), 2),
        round(3 * scale * np.sqrt(np.max(edgeVariances)), 2),
    )

    print(LERPanel)


def LWR(
    lines: dict[str, list[tuple]],
    scale: float,
    resamplededgelinesdata,
    edgeCorrelations,
):
    goodlineslabels = [i != 1 for i in range(1, len(lines) + 1)]

    linewidthsdata = np.diff(resamplededgelinesdata[:, :, 2])

    whiteWidthMeans = np.mean(linewidthsdata[::2] + 1, axis=1)
    blackWidthMeans = np.mean(linewidthsdata[1:-1:2] - 1, axis=1)

    whiteLW = np.median(whiteWidthMeans)
    blackLW = np.median(blackWidthMeans)

    dutycycle = whiteLW / (whiteLW + blackLW)

    whiteWidthVariances = np.var(linewidthsdata[::2] + 1, axis=1)
    blackWidthVariances = np.var(linewidthsdata[1:-1:2] - 1, axis=1)

    whitec = edgeCorrelations[1::2]
    blackc = [
        edgeCorrelations[i]
        for i in range(1, len(edgeCorrelations), 2)
        if goodlineslabels[i // 2]
    ]

    whiteLWRPanel_list = (
        len(whiteWidthVariances),
        round(scale * whiteLW, 0.01),
        round(dutycycle, 0.01),
        round(3 * scale * np.sqrt(np.median(whiteWidthVariances)), 0.01),
        round(3 * scale * np.sqrt(np.min(whiteWidthVariances)), 0.01),
        round(3 * scale * np.sqrt(np.max(whiteWidthVariances)), 0.01),
        round(np.median(whitec), 0.01),
        round(np.min(whitec), 0.01),
        round(np.max(whitec), 0.01),
    )

    whiteLWRPanel = """
    White Line Width Roughness
    No. of measured lines: {}
    Average Line Width: {} nm
    Line duty cycle: {}
    Median LWR: 3σw = {} nm
    LWR 3σw range: {} - {} nm
    Median Lin. corr. coeff. c_white = {}
    c_white range: {} - {}
    """.format(*whiteLWRPanel_list)

    blackLWRPanel_list = (
        len(blackWidthVariances),
        round(scale * blackLW, 0.01),
        1 - round(dutycycle, 0.01),
        round(3 * scale * np.sqrt(np.median(blackWidthVariances)), 0.01),
        round(3 * scale * np.sqrt(np.min(blackWidthVariances)), 0.01),
        round(3 * scale * np.sqrt(np.max(blackWidthVariances)), 0.01),
        round(np.median(blackc), 0.01),
        round(np.min(blackc), 0.01),
        round(np.max(blackc), 0.01),
    )

    blackLWRPanel = """
    Black Line Width Roughness
    No. of measured lines: {}
    Average Line Width: {} nm
    Line duty cycle: {}
    Median LWR: 3σw = {} nm
    LWR 3σw range: {} - {} nm
    Median Lin. corr. coeff. c_black = {}
    c_black range: {} - {}
    """.format(*blackLWRPanel_list)

    print(whiteLWRPanel)
    print(blackLWRPanel)
