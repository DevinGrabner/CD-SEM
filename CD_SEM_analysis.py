import numpy as np


def LER(lines: dict, scale: float):
    goodlineslabels = [i != 1 for i in range(1, len(lines) + 1)]

    # LER section
    edgeVariances = [np.var([coord[1] for coord in line]) for line in lines.values()]
    edgeCorrelations = [
        np.corrcoef([coord[1] for coord in line[:-1]], [coord[1] for coord in line[1:]], rowvar=False)[0, 1]
        for line in lines.values()
    ]

    # Separate whitec and blackc values explicitly
    whitec = edgeCorrelations[1::2]
    blackc = [edgeCorrelations[i] for i in range(1, len(edgeCorrelations), 2) if goodlineslabels[i // 2]]


    print("White Line c =", np.median(whitec))
    print("Black Line c =", np.median(blackc))

    # Calculate xmax and xmin from the first and last rows of the lists in the dictionary
    xmax = max(line[-1][0] for line in lines.values())
    xmin = min(line[0][0] for line in lines.values())

    LERPanel = """
    Line Edge Roughness
    No. of measured edges: {}
    Cutoff wavelengths: \(\lambda\)min = {} nm, \(\lambda\)max = {} nm
    Median LER: 3\(\sigma\)ε = {} nm
    LER 3\(\sigma\)e range \(\Rightarrow\) {} - {} nm
    """.format(
        len(lines),
        round(2 * scale, 2),
        round(scale * (xmax - xmin + 1), 2),
        round(3 * scale * np.sqrt(np.median(edgeVariances)), 2),
        round(3 * scale * np.sqrt(np.min(edgeVariances)), 2),
        round(3 * scale * np.sqrt(np.max(edgeVariances)), 2),
    )

    print(LERPanel)
