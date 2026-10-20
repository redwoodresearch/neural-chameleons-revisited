"""The figures' shared look: the colours (BLUE, ORANGE, GRAY, INK, MUTED, FRAME, TEAL, PLUM), PAPER_STYLE (the paper's
type), and dejavu_style(), its DejaVu Serif variant used by the doing-X-while-hiding and trade-off figures.
"""

import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY, INK, MUTED, FRAME = "#0173b2", "#de8f05", "#949494", "#262626", "#555555", "#cccccc"
TEAL, PLUM = "#029e73", "#cc78bc"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


PAPER_STYLE = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
               "mathtext.fontset": "stix", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}


def dejavu_style() -> dict:
    """PAPER_STYLE set in DejaVu Serif at 11 pt: the type of the doing-X-while-hiding and trade-off figures."""
    return {**PAPER_STYLE, "font.size": 11, "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif"}
