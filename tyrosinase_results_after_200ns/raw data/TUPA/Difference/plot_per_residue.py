#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

# ============================================================
# USER INPUT
# ============================================================

filename = "dark-white.dat"
delimiter = None              # Auto-detect whitespace
                                # Use "," or "\t" if needed

# ------------------------------------------------------------
# Amino acid charge groups
# ------------------------------------------------------------

positive = ['K', 'R']
negative = ['D', 'E']
neutral = ['A', 'N', 'C', 'Q', 'G', 'I', 'L', 'M',
           'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'H']

histidine = ['X']

# Mutation site code
mutation_site = ['B']


# ============================================================
# PLOT SETTINGS
# ============================================================

# Figure size
FIGSIZE = (12, 7)

# ------------------------------------------------------------
# Font sizes
# ------------------------------------------------------------

AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 20
LEGEND_FONTSIZE =  18

# ------------------------------------------------------------
# Tick spacing
# ------------------------------------------------------------

X_TICK_SPACING = 50
Y_TICK_SPACING = 1

# ------------------------------------------------------------
# Axis limits
# ------------------------------------------------------------

X_LIM = None          # Example: (0, 400)
Y_LIM = (-3, 3)

# ------------------------------------------------------------
# Marker settings
# ------------------------------------------------------------

MARKER_SIZE = 100
MARKER_EDGE_WIDTH = 0.5

# ------------------------------------------------------------
# Reference line
# ------------------------------------------------------------

REFERENCE_LINE_COLOR = 'gray'
REFERENCE_LINE_STYLE = '--'
REFERENCE_LINE_WIDTH = 1


# ============================================================
# LOAD DATA
# ============================================================

# Expected format:
#
# residue_index   value   amino_acid
#
# Example:
#
# 1   0.85   K
# 2   0.60   D
# 3   0.75   A
# 4   0.90   B

data = np.genfromtxt(
    filename,
    dtype=str,
    delimiter=delimiter
)

# If file has only one line, make it 2D
if data.ndim == 1:
    data = np.array([data])


# ============================================================
# PARSE DATA
# ============================================================

x = data[:, 0].astype(float)
y = data[:, 1].astype(float)
aas = data[:, 2]


# ============================================================
# FILTER VALUES
# ============================================================

# Keep only values greater than +0.2
# or less than -0.2

mask = (y > 0.2) | (y < -0.2)

x = x[mask]
y = y[mask]
aas = aas[mask]


# ============================================================
# ASSIGN COLORS BY AMINO ACID TYPE
# ============================================================

colors = []

for aa in aas:

    if aa in positive:
        colors.append('blue')

    elif aa in negative:
        colors.append('red')

    elif aa in histidine:
        colors.append('green')

    elif aa in mutation_site:
        colors.append('purple')

    else:
        colors.append('black')


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=FIGSIZE
)


# ============================================================
# ZERO REFERENCE LINE
# ============================================================

ax.axhline(
    y=0,
    color=REFERENCE_LINE_COLOR,
    linestyle=REFERENCE_LINE_STYLE,
    linewidth=REFERENCE_LINE_WIDTH
)


# ============================================================
# PLOT EACH AMINO ACID GROUP
# ============================================================

plot_groups = [

    ('positive', 'o', 'blue', 'Positive'),

    ('negative', 'o', 'red', 'Negative'),

    ('neutral', 'o', 'black', 'Neutral'),

    ('histidine', 'p', 'green',
     'Active site Histidine'),

    ('mutation', 'x', 'rebeccapurple',
     'Mutation Site')
]


for aa_type, marker, color, label in plot_groups:

    indices = []

    for i, aa in enumerate(aas):

        if aa_type == 'positive' and aa in positive:
            indices.append(i)

        elif aa_type == 'negative' and aa in negative:
            indices.append(i)

        elif aa_type == 'neutral' and aa in neutral:
            indices.append(i)

        elif aa_type == 'histidine' and aa in histidine:
            indices.append(i)

        elif aa_type == 'mutation' and aa in mutation_site:
            indices.append(i)

        if indices:

            if aa_type == 'mutation':
                ax.scatter(
                x[indices],
                y[indices],
                marker='x',
                c='rebeccapurple',
                s=100,
                linewidths=3.0,
                label=label
                )

            else:
                ax.scatter(
                x[indices],
                y[indices],
                marker=marker,
                c=color,
                s=MARKER_SIZE,
                edgecolors='k',
                linewidths=MARKER_EDGE_WIDTH,
                label=label
                )

# ============================================================
# AXIS LABELS
# ============================================================

ax.set_xlabel(
    "Residue Index",
    fontsize=AXIS_LABEL_FONTSIZE
)

ax.set_ylabel(
    r"$\Delta$E (MV/cm)",
    fontsize=AXIS_LABEL_FONTSIZE
)


# ============================================================
# AXIS LIMITS
# ============================================================

if X_LIM is not None:
    ax.set_xlim(X_LIM)

if Y_LIM is not None:
    ax.set_ylim(Y_LIM)


# ============================================================
# TICK SPACING
# ============================================================

ax.xaxis.set_major_locator(
    MultipleLocator(X_TICK_SPACING)
)

ax.yaxis.set_major_locator(
    MultipleLocator(Y_TICK_SPACING)
)


# ============================================================
# TICK FONT SIZE
# ============================================================

ax.tick_params(
    axis='both',
    which='major',
    labelsize=TICK_LABEL_FONTSIZE
)


# ============================================================
# LEGEND
# ============================================================

legend_handles = [
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        markerfacecolor='blue',
        markeredgecolor='k',
        markersize=8,
        label='Positive'
    ),

    Line2D(
        [0], [0],
        marker='o',
        color='w',
        markerfacecolor='red',
        markeredgecolor='k',
        markersize=8,
        label='Negative'
    ),

    Line2D(
        [0], [0],
        marker='o',
        color='w',
        markerfacecolor='black',
        markeredgecolor='k',
        markersize=8,
        label='Neutral'
    ),

    Line2D(
        [0], [0],
        marker='p',
        color='w',
        markerfacecolor='green',
        markeredgecolor='k',
        markersize=9,
        label='Active site Histidine'
    ),

    Line2D(
        [0], [0],
        marker='x',
        color='rebeccapurple',
        markersize=9,
        markeredgewidth=2,
        label='Mutation Site'
    )
]

ax.legend(
    handles=legend_handles,
    fontsize=LEGEND_FONTSIZE,
    frameon=True
)

# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout()


# ============================================================
# SAVE FIGURE
# ============================================================

plt.savefig(
    'white_gamd_tupa_cu1_new.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)


# ============================================================
# DISPLAY
# ============================================================

plt.show()
