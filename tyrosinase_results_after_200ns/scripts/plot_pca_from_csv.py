#!/usr/bin/env python3

import os
import gc

import pandas as pd
import matplotlib.pyplot as plt


# ===================================================
# USER SETTINGS
# ===================================================

# Directory containing the PCA coordinate CSV files
input_dir = "pca_results"

# Directory where new figures will be saved
output_dir = "pca_figures_modified"

os.makedirs(
    output_dir,
    exist_ok=True
)


# ===================================================
# FILE DEFINITIONS
# ===================================================

systems = [
    "dark",
    "white"
]

replicates = [
    1,
    2,
    3
]


# ===================================================
# REPLICATE COLORS
# ===================================================

rep_colors = {
    1: "blue",
    2: "orange",
    3: "green"
}


# ===================================================
# REPLICATE LEGEND NAMES
# ===================================================

# Change these to whatever you want the legend to say.

#rep_labels = {
#    1: "Replicate 1",
#    2: "Replicate 2",
#    3: "Replicate 3"
#}


# ===================================================
# SYSTEM-SPECIFIC LEGEND NAMES
# ===================================================

# If you want different names for dark and white,
# use this instead of rep_labels above.

# Example:
#
rep_labels = {
     "dark": {
         1: "Dark Replicate 1",
         2: "Dark Replicate 2",
         3: "Dark Replicate 3"
    },
    "white": {
         1: "White Replicate 1",
         2: "White Replicate 2",
         3: "White Replicate 3"
     }
}


# ===================================================
# MARKER SETTINGS
# ===================================================

MARKER = "o"

MARKER_SIZE = 100

EDGE_COLOR = "black"

EDGE_WIDTH = 1.0

ALPHA = 1.0


# ===================================================
# AXIS LIMITS
# ===================================================

# These are the values you were using in your
# original PCA plotting script.

PC1_MIN = -45
PC1_MAX = 20

PC2_MIN = -30
PC2_MAX = 30


# ===================================================
# AXIS LABELS
# ===================================================

X_LABEL = "PC1"

Y_LABEL = "PC2"


# ===================================================
# FONT SIZES
# ===================================================

AXIS_LABEL_SIZE = 18

TICK_LABEL_SIZE = 18

TITLE_SIZE = 16

LEGEND_FONT_SIZE = 18


# ===================================================
# TITLES
# ===================================================

# Individual replicate plot titles

individual_titles = {
    "dark": "Dark Replicate {rep}",
    "white": "White Replicate {rep}"
}


# All-replicate plot titles

all_replicate_titles = {
    "dark": "Dark PCA",
    "white": "White PCA"
}


# ===================================================
# LEGEND SETTINGS
# ===================================================

LEGEND_FRAME = True

LEGEND_FACE_COLOR = "white"

LEGEND_EDGE_COLOR = "gray"

LEGEND_FRAME_ALPHA = 0.5

LEGEND_FANCYBOX = False


# ===================================================
# GRID SETTINGS
# ===================================================

GRID_ENABLED = True

GRID_LINE_STYLE = "-"

GRID_LINE_WIDTH = 0.7

GRID_ALPHA = 1.0


# ===================================================
# FIGURE SETTINGS
# ===================================================

FIGURE_WIDTH = 9

FIGURE_HEIGHT = 7

DPI = 300


# ===================================================
# HELPER FUNCTION
# ===================================================

def get_csv_filename(
    system_name,
    rep_id
):

    return os.path.join(
        input_dir,
        f"pc_coordinates_"
        f"{system_name}_rep{rep_id}.csv"
    )


# ===================================================
# CHECK INPUT FILES
# ===================================================

print()
print("=" * 60)
print("CHECKING PCA CSV FILES")
print("=" * 60)

for system_name in systems:

    for rep_id in replicates:

        filename = get_csv_filename(
            system_name,
            rep_id
        )

        if not os.path.exists(filename):

            raise FileNotFoundError(
                f"\nPCA coordinate file not found:\n"
                f"{filename}"
            )

        print(
            f"{system_name} "
            f"rep{rep_id}: OK"
        )


# ===================================================
# FUNCTION:
# LOAD PCA COORDINATES
# ===================================================

def load_pca_coordinates(
    system_name,
    rep_id
):

    filename = get_csv_filename(
        system_name,
        rep_id
    )

    data = pd.read_csv(
        filename,
        usecols=[
            "PC1",
            "PC2"
        ]
    )

    return data


# ===================================================
# FUNCTION:
# APPLY COMMON PLOT SETTINGS
# ===================================================

def format_axes(
    ax
):

    # -----------------------------------------------
    # Axis limits
    # -----------------------------------------------

    ax.set_xlim(
        PC1_MIN,
        PC1_MAX
    )

    ax.set_ylim(
        PC2_MIN,
        PC2_MAX
    )

    # -----------------------------------------------
    # Axis labels
    # -----------------------------------------------

    ax.set_xlabel(
        X_LABEL,
        fontsize=AXIS_LABEL_SIZE
    )

    ax.set_ylabel(
        Y_LABEL,
        fontsize=AXIS_LABEL_SIZE
    )

    # -----------------------------------------------
    # Tick labels
    # -----------------------------------------------

    ax.tick_params(
        axis="both",
        labelsize=TICK_LABEL_SIZE
    )

    # -----------------------------------------------
    # Grid
    # -----------------------------------------------

    if GRID_ENABLED:

        ax.grid(
            True,
            which="major",
            axis="both",
            linestyle=GRID_LINE_STYLE,
            linewidth=GRID_LINE_WIDTH,
            alpha=GRID_ALPHA
        )


# ===================================================
# MAKE INDIVIDUAL REPLICATE PLOTS
# ===================================================

print()
print("=" * 60)
print("MAKING INDIVIDUAL REPLICATE PLOTS")
print("=" * 60)


for system_name in systems:

    for rep_id in replicates:

        print(
            f"Plotting "
            f"{system_name} rep{rep_id}"
        )

        # -------------------------------------------
        # Load CSV
        # -------------------------------------------

        data = load_pca_coordinates(
            system_name,
            rep_id
        )

        # -------------------------------------------
        # Create figure
        # -------------------------------------------

        fig, ax = plt.subplots(
            figsize=(
                FIGURE_WIDTH,
                FIGURE_HEIGHT
            )
        )

        # -------------------------------------------
        # Plot
        # -------------------------------------------

        ax.scatter(
            data["PC1"],
            data["PC2"],
            s=MARKER_SIZE,
            marker=MARKER,
            color=rep_colors[rep_id],
            edgecolor=EDGE_COLOR,
            linewidth=EDGE_WIDTH,
            alpha=ALPHA,
            label=rep_labels[system_name][rep_id]
        )

        # -------------------------------------------
        # Format axes
        # -------------------------------------------

        format_axes(
            ax
        )

        # -------------------------------------------
        # Title
        # -------------------------------------------

        #ax.set_title(
        #    individual_titles[
        #        system_name
        #    ].format(
        #        rep=rep_id
        #    ),
        #    fontsize=TITLE_SIZE
        #)

        # -------------------------------------------
        # Legend
        # -------------------------------------------

        ax.legend(
            fontsize=LEGEND_FONT_SIZE,
            frameon=LEGEND_FRAME,
            facecolor=LEGEND_FACE_COLOR,
            edgecolor=LEGEND_EDGE_COLOR,
            framealpha=LEGEND_FRAME_ALPHA,
            fancybox=LEGEND_FANCYBOX
        )

        # -------------------------------------------
        # Layout
        # -------------------------------------------

        fig.tight_layout()

        # -------------------------------------------
        # Output filename
        # -------------------------------------------

        output_file = os.path.join(
            output_dir,
            f"pca_{system_name}_rep{rep_id}.png"
        )

        # -------------------------------------------
        # Save
        # -------------------------------------------

        fig.savefig(
            output_file,
            dpi=DPI,
            bbox_inches="tight"
        )

        plt.close(
            fig
        )

        print(
            f"Saved: {output_file}"
        )

        del data

        gc.collect()


# ===================================================
# MAKE ALL-REPLICATE SYSTEM PLOTS
# ===================================================

print()
print("=" * 60)
print("MAKING ALL-REPLICATE SYSTEM PLOTS")
print("=" * 60)


for system_name in systems:

    print(
        f"Plotting all replicates: "
        f"{system_name}"
    )

    # -----------------------------------------------
    # Create figure
    # -----------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            FIGURE_WIDTH,
            FIGURE_HEIGHT
        )
    )

    # -----------------------------------------------
    # Plot each replicate
    # -----------------------------------------------

    for rep_id in replicates:

        data = load_pca_coordinates(
            system_name,
            rep_id
        )

        ax.scatter(
            data["PC1"],
            data["PC2"],
            s=MARKER_SIZE,
            marker=MARKER,
            color=rep_colors[rep_id],
            edgecolor=EDGE_COLOR,
            linewidth=EDGE_WIDTH,
            alpha=ALPHA,
            label=rep_labels[system_name][rep_id]
        )

        del data

        gc.collect()

    # -----------------------------------------------
    # Format axes
    # -----------------------------------------------

    format_axes(
        ax
    )

    # -----------------------------------------------
    # Title
    # -----------------------------------------------

    ax.set_title(
        all_replicate_titles[
            system_name
        ],
        fontsize=TITLE_SIZE
    )

    # -----------------------------------------------
    # Legend
    # -----------------------------------------------

    ax.legend(
        fontsize=LEGEND_FONT_SIZE,
        frameon=LEGEND_FRAME,
        facecolor=LEGEND_FACE_COLOR,
        edgecolor=LEGEND_EDGE_COLOR,
        framealpha=LEGEND_FRAME_ALPHA,
        fancybox=LEGEND_FANCYBOX
    )

    # -----------------------------------------------
    # Layout
    # -----------------------------------------------

    fig.tight_layout()

    # -----------------------------------------------
    # Output filename
    # -----------------------------------------------

    output_file = os.path.join(
        output_dir,
        f"pca_{system_name}_all_replicates.png"
    )

    # -----------------------------------------------
    # Save
    # -----------------------------------------------

    fig.savefig(
        output_file,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output_file}" 
    )


# ===================================================
# DONE
# ===================================================

print()
print("=" * 60)
print("PLOTTING COMPLETE")
print("=" * 60)

print()
print(
    f"Input CSV directory: "
    f"{input_dir}/"
)

print(
    f"Output figure directory: "
    f"{output_dir}/"
)

print()
print(
    "No PCA calculation was performed."
)

print(
    "Only the saved PC1/PC2 coordinates were used."
)
