#!/usr/bin/env python3
"""
Plot multiple RMSD/data files with user-defined colors.

Usage examples
--------------
1. Use automatic default colors:
   python plot_multiple_data.py file1.dat file2.dat file3.dat

2. Provide one color per input file:
   python plot_multiple_data.py file1.dat file2.dat file3.dat \
       --colors red blue green

3. Use hex colors:
   python plot_multiple_data.py file1.dat file2.dat \
       --colors "#1f77b4" "#d62728"

4. Mix named and hex colors:
   python plot_multiple_data.py file1.dat file2.dat file3.dat \
       --colors black "#E69F00" "#009E73"
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import argparse


def plot_multiple_data(
    filenames,
    colors=None,
    labels=None,
    xlabel="X-axis",
    ylabel="Y-axis",
    xlim=None,
    ylim=None,
):
    """
    Plot multiple two-column data files.

    Parameters
    ----------
    filenames : list of str
        Input data files. First column is treated as time/frame,
        second column as the plotted value.

    colors : list of str or None
        One matplotlib-compatible color per input file.
        If None, matplotlib's default color cycle is used.

    xlabel : str
        X-axis label.

    ylabel : str
        Y-axis label.

    xlim : tuple or None
        Optional x-axis limits, e.g. (0, 100).

    ylim : tuple or None
        Optional y-axis limits, e.g. (1, 3).
    """

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 6))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Use matplotlib default color cycle if colors are not supplied
    if colors is None:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = [default_colors[i % len(default_colors)] for i in range(len(filenames))]

    # Ensure each input file has a corresponding color
    if len(colors) != len(filenames):
        raise ValueError(
            f"Number of colors ({len(colors)}) must match the number of "
            f"input files ({len(filenames)})."
        )

    for filename, color, label in zip(filenames, colors, labels):
        try:
            try:
                data = np.loadtxt(filename)
            except Exception:
                data = pd.read_csv(
                    filename,
                    sep=r"\s+",
                    header=None,
                ).values

            # Confirm the file has at least two columns
            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError(
                    "Input file must contain at least two columns: X and Y."
                )

            x_original = data[:, 0]
            y = data[:, 1]

            # Convert frames to ns assuming 100 frames = 1 ns
            x_scaled = x_original + 26


            ax.plot(
                x_scaled,
                y,
                linewidth=2.5,
                alpha=1,
                color=color,
                label=label,
            )

            print(f"Successfully plotted {len(x_scaled)} points from {filename}")
            print(f"Color used: {color}")
            print(f"X range: {x_scaled.min():.2f} to {x_scaled.max():.2f}")

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)

#    ax.grid(True, alpha=0.15, linestyle="-", color="black")
    ax.tick_params(axis="both", which="major", labelsize=20)

    ax.legend(frameon=True, fontsize=20)

    plt.tight_layout()

    return fig, ax

        # Use filenames as default legend labels
    if labels is None:
        labels = [
            os.path.splitext(os.path.basename(f))[0]
            for f in filenames
        ]

    if len(labels) != len(filenames):
        raise ValueError(
            f"Number of labels ({len(labels)}) must match the number of "
            f"input files ({len(filenames)})."
        )

def main():
    parser = argparse.ArgumentParser(
        description="Plot multiple RMSD/data files with optional user-defined colors."
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="Input data files containing at least two columns.",
    )

    parser.add_argument(
        "--colors",
        nargs="+",
        default=None,
        help=(
            "Colors corresponding to each input file. "
            "Example: --colors red blue green "
            'or --colors "#1f77b4" "#d62728"'
        ),
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help=(
            "Legend labels corresponding to each input file. "
            "Example: --labels WT Mutant1 Mutant2"
        ),
    )
    parser.add_argument(
        "--output",
        default="combined_rmsd.png",
        help="Output image filename. Default: combined_rmsd.png",
    )

    parser.add_argument(
        "--xlabel",
        default="Time (ns)",
        help="X-axis label. Default: Time (ns)",
    )

    parser.add_argument(
        "--ylabel",
        default="RMSF ($\\AA$)",
        help="Y-axis label. Default: RMSD (Å)",
    )

    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=(1, 3),
        metavar=("YMIN", "YMAX"),
        help="Y-axis limits. Default: 1 3",
    )

    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Optional X-axis limits.",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening the plot window.",
    )

    args = parser.parse_args()

    # Validate supplied matplotlib colors before plotting
    if args.colors is not None:
        if len(args.colors) != len(args.files):
            parser.error(
                f"You supplied {len(args.colors)} colors for {len(args.files)} files. "
                "Provide exactly one color for each file."
            )

        for color in args.colors:
            if not plt.matplotlib.colors.is_color_like(color):
                parser.error(
                    f"'{color}' is not a valid matplotlib color. "
                    "Use a named color (e.g., red), hex color (e.g., #1f77b4), "
                    "or RGB/RGBA-compatible matplotlib color."
                )

    fig, ax = plot_multiple_data(
        filenames=args.files,
        colors=args.colors,
        labels=args.labels,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        xlim=args.xlim,
        ylim=args.ylim,
    )

    fig.savefig(
        args.output,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"\nSaved figure as: {args.output}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

