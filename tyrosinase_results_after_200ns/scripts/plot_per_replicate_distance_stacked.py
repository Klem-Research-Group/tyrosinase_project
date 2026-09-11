#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from matplotlib.ticker import MultipleLocator

def plot_multiple_data(
    filenames,
    labels=None,
    colors=None,
    xlabel="Time (ns)",
    ylabel="Distance ($\\AA$)",
    xlim=None,
    ylim=None,
    tick_fontsize=14
):
    n = len(filenames)

    fig, axes = plt.subplots(
        n, 1,
        sharex=True,
        figsize=(6.5, 2.5 * n)
    )

    if n == 1:
        axes = [axes]

    # Default colors
    if colors is None:
        default = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = [default[i % len(default)] for i in range(n)]

    # Default labels
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(n)]

    if len(labels) != n:
        raise ValueError(
            f"Expected {n} labels but received {len(labels)}."
        )

    for ax, fn, lab, c in zip(axes, filenames, labels, colors):
        try:
            try:
                data = np.loadtxt(fn)
            except Exception:
                data = pd.read_csv(fn, sep=r"\s+", header=None).values

            x = data[:, 0] / 100.0
            y = data[:, 1]

            ax.plot(
                x,
                y,
                color=c,
                lw=2,
                label=lab
            )

            if ylim is not None:
                ax.set_ylim(ylim)

            if xlim is not None:
                ax.set_xlim(xlim)

            ax.legend(
                loc="upper right",
                fontsize=16,
                frameon=False
            )
            ax.yaxis.set_major_locator(MultipleLocator(2.0))
            ax.tick_params(
               axis="both",
                which="major",
            labelsize=tick_fontsize
            )

        except Exception as e:
            print(f"Error reading {fn}: {e}")

    # One shared x-axis label
    fig.supxlabel(
        xlabel,
        fontsize=18
    )

    # One shared y-axis label
    fig.supylabel(
        ylabel,
        fontsize=18
    )

    # Leave room for shared labels
    #plt.tight_layout(
    #    rect=(0.06, 0.05, 1, 1)
    #)

    return fig


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "files",
        nargs="+",
        help="Input data files"
    )

    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Legend labels (one per input file)"
    )

    p.add_argument(
        "--colors",
        nargs="+",
        default=None,
        help="Line colors"
    )

    p.add_argument(
        "--output",
        default="combined_distance.png"
    )

    p.add_argument(
        "--xlabel",
        default="Time (ns)"
    )

    p.add_argument(
        "--ylabel",
        default="Distance ($\\AA$)"
    )

    p.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None
    )

    p.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None
    )

    p.add_argument(
        "--no-show",
        action="store_true"
    )

    args = p.parse_args()

    fig = plot_multiple_data(
        filenames=args.files,
        labels=args.labels,
        colors=args.colors,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        xlim=args.xlim,
        ylim=args.ylim,
    )

    fig.savefig(
        args.output,
        dpi=300,
        bbox_inches="tight"
    )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
