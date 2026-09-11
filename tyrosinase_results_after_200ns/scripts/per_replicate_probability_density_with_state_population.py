#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import argparse
import os


# ============================================================
# USER FUNCTIONS
# ============================================================

def calculate_kde(distances, bandwidth=None):
    """Calculate KDE for a set of distance values."""

    kde = gaussian_kde(
        distances,
        bw_method=bandwidth
    )

    x_min = distances.min()
    x_max = distances.max()

    # Add margin around the data
    margin = 0.05 * (x_max - x_min)

    if margin == 0:
        margin = 0.1

    x = np.linspace(
        x_min - margin,
        x_max + margin,
        10000
    )

    density = kde(x)

    return x, density


def find_kde_peaks(x, density, prominence=None):
    """Find all local KDE peaks."""

    peak_indices, properties = find_peaks(
        density,
        prominence=prominence
    )

    peak_values = x[peak_indices]
    peak_densities = density[peak_indices]

    return peak_values, peak_densities


def calculate_state_populations(distances, peak_values):
    """Calculate the population of each peak ("state").

    Each data point is assigned to whichever peak value it is closest
    to, and the population of a peak/state is the fraction of all data
    points assigned to it. Populations across all peaks sum to 1.0.
    """

    if len(peak_values) == 0:
        return np.array([])

    distances = np.asarray(distances)
    peak_values = np.asarray(peak_values)

    # Distance from every data point to every peak
    diffs = np.abs(
        distances[:, None] - peak_values[None, :]
    )

    # Index of the nearest peak for each data point
    nearest_peak_idx = np.argmin(diffs, axis=1)

    populations = np.array([
        np.sum(nearest_peak_idx == i) / len(distances)
        for i in range(len(peak_values))
    ])

    return populations


def analyze_file(filename, bandwidth=None, prominence=None):
    """Analyze one .dat file."""

    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------

    data = np.loadtxt(
        filename,
        comments="#"
    )

    distances = data[:, 1]

    # --------------------------------------------------------
    # KDE
    # --------------------------------------------------------

    x, density = calculate_kde(
        distances,
        bandwidth=bandwidth
    )

    # --------------------------------------------------------
    # Find peaks
    # --------------------------------------------------------

    peak_values, peak_densities = find_kde_peaks(
        x,
        density,
        prominence=prominence
    )

    # --------------------------------------------------------
    # Sort peaks by density
    #
    # Largest density = first
    # Second largest density = second
    # --------------------------------------------------------

    if len(peak_values) > 0:

        sort_order = np.argsort(
            peak_densities
        )[::-1]

        peak_values = peak_values[sort_order]
        peak_densities = peak_densities[sort_order]

    # --------------------------------------------------------
    # State populations
    #
    # Fraction of data points belonging to each peak/state,
    # in the same (density-sorted) order as peak_values.
    # --------------------------------------------------------

    peak_populations = calculate_state_populations(
        distances,
        peak_values
    )

    return (
        distances,
        x,
        density,
        peak_values,
        peak_densities,
        peak_populations
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Perform KDE on multiple .dat files and "
            "calculate mean/std of the largest and "
            "second-largest peaks."
        )
    )

    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input .dat files"
    )

    parser.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help=(
            "KDE bandwidth. Default: "
            "scipy automatic bandwidth."
        )
    )

    parser.add_argument(
        "--prominence",
        type=float,
        default=None,
        help=(
            "Minimum KDE peak prominence. "
            "Default: no filtering."
        )
    )

    parser.add_argument(
        "--output",
        default="kde_peak_results.dat",
        help="Output per-file peak results."
    )

    parser.add_argument(
        "--summary",
        default="kde_peak_summary.dat",
        help="Output mean/std summary."
    )

    parser.add_argument(
        "--plot",
        default="kde_all.png",
        help="Output KDE plot."
    )

    args = parser.parse_args()

    # ========================================================
    # STORAGE
    # ========================================================

    largest_values = []
    largest_densities = []

    second_values = []
    second_densities = []

    largest_populations = []
    second_populations = []

    results = []

    # ========================================================
    # ANALYZE EACH FILE
    # ========================================================

    print("\n==============================================")
    print(" KDE Analysis of Multiple Files")
    print("==============================================\n")

    for filename in args.input_files:

        print(f"Analyzing: {filename}")

        try:

            (
                distances,
                x,
                density,
                peak_values,
                peak_densities,
                peak_populations
            ) = analyze_file(
                filename,
                bandwidth=args.bandwidth,
                prominence=args.prominence
            )

        except Exception as e:

            print(
                f"ERROR analyzing {filename}: {e}"
            )

            continue

        # ----------------------------------------------------
        # Print all peaks
        # ----------------------------------------------------

        print(
            f"  Number of peaks: {len(peak_values)}"
        )

        for i, (value, peak_density, population) in enumerate(
            zip(peak_values, peak_densities, peak_populations),
            start=1
        ):

            print(
                f"    Peak {i}: "
                f"{value:.4f} "
                f"(density = {peak_density:.6f}, "
                f"population = {population:.4f})"
            )

        # ----------------------------------------------------
        # Largest peak
        # ----------------------------------------------------

        if len(peak_values) >= 1:

            largest_value = peak_values[0]
            largest_density = peak_densities[0]
            largest_population = peak_populations[0]

            largest_values.append(
                largest_value
            )

            largest_densities.append(
                largest_density
            )

            largest_populations.append(
                largest_population
            )

        else:

            largest_value = np.nan
            largest_density = np.nan
            largest_population = np.nan

        # ----------------------------------------------------
        # Second-largest peak
        # ----------------------------------------------------

        if len(peak_values) >= 2:

            second_value = peak_values[1]
            second_density = peak_densities[1]
            second_population = peak_populations[1]

            second_values.append(
                second_value
            )

            second_densities.append(
                second_density
            )

            second_populations.append(
                second_population
            )

        else:

            second_value = np.nan
            second_density = np.nan
            second_population = np.nan

        # ----------------------------------------------------
        # Save per-file result
        # ----------------------------------------------------

        results.append(
            (
                filename,
                largest_value,
                largest_density,
                largest_population,
                second_value,
                second_density,
                second_population
            )
        )

        print(
            f"  Largest peak: "
            f"{largest_value:.4f} "
            f"(population = {largest_population:.4f})"
        )

        print(
            f"  Second-largest peak: "
            f"{second_value:.4f} "
            f"(population = {second_population:.4f})"
        )

        print()

    # ========================================================
    # CONVERT TO ARRAYS
    # ========================================================

    largest_values = np.array(
        largest_values
    )

    largest_densities = np.array(
        largest_densities
    )

    second_values = np.array(
        second_values
    )

    second_densities = np.array(
        second_densities
    )

    largest_populations = np.array(
        largest_populations
    )

    second_populations = np.array(
        second_populations
    )

    # ========================================================
    # CALCULATE STATISTICS
    # ========================================================

    largest_mean = np.mean(
        largest_values
    )

    largest_std = np.std(
        largest_values,
        ddof=1
    )

    largest_density_mean = np.mean(
        largest_densities
    )

    largest_density_std = np.std(
        largest_densities,
        ddof=1
    )

    largest_population_mean = np.mean(
        largest_populations
    )

    largest_population_std = np.std(
        largest_populations,
        ddof=1
    )

    # --------------------------------------------------------
    # Second peak
    # --------------------------------------------------------

    if len(second_values) > 0:

        second_mean = np.mean(
            second_values
        )

        second_std = np.std(
            second_values,
            ddof=1
        )

        second_density_mean = np.mean(
            second_densities
        )

        second_density_std = np.std(
            second_densities,
            ddof=1
        )

        second_population_mean = np.mean(
            second_populations
        )

        second_population_std = np.std(
            second_populations,
            ddof=1
        )

    else:

        second_mean = np.nan
        second_std = np.nan

        second_density_mean = np.nan
        second_density_std = np.nan

        second_population_mean = np.nan
        second_population_std = np.nan

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n==============================================")
    print(" FINAL SUMMARY")
    print("==============================================\n")

    print(
        "Largest peak:"
    )

    print(
        f"  Mean distance:       "
        f"{largest_mean:.4f}"
    )

    print(
        f"  Std. deviation:      "
        f"{largest_std:.4f}"
    )

    print(
        f"  Mean density:        "
        f"{largest_density_mean:.6f}"
    )

    print(
        f"  Density std. dev.:   "
        f"{largest_density_std:.6f}"
    )

    print(
        f"  Mean population:     "
        f"{largest_population_mean:.4f}"
    )

    print(
        f"  Population std. dev.:"
        f" {largest_population_std:.4f}"
    )

    print()

    print(
        "Second-largest peak:"
    )

    print(
        f"  Mean distance:       "
        f"{second_mean:.4f}"
    )

    print(
        f"  Std. deviation:      "
        f"{second_std:.4f}"
    )

    print(
        f"  Mean density:        "
        f"{second_density_mean:.6f}"
    )

    print(
        f"  Density std. dev.:   "
        f"{second_density_std:.6f}"
    )

    print(
        f"  Mean population:     "
        f"{second_population_mean:.4f}"
    )

    print(
        f"  Population std. dev.:"
        f" {second_population_std:.4f}"
    )

    # ========================================================
    # SAVE PER-FILE RESULTS
    # ========================================================

    with open(args.output, "w") as f:

        f.write(
            "# File "
            "Largest_Peak "
            "Largest_Density "
            "Largest_Population "
            "Second_Largest_Peak "
            "Second_Largest_Density "
            "Second_Largest_Population\n"
        )

        for result in results:

            filename = result[0]
            largest_value = result[1]
            largest_density = result[2]
            largest_population = result[3]
            second_value = result[4]
            second_density = result[5]
            second_population = result[6]

            f.write(
                f"{filename} "
                f"{largest_value:.8f} "
                f"{largest_density:.8f} "
                f"{largest_population:.8f} "
                f"{second_value:.8f} "
                f"{second_density:.8f} "
                f"{second_population:.8f}\n"
            )

    print(
        f"\nPer-file results saved to: "
        f"{args.output}"
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    with open(args.summary, "w") as f:

        f.write(
            "# Peak_Type "
            "Mean_Distance "
            "Std_Distance "
            "Mean_Density "
            "Std_Density "
            "Mean_Population "
            "Std_Population\n"
        )

        f.write(
            f"Largest "
            f"{largest_mean:.8f} "
            f"{largest_std:.8f} "
            f"{largest_density_mean:.8f} "
            f"{largest_density_std:.8f} "
            f"{largest_population_mean:.8f} "
            f"{largest_population_std:.8f}\n"
        )

        f.write(
            f"Second_Largest "
            f"{second_mean:.8f} "
            f"{second_std:.8f} "
            f"{second_density_mean:.8f} "
            f"{second_density_std:.8f} "
            f"{second_population_mean:.8f} "
            f"{second_population_std:.8f}\n"
        )

    print(
        f"Summary saved to: "
        f"{args.summary}"
    )

    # ========================================================
    # PLOT ALL KDEs
    # ========================================================

    plt.figure(figsize=(9, 6))

    for filename in args.input_files:

        try:

            (
                distances,
                x,
                density,
                peak_values,
                peak_densities,
                peak_populations
            ) = analyze_file(
                filename,
                bandwidth=args.bandwidth,
                prominence=args.prominence
            )

            plt.plot(
                x,
                density,
                linewidth=2,
                label=os.path.basename(filename)
            )

        except Exception:
            continue

    plt.xlabel("Distance")
    plt.ylabel("Probability Density")

    plt.title(
        "KDE Probability Density Distributions"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        args.plot,
        dpi=300
    )

    plt.show()

    print(
        f"Plot saved to: {args.plot}"
    )


if __name__ == "__main__":
    main()
