#!/usr/bin/env python3

import os
import gc

import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt


# ===================================================
# USER INPUTS
# ===================================================

top_A = "../dark/rep1/strip_only_CA.strip_all_heavy.dark_tyr_solv.prmtop"

top_B = "../white/rep1/strip_CA_nobox.strip_all_heavy.white_tyr_solv.prmtop"

sysA_trajs = [
    "../dark/rep1/dark_only_CA_after_200ns.nc",
    "../dark/rep2/dark_only_CA_after_200ns.nc",
    "../dark/rep3/dark_only_CA_after_200ns.nc"
]

sysB_trajs = [
    "../white/rep1/white_only_CA_after200ns.nc",
    "../white/rep2/white_only_CA_after200ns.nc",
    "../white/rep3/white_only_CA_after200ns.nc"
]

ref_pdb = "../dark/rep1/dark_tyr_ref.pdb"

selection = "resid 4-432 and name CA"


# ===================================================
# PCA SETTINGS
# ===================================================

N_PCS = 30

# Analyze every Nth frame
FRAME_STRIDE = 10

# Number of frames processed at a time
BATCH_SIZE = 500

# Use float32 to reduce memory
DTYPE = np.float32


# ===================================================
# PLOT SETTINGS
# ===================================================

PC1_MIN = -45
PC1_MAX = 20

PC2_MIN = -30
PC2_MAX = 30


rep_colors = {
    1: "blue",
    2: "orange",
    3: "green"
}


sys_names = {
    "A": "dark",
    "B": "white"
}


PLOT_PADDING = 0.05


# ===================================================
# OUTPUT DIRECTORY
# ===================================================

output_dir = "pca_results"

os.makedirs(
    output_dir,
    exist_ok=True
)


# ===================================================
# LOAD REFERENCE
# ===================================================

print()
print("=" * 60)
print("LOADING REFERENCE")
print("=" * 60)

u_ref = mda.Universe(
    ref_pdb
)

ca_ref = u_ref.select_atoms(
    selection
)

n_atoms = ca_ref.n_atoms

n_coordinates = n_atoms * 3

print(
    f"Reference CA atoms: {n_atoms}"
)

print(
    f"Coordinates per frame: {n_coordinates}"
)


# ===================================================
# SYSTEM DEFINITIONS
# ===================================================

all_systems = [
    ("A", top_A, sysA_trajs),
    ("B", top_B, sysB_trajs)
]


# ===================================================
# CHECK FILES
# ===================================================

print()
print("=" * 60)
print("CHECKING INPUT FILES")
print("=" * 60)

for sys_label, topology, trajectories in all_systems:

    if not os.path.exists(topology):

        raise FileNotFoundError(
            f"Topology not found:\n{topology}"
        )

    for rep_id, trajectory in enumerate(
        trajectories,
        start=1
    ):

        if not os.path.exists(trajectory):

            raise FileNotFoundError(
                f"Trajectory not found:\n{trajectory}"
            )

        print(
            f"{sys_names[sys_label]} "
            f"rep{rep_id}: OK"
        )


# ===================================================
# FUNCTION:
# READ TRAJECTORY IN BATCHES
# ===================================================

def coordinate_batches(
    topology,
    trajectory,
    system_name,
    rep_id
):

    """
    Reads a trajectory without loading it entirely
    into memory.

    Every FRAME_STRIDE-th frame is used.

    Each returned batch contains aligned C-alpha
    coordinates flattened as:

        frame × (residue × XYZ)
    """

    print()
    print(
        f"Opening {system_name} "
        f"rep{rep_id}"
    )

    u = mda.Universe(
        topology,
        trajectory
    )

    ca = u.select_atoms(
        selection
    )

    if ca.n_atoms != n_atoms:

        raise ValueError(
            f"{system_name} rep{rep_id}: "
            f"found {ca.n_atoms} CA atoms; "
            f"expected {n_atoms}"
        )

    batch = np.empty(
        (
            BATCH_SIZE,
            n_coordinates
        ),
        dtype=DTYPE
    )

    batch_count = 0

    selected_frames = 0

    for frame_number, ts in enumerate(
        u.trajectory
    ):

        # --------------------------------------------
        # STRIDE
        # --------------------------------------------

        if frame_number % FRAME_STRIDE != 0:

            continue

        # --------------------------------------------
        # ALIGN FRAME
        # --------------------------------------------

        align.alignto(
            u,
            u_ref,
            select=selection
        )

        # --------------------------------------------
        # SAVE COORDINATES
        # --------------------------------------------

        batch[
            batch_count
        ] = ca.positions.reshape(-1)

        batch_count += 1

        selected_frames += 1

        # --------------------------------------------
        # RETURN FULL BATCH
        # --------------------------------------------

        if batch_count == BATCH_SIZE:

            yield batch.copy()

            batch_count = 0

    # -----------------------------------------------
    # RETURN FINAL PARTIAL BATCH
    # -----------------------------------------------

    if batch_count > 0:

        yield (
            batch[
                :batch_count
            ].copy()
        )

    print(
        f"Selected frames: "
        f"{selected_frames}"
    )

    del u
    del ca
    del batch

    gc.collect()


# ===================================================
# FIRST PASS:
# FIT COMBINED PCA
# ===================================================

print()
print("=" * 60)
print("FIRST PASS: FITTING COMBINED PCA")
print("=" * 60)

pca = IncrementalPCA(
    n_components=N_PCS,
    batch_size=BATCH_SIZE
)

total_frames = 0

for sys_label, topology, trajectories in all_systems:

    system_name = sys_names[
        sys_label
    ]

    for rep_id, trajectory in enumerate(
        trajectories,
        start=1
    ):

        for batch in coordinate_batches(
            topology,
            trajectory,
            system_name,
            rep_id
        ):

            pca.partial_fit(
                batch
            )

            total_frames += len(
                batch
            )

            print(
                f"Frames processed: "
                f"{total_frames}"
            )

            del batch

            gc.collect()


print()
print(
    "PCA fitting complete."
)


# ===================================================
# PCA DATA
# ===================================================

components = pca.components_

pca_mean = pca.mean_

explained_variance = (
    pca.explained_variance_ratio_
)

pcs = np.arange(
    1,
    N_PCS + 1
)


# ===================================================
# SAVE COMBINED PCA VARIANCE
# ===================================================

pd.DataFrame(
    {
        "PC": pcs,
        "Variance": explained_variance,
        "Cumulative": np.cumsum(
            explained_variance
        )
    }
).to_csv(
    os.path.join(
        output_dir,
        "combined_variance.csv"
    ),
    index=False
)


# ===================================================
# RESHAPE PCA COMPONENTS
# ===================================================

components_reshaped = components.reshape(
    N_PCS,
    n_atoms,
    3
)


# ===================================================
# COMBINED RESIDUE CONTRIBUTIONS
#
# This uses the PCA eigenvectors directly:
#
# contribution =
# X^2 + Y^2 + Z^2
# ===================================================

print()
print("=" * 60)
print("CALCULATING COMBINED RESIDUE CONTRIBUTIONS")
print("=" * 60)

residue_contrib_combined = np.zeros(
    (
        N_PCS,
        n_atoms
    ),
    dtype=np.float64
)

for pc in range(N_PCS):

    residue_contrib_combined[pc] = np.sum(
        components_reshaped[pc] ** 2,
        axis=1
    )


# ===================================================
# SECOND PASS:
# PROJECT TRAJECTORIES
# ===================================================

print()
print("=" * 60)
print("SECOND PASS: PROJECTING TRAJECTORIES")
print("=" * 60)


# ===================================================
# SYSTEM-LEVEL PC STATISTICS
# ===================================================

dark_sum = np.zeros(
    N_PCS,
    dtype=np.float64
)

dark_sum_sq = np.zeros(
    N_PCS,
    dtype=np.float64
)

dark_count = 0


white_sum = np.zeros(
    N_PCS,
    dtype=np.float64
)

white_sum_sq = np.zeros(
    N_PCS,
    dtype=np.float64
)

white_count = 0


# ===================================================
# RESIDUE-LEVEL PROJECTED VARIANCE
#
# Shape:
#
# PC × residue
# ===================================================

residue_contrib_dark_sum = np.zeros(
    (
        N_PCS,
        n_atoms
    ),
    dtype=np.float64
)

residue_contrib_white_sum = np.zeros(
    (
        N_PCS,
        n_atoms
    ),
    dtype=np.float64
)


# ===================================================
# PROJECT EACH TRAJECTORY
# ===================================================

for sys_label, topology, trajectories in all_systems:

    system_name = sys_names[
        sys_label
    ]

    for rep_id, trajectory in enumerate(
        trajectories,
        start=1
    ):

        print()
        print(
            f"Processing "
            f"{system_name} rep{rep_id}"
        )

        rep_pc1 = []

        rep_pc2 = []

        for batch in coordinate_batches(
            topology,
            trajectory,
            system_name,
            rep_id
        ):

            # ========================================
            # PCA PROJECTION
            # ========================================

            projected = pca.transform(
                batch
            )

            # ========================================
            # SAVE PC1 / PC2
            # ========================================

            rep_pc1.append(
                projected[:, 0].astype(
                    DTYPE
                )
            )

            rep_pc2.append(
                projected[:, 1].astype(
                    DTYPE
                )
            )

            # ========================================
            # SYSTEM-LEVEL VARIANCE
            # ========================================

            batch_sum = np.sum(
                projected,
                axis=0,
                dtype=np.float64
            )

            batch_sum_sq = np.sum(
                projected ** 2,
                axis=0,
                dtype=np.float64
            )

            if sys_label == "A":

                dark_sum += batch_sum

                dark_sum_sq += batch_sum_sq

                dark_count += len(
                    projected
                )

            else:

                white_sum += batch_sum

                white_sum_sq += batch_sum_sq

                white_count += len(
                    projected
                )

            # ========================================
            # CENTERED COORDINATES
            # ========================================

            centered = (
                batch.astype(
                    np.float64
                )
                -
                pca_mean
            )

            centered = centered.reshape(
                len(batch),
                n_atoms,
                3
            )

            # ========================================
            # RESIDUE × PC PROJECTIONS
            #
            # residue_projection:
            #
            # frame × residue × PC
            # ========================================

            residue_projection = np.einsum(
                "bar,par->bap",
                centered,
                components_reshaped,
                optimize=True
            )

            # ========================================
            # SQUARED RESIDUE PROJECTIONS
            # ========================================

            residue_squared = (
                residue_projection ** 2
            )

            # ========================================
            # AVERAGING IS DONE LATER
            #
            # Sum over frames here.
            #
            # Result:
            #
            # residue × PC
            # ========================================

            residue_batch_sum = np.sum(
                residue_squared,
                axis=0,
                dtype=np.float64
            )

            # ========================================
            # TRANSPOSE
            #
            # PC × residue
            # ========================================

            residue_batch_sum = (
                residue_batch_sum.T
            )

            # ========================================
            # ADD TO SYSTEM TOTAL
            # ========================================

            if sys_label == "A":

                residue_contrib_dark_sum += (
                    residue_batch_sum
                )

            else:

                residue_contrib_white_sum += (
                    residue_batch_sum
                )

            # ========================================
            # FREE MEMORY
            # ========================================

            del batch
            del projected
            del centered
            del residue_projection
            del residue_squared
            del residue_batch_sum

            gc.collect()

        # =================================================
        # SAVE REPLICATE PC COORDINATES
        # =================================================

        rep_pc1 = np.concatenate(
            rep_pc1
        )

        rep_pc2 = np.concatenate(
            rep_pc2
        )

        pc_filename = os.path.join(
            output_dir,
            f"pc_coordinates_"
            f"{system_name}_rep{rep_id}.csv"
        )

        pd.DataFrame(
            {
                "Frame": np.arange(
                    len(rep_pc1)
                ),
                "PC1": rep_pc1,
                "PC2": rep_pc2
            }
        ).to_csv(
            pc_filename,
            index=False
        )

        print(
            f"Saved: {pc_filename}"
        )

        del rep_pc1
        del rep_pc2

        gc.collect()


# ===================================================
# SYSTEM-LEVEL VARIANCE
# ===================================================

print()
print("=" * 60)
print("CALCULATING SYSTEM VARIANCE")
print("=" * 60)


dark_mean = (
    dark_sum /
    dark_count
)

dark_variance = (
    dark_sum_sq /
    dark_count
    -
    dark_mean ** 2
)


white_mean = (
    white_sum /
    white_count
)

white_variance = (
    white_sum_sq /
    white_count
    -
    white_mean ** 2
)


# Avoid tiny negative numerical values
dark_variance = np.maximum(
    dark_variance,
    0
)

white_variance = np.maximum(
    white_variance,
    0
)


# ===================================================
# NORMALIZED SYSTEM VARIANCE
#
# This normalizes dark + white together
# across all PCs.
# ===================================================

total_system_variance = (
    np.sum(dark_variance)
    +
    np.sum(white_variance)
)

dark_variance_fraction = (
    dark_variance /
    total_system_variance
)

white_variance_fraction = (
    white_variance /
    total_system_variance
)


# ===================================================
# ALSO CALCULATE WITHIN-SYSTEM PC VARIANCE
#
# For each PC:
#
# dark PC variance =
# dark_variance / (dark + white)
#
# This tells you what fraction of the
# variance of a given PC comes from dark.
# ===================================================

pc_total_variance = (
    dark_variance +
    white_variance
)

dark_pc_fraction = np.divide(
    dark_variance,
    pc_total_variance,
    out=np.zeros_like(dark_variance),
    where=pc_total_variance > 0
)

white_pc_fraction = np.divide(
    white_variance,
    pc_total_variance,
    out=np.zeros_like(white_variance),
    where=pc_total_variance > 0
)


# ===================================================
# SAVE SYSTEM VARIANCE
# ===================================================

df_variance = pd.DataFrame(
    {
        "PC": pcs,

        "Combined_ExplainedVariance":
            explained_variance,

        "Dark_Variance":
            dark_variance,

        "White_Variance":
            white_variance,

        "Dark_Fraction_Total":
            dark_variance_fraction,

        "White_Fraction_Total":
            white_variance_fraction,

        "Dark_Fraction_of_PC":
            dark_pc_fraction,

        "White_Fraction_of_PC":
            white_pc_fraction
    }
)


df_variance.to_csv(
    os.path.join(
        output_dir,
        "system_variance.csv"
    ),
    index=False
)


# ===================================================
# PRINT VARIANCE SUMMARY
# ===================================================

print()
print(
    "System variance summary:"
)

print()

for i in range(N_PCS):

    print(
        f"PC{i+1:2d}: "
        f"Dark = "
        f"{dark_variance[i]:.6f}, "
        f"White = "
        f"{white_variance[i]:.6f}, "
        f"Dark fraction of PC = "
        f"{dark_pc_fraction[i]*100:.2f}%, "
        f"White fraction of PC = "
        f"{white_pc_fraction[i]*100:.2f}%"
    )


# ===================================================
# RESIDUE CONTRIBUTIONS
#
# Divide by number of frames to get
# average squared projected contribution.
# ===================================================

print()
print("=" * 60)
print("CALCULATING RESIDUE PROJECTED VARIANCE")
print("=" * 60)


residue_contrib_dark = (
    residue_contrib_dark_sum /
    dark_count
)

residue_contrib_white = (
    residue_contrib_white_sum /
    white_count
)


# ===================================================
# COMBINED TRAJECTORY-BASED RESIDUE CONTRIBUTION
#
# We calculate this separately from the
# eigenvector-only contribution.
# ===================================================

# Since dark and white have potentially different
# numbers of frames, weight them according to their
# actual frame counts.

residue_contrib_combined_trajectory = (
    residue_contrib_dark_sum
    +
    residue_contrib_white_sum
) / (
    dark_count +
    white_count
)


# ===================================================
# SAVE RAW RESIDUE CONTRIBUTIONS
# ===================================================

def save_residue_csv(
    data,
    filename
):

    df = pd.DataFrame(
        data
    )

    df.index = [
        f"PC{i+1}"
        for i in range(N_PCS)
    ]

    df.columns = [
        f"Residue{i+1}"
        for i in range(n_atoms)
    ]

    df.to_csv(
        filename
    )


# ---------------------------------------------------
# Combined eigenvector contribution
# ---------------------------------------------------

save_residue_csv(
    residue_contrib_combined,
    os.path.join(
        output_dir,
        "residue_contributions_combined_eigenvector.csv"
    )
)


# ---------------------------------------------------
# Combined trajectory contribution
# ---------------------------------------------------

save_residue_csv(
    residue_contrib_combined_trajectory,
    os.path.join(
        output_dir,
        "residue_contributions_combined_trajectory.csv"
    )
)


# ---------------------------------------------------
# Dark
# ---------------------------------------------------

save_residue_csv(
    residue_contrib_dark,
    os.path.join(
        output_dir,
        "residue_contributions_dark.csv"
    )
)


# ---------------------------------------------------
# White
# ---------------------------------------------------

save_residue_csv(
    residue_contrib_white,
    os.path.join(
        output_dir,
        "residue_contributions_white.csv"
    )
)


# ===================================================
# RESIDUE PERCENTAGE CONTRIBUTIONS
#
# Each PC is normalized across residues.
#
# Therefore:
#
# sum(residue percentages) = 100%
#
# for every PC.
# ===================================================

def residue_percentages(
    data
):

    denominator = (
        data.sum(
            axis=1,
            keepdims=True
        )
    )

    return np.divide(
        data,
        denominator,
        out=np.zeros_like(data),
        where=denominator > 0
    ) * 100.0


# ===================================================
# CALCULATE PERCENTAGES
# ===================================================

residue_percent_combined_eigenvector = (
    residue_percentages(
        residue_contrib_combined
    )
)


residue_percent_combined_trajectory = (
    residue_percentages(
        residue_contrib_combined_trajectory
    )
)


residue_percent_dark = (
    residue_percentages(
        residue_contrib_dark
    )
)


residue_percent_white = (
    residue_percentages(
        residue_contrib_white
    )
)


# ===================================================
# SAVE PERCENTAGE FILES
# ===================================================

def save_residue_percent_csv(
    data,
    filename
):

    df = pd.DataFrame(
        data
    )

    df.index = [
        f"PC{i+1}"
        for i in range(N_PCS)
    ]

    df.columns = [
        f"Residue{i+1}"
        for i in range(n_atoms)
    ]

    df.to_csv(
        filename
    )


# ---------------------------------------------------
# Combined eigenvector
# ---------------------------------------------------

save_residue_percent_csv(
    residue_percent_combined_eigenvector,
    os.path.join(
        output_dir,
        "residue_percentage_contributions_combined_eigenvector.csv"
    )
)


# ---------------------------------------------------
# Combined trajectory
# ---------------------------------------------------

save_residue_percent_csv(
    residue_percent_combined_trajectory,
    os.path.join(
        output_dir,
        "residue_percentage_contributions_combined_trajectory.csv"
    )
)


# ---------------------------------------------------
# Dark
# ---------------------------------------------------

save_residue_percent_csv(
    residue_percent_dark,
    os.path.join(
        output_dir,
        "residue_percentage_contributions_dark.csv"
    )
)


# ---------------------------------------------------
# White
# ---------------------------------------------------

save_residue_percent_csv(
    residue_percent_white,
    os.path.join(
        output_dir,
        "residue_percentage_contributions_white.csv"
    )
)


# ===================================================
# VERIFY RESIDUE PERCENTAGES
# ===================================================

print()
print("=" * 60)
print("VERIFYING RESIDUE PERCENTAGES")
print("=" * 60)

print()

print(
    "Dark:"
)

print(
    "PC1 sum = "
    f"{residue_percent_dark[0].sum():.6f}%"
)

print(
    "PC2 sum = "
    f"{residue_percent_dark[1].sum():.6f}%"
)

print()

print(
    "White:"
)

print(
    "PC1 sum = "
    f"{residue_percent_white[0].sum():.6f}%"
)

print(
    "PC2 sum = "
    f"{residue_percent_white[1].sum():.6f}%"
)


# ===================================================
# FIND GLOBAL PCA LIMITS
# ===================================================

print()
print("=" * 60)
print("CALCULATING GLOBAL PCA LIMITS")
print("=" * 60)


global_pc1_min = np.inf
global_pc1_max = -np.inf

global_pc2_min = np.inf
global_pc2_max = -np.inf


for sys_label in ["A", "B"]:

    system_name = sys_names[
        sys_label
    ]

    for rep_id in [1, 2, 3]:

        filename = os.path.join(
            output_dir,
            f"pc_coordinates_"
            f"{system_name}_rep{rep_id}.csv"
        )

        data = pd.read_csv(
            filename,
            usecols=[
                "PC1",
                "PC2"
            ]
        )

        global_pc1_min = min(
            global_pc1_min,
            data["PC1"].min()
        )

        global_pc1_max = max(
            global_pc1_max,
            data["PC1"].max()
        )

        global_pc2_min = min(
            global_pc2_min,
            data["PC2"].min()
        )

        global_pc2_max = max(
            global_pc2_max,
            data["PC2"].max()
        )

        del data

        gc.collect()


# ===================================================
# ADD PADDING
# ===================================================

pc1_range = (
    global_pc1_max -
    global_pc1_min
)

pc2_range = (
    global_pc2_max -
    global_pc2_min
)

pc1_padding = (
    pc1_range *
    PLOT_PADDING
)

pc2_padding = (
    pc2_range *
    PLOT_PADDING
)

GLOBAL_X_MIN = (
    global_pc1_min -
    pc1_padding
)

GLOBAL_X_MAX = (
    global_pc1_max +
    pc1_padding
)

GLOBAL_Y_MIN = (
    global_pc2_min -
    pc2_padding
)

GLOBAL_Y_MAX = (
    global_pc2_max +
    pc2_padding
)


print()
print(
    "THE FOLLOWING LIMITS WILL BE USED "
    "FOR EVERY SINGLE PLOT:"
)

print()

print(
    f"X axis: "
    f"{GLOBAL_X_MIN:.6f} "
    f"to "
    f"{GLOBAL_X_MAX:.6f}"
)

print(
    f"Y axis: "
    f"{GLOBAL_Y_MIN:.6f} "
    f"to "
    f"{GLOBAL_Y_MAX:.6f}"
)


# ===================================================
# LEGEND SETTINGS
# ===================================================

LEGEND_FONTSIZE = 14

LEGEND_FRAMEON = True

LEGEND_FACECOLOR = "white"

LEGEND_EDGECOLOR = "black"

LEGEND_FRAMEALPHA = 1.0

LEGEND_FANCYBOX = False


# ===================================================
# PLOT FUNCTION
# ===================================================

def plot_single_replicate(
    system_name,
    rep_id
):

    filename = os.path.join(
        output_dir,
        f"pc_coordinates_"
        f"{system_name}_rep{rep_id}.csv"
    )

    data = pd.read_csv(
        filename,
        usecols=[
            "PC1",
            "PC2"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.scatter(
        data["PC1"],
        data["PC2"],
        s=60,
        marker="o",
        color=rep_colors[rep_id],
        edgecolor="black",
        linewidth=0.75,
        alpha=1.0,
        label=f"{system_name.capitalize()} Replicate {rep_id}"
    )

    ax.set_xlim(
        PC1_MIN,
        PC1_MAX
    )

    ax.set_ylim(
        PC2_MIN,
        PC2_MAX
    )

    ax.set_xlabel(
        "PC1",
        fontsize=16
    )

    ax.set_ylabel(
        "PC2",
        fontsize=16
    )

    ax.tick_params(
        axis="both",
        labelsize=16
    )

    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle="-",
        linewidth=0.7,
        alpha=1.0
    )

    ax.legend(
        fontsize=LEGEND_FONTSIZE,
        frameon=LEGEND_FRAMEON,
        facecolor=LEGEND_FACECOLOR,
        edgecolor=LEGEND_EDGECOLOR,
        framealpha=LEGEND_FRAMEALPHA,
        fancybox=LEGEND_FANCYBOX
    )

    fig.tight_layout()

    output_file = os.path.join(
        output_dir,
        f"pca_{system_name}_rep{rep_id}.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
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
# MAKE INDIVIDUAL REPLICATE PLOTS
# ===================================================

print()
print("=" * 60)
print("MAKING INDIVIDUAL REPLICATE PLOTS")
print("=" * 60)


for system_name in [
    "dark",
    "white"
]:

    for rep_id in [
        1,
        2,
        3
    ]:

        plot_single_replicate(
            system_name,
            rep_id
        )


# ===================================================
# MAKE ALL-REPLICATE SYSTEM PLOTS
# ===================================================

print()
print("=" * 60)
print("MAKING ALL-REPLICATE PLOTS")
print("=" * 60)


for system_name in [
    "dark",
    "white"
]:

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    for rep_id in [
        1,
        2,
        3
    ]:

        filename = os.path.join(
            output_dir,
            f"pc_coordinates_"
            f"{system_name}_rep{rep_id}.csv"
        )

        data = pd.read_csv(
            filename,
            usecols=[
                "PC1",
                "PC2"
            ]
        )

        ax.scatter(
            data["PC1"],
            data["PC2"],
            s=60,
            marker="o",
            color=rep_colors[rep_id],
            edgecolor="black",
            linewidth=0.75,
            alpha=1.0,
            label=f"{system_name.capitalize()} Replicate {rep_id}"
        )

        del data

        gc.collect()

    ax.set_xlim(
        PC1_MIN,
        PC1_MAX
    )

    ax.set_ylim(
        PC2_MIN,
        PC2_MAX
    )

    ax.set_xlabel(
        "PC1",
        fontsize=16
    )

    ax.set_ylabel(
        "PC2",
        fontsize=16
    )

    ax.set_title(
        f"{system_name.capitalize()} PCA",
        fontsize=16
    )

    ax.tick_params(
        axis="both",
        labelsize=16
    )

    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle="-",
        linewidth=0.7,
        alpha=1.0
    )

    ax.legend(
        fontsize=LEGEND_FONTSIZE,
        frameon=LEGEND_FRAMEON,
        facecolor=LEGEND_FACECOLOR,
        edgecolor=LEGEND_EDGECOLOR,
        framealpha=LEGEND_FRAMEALPHA,
        fancybox=LEGEND_FANCYBOX
    )

    fig.tight_layout()

    output_file = os.path.join(
        output_dir,
        f"pca_{system_name}_all_replicates.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output_file}"
    )


# ===================================================
# FINAL SUMMARY
# ===================================================

print()
print("=" * 60)
print("PCA ANALYSIS COMPLETE")
print("=" * 60)

print(
    f"Frame stride = {FRAME_STRIDE}"
)

print(
    f"Batch size = {BATCH_SIZE}"
)

print(
    f"Number of PCs = {N_PCS}"
)

print()

print(
    "COMMON AXES FOR ALL PLOTS:"
)

print(
    f"PC1: "
    f"{GLOBAL_X_MIN:.6f} "
    f"to "
    f"{GLOBAL_X_MAX:.6f}"
)

print(
    f"PC2: "
    f"{GLOBAL_Y_MIN:.6f} "
    f"to "
    f"{GLOBAL_Y_MAX:.6f}"
)

print()

print(
    "IMPORTANT OUTPUT FILES:"
)

print(
    "  system_variance.csv"
)

print(
    "  residue_contributions_dark.csv"
)

print(
    "  residue_contributions_white.csv"
)

print(
    "  residue_percentage_contributions_dark.csv"
)

print(
    "  residue_percentage_contributions_white.csv"
)

print(
    "  residue_percentage_contributions_combined_trajectory.csv"
)

print()

print(
    f"Results saved in: "
    f"{output_dir}/"
)
