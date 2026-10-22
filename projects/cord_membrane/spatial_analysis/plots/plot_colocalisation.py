from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import typer

from happy.organs import get_organ
from happy.utils.utils import get_project_dir
from projects.cord_membrane.spatial_analysis.plots.style import (
    BAND,
    INK,
    LINE,
    MUTED,
    style_axes,
)


def main(
    run_id: int = typer.Option(...),
    organ_name: str = typer.Option(..., help="placenta_cord or placenta_membrane"),
    project_name: str = "cord_membrane",
    tissue: Optional[str] = None,
):
    """Plot leukocyte colocalisation with each target cell type for one run.

    One panel per target cell type, all on the same axes: the observed K over the
    median K when leukocytes are scattered at random (1 means random), with the
    2.5-97.5% range of the random scatterings as a band. Reads the curves saved by
    run_colocalisation and saves run_N_k_ratio.png next to them.

    Run after run_colocalisation:
        python -m projects.cord_membrane.spatial_analysis.plots.plot_colocalisation \\
            --run-id 1 --organ-name placenta_membrane

    Args:
        run_id: eval run the colocalisation was run on
        organ_name: placenta_cord or placenta_membrane (membrane rolls)
        project_name: project dir containing the results
        tissue: tissue label if run_colocalisation was run with --tissue
    """
    organ = get_organ(organ_name)
    results_dir = get_project_dir(project_name) / "results" / "colocalisation"
    name = f"run_{run_id}_{tissue}" if tissue else f"run_{run_id}"
    curves = pd.read_csv(results_dir / f"{name}_curves.csv")
    summary = pd.read_csv(results_dir / f"{name}_summary.csv").set_index("target")
    targets = list(dict.fromkeys(curves["target"]))

    fig, axes = plt.subplots(
        1,
        len(targets),
        figsize=(2.8 * len(targets), 3.4),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for ax, target in zip(axes[0], targets):
        curve = curves[curves["target"] == target]
        median = curve["random_median_k_um2"]
        ax.fill_between(
            curve["distance_um"],
            curve["random_lower_k_um2"] / median,
            curve["random_upper_k_um2"] / median,
            color=BAND,
            linewidth=0,
            label="Random scatter (95% range)",
        )
        ax.axhline(1, color=MUTED, linewidth=0.8)
        ax.plot(curve["distance_um"], curve["k_ratio"], color=LINE, linewidth=1.5, label="Observed")
        ax.set_title(
            f"{organ.cell_by_label(target).name}\n{summary.loc[target, 'targets']:,} cells",
            fontsize=9,
            color=INK,
        )
        ax.set_xlabel("Distance (µm)", fontsize=8, color=INK)
        ax.set_ylim(bottom=0)
        style_axes(ax)
    axes[0][0].set_ylabel("K ratio (observed / random)", fontsize=8, color=INK)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=8)
    region = tissue or "all tissue"
    fig.suptitle(
        f"Leukocyte colocalisation, run {run_id} "
        f"({summary['leukocytes'].iloc[0]:,} leukocytes, {region})",
        fontsize=10,
        color=INK,
    )

    save_path = results_dir / f"{name}_k_ratio.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    typer.run(main)
