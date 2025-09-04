import warnings

import umap
import umap.plot
import pandas as pd
import numpy as np
from bokeh.plotting import output_file
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# temporary fix for datashader version bug. Should be fixed by new datashader release.
np.warnings = warnings


def plot_interactive(
    plot_name, slide_name, organ, predictions, confidence, coords, mapper
):
    output_file(plot_name, title=f"UMAP Embeddings of Slide {slide_name}")

    label_colours = {cell.id: cell.colour for cell in organ.cells}
    label_ids = {cell.id: cell.label for cell in organ.cells}

    df = pd.DataFrame(
        {
            "pred": predictions,
            "confidence": confidence,
            "x_": coords[:, 0],
            "y_": coords[:, 1],
        }
    )
    df["pred"] = df.pred.map(label_ids)

    return umap.plot.interactive(
        mapper,
        labels=predictions,
        color_key=label_colours,
        interactive_text_search=True,
        hover_data=df,
        point_size=2,
    )


def plot_umap(organ, predictions, mapper, tissue=False):
    if tissue:
        colours_dict = {tissue.label: tissue.colour for tissue in organ.tissues}
        colours_dict.pop("Unlabelled")
        colours_dict.pop("MVilli")
        colours_dict.pop("ImIVilli")
        colours_dict.pop("Inflam")
        predictions_labelled = np.array(
            [organ.tissues[pred].label for pred in predictions]
        )
    else:
        colours_dict = {cell.label: cell.colour for cell in organ.cells}
        predictions_labelled = np.array(
            [organ.cells[pred].label for pred in predictions]
        )
    return umap.plot.points(mapper, labels=predictions_labelled, color_key=colours_dict)


def plot_3d(organ, result, predictions, custom_colours=None):
    matplotlib.use("TkAgg")
    sns.set(style="white", context="poster", rc={"figure.figsize": (14, 10)})
    if not custom_colours:
        custom_colours = np.array([cell.colour for cell in organ.cell])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        result[:, 0], result[:, 1], result[:, 2], c=custom_colours[predictions], s=1
    )
