import torch
import umap
import umap.plot
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import numpy as np

from analysis.embeddings.plots import plot_umap


def generate_umap(
    model_type, model, x, edge_index, organ, predictions, run_path, plot_name
):
    graph_embeddings = get_graph_embeddings(model_type, model, x, edge_index)
    fitted_umap = fit_umap(graph_embeddings)
    plot_cell_graph_umap(organ, predictions, fitted_umap, run_path, plot_name)


def plot_cell_graph_umap(organ, predictions, mapper, save_dir, plot_name):
    plot = plot_umap(organ, predictions, mapper)
    print(f"saving umap to {save_dir / plot_name}")
    plot.figure.savefig(save_dir / plot_name)
    plt.close(plot.figure)


@torch.no_grad()
def get_graph_embeddings(model_type, model, x, edge_index):
    print("Generating node embeddings")
    model.eval()
    if model_type == "graphsage":
        out = model.full_forward(x, edge_index).cpu()
    elif model_type == "infomax":
        out = model.encoder.full_forward(x, edge_index).cpu()
    else:
        raise ValueError(f"No such model type implemented: {model_type}")
    return out


def fit_umap(graph_embeddings):
    print("Fitting UMAP to embeddings")
    reducer = umap.UMAP(random_state=42, verbose=True, min_dist=0.0, n_neighbors=30)
    return reducer.fit(graph_embeddings)


def fit_clustering(num_clusters, graph_embeddings, clustering_method, mapper=None):
    if clustering_method == "kmeans":
        cluster_method = KMeans(n_clusters=num_clusters).fit(graph_embeddings)
    elif clustering_method == "dbscan":
        cluster_method = DBSCAN(eps=0.1).fit(graph_embeddings)
    elif clustering_method == "umap":
        graph_embeddings = mapper.transform(graph_embeddings)
        cluster_method = KMeans(n_clusters=num_clusters).fit(graph_embeddings)
    else:
        raise ValueError(f"No such clustering method: {clustering_method}")
    labels = cluster_method.predict(graph_embeddings)
    return labels, cluster_method


def plot_tissue_umap(
    organ, mapper, plot_name, save_dir, predicted_labels
):
    colours_dict = {tissue.label: tissue.colour for tissue in organ.tissues}
    labels = np.array([organ.tissues[pred].label for pred in predicted_labels])

    filtered_colour_dict = colours_dict.copy()
    for key in colours_dict.keys():
        if key not in labels:
            filtered_colour_dict.pop(key)

    plot = umap.plot.points(mapper, labels=labels, color_key=filtered_colour_dict)
    plot_name = f"tissue_{plot_name}_umap.png"
    plot.figure.savefig(save_dir / plot_name)
    print(f"Clustered UMAP saved to {save_dir / plot_name}")
    plt.close(plot.figure)
