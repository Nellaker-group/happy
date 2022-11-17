import json
from itertools import combinations

import typer
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

from happy.utils.utils import get_project_dir
from projects.placenta.results.labelbox.path_mapping import PATHOLOGIST_MAPPING


ALL_LABELBOX_LABELS = [
    "terminal_villi",
    "mature_intermediary_villi",
    "stem_villi",
    "villus_sprout",
    "anchoring_villi",
    "chorion_amnion",
    "basal_plate_septa",
    "fibrin",
    "avascular_villi",
]
MODEL_LABELS = [
    "Terminal Villi",
    "Mature Intermediate Villi",
    "Stem Villi",
    "Villus Sprout",
    "Anchoring Villi",
    "Chorionic Plate",
    "Basal Plate/Septum",
    "Fibrin",
    "Avascular Villi",
]


def main(
    file_name: str = typer.Option(...),
    original_file_name: str = typer.Option(...),
    original_subregion_file_name: str = typer.Option(...),
    path_to_patches_file: str = None,
):
    # Process labelbox json file
    project_dir = get_project_dir("placenta")
    path_to_file = project_dir / "results" / "labelbox" / file_name
    with open(path_to_file, "r") as f:
        raw_data = json.load(f)
    df = _process_labelbox_data(raw_data)

    # process original annotation files
    path_to_original_file = project_dir / "results" / "labelbox" / original_file_name
    path_to_original_subregion_file = (
        project_dir / "results" / "labelbox" / original_subregion_file_name
    )
    with open(path_to_original_file, "r") as f:
        original_data = json.load(f)
    with open(path_to_original_subregion_file, "r") as f:
        original_subregion_data = json.load(f)
    original_df = _process_original_data(original_data, original_subregion_data)
    original_df.label = _map_to_labelbox_classes(original_df.label)

    # Setup datastructures for analysis bellow
    path_df = df[df["is_pathologist"] == 1]
    nonpath_df = df[df["is_pathologist"] == 0]
    path_predictions = get_images_and_labels_by_person(df, pathologist_only=True)
    nonpath_predictions = get_images_and_labels_by_person(df, pathologist_only=False)
    path_majority_df = get_majority_labels(path_df)
    combined_majority_df = combine_path_with_original(path_majority_df, original_df)

    if path_to_patches_file is not None:
        all_new_annots = pd.read_csv(
            project_dir / "graph_splits" / path_to_patches_file
        )
        all_new_annots.new_annot = _map_to_labelbox_classes(all_new_annots.new_annot)
        xs = []
        ys = []
        for image_name in combined_majority_df["image_name"]:
            image_splits = image_name.split("_")
            xs.append(int(image_splits[1].split("x")[1]))
            ys.append(int(image_splits[2].split("y")[1]))
        combined_majority_df["x"] = xs
        combined_majority_df["y"] = ys
        merged_df = pd.merge(combined_majority_df, all_new_annots)
        merged_df.drop(["x", "y", "width", "height"], axis=1, inplace=True)
        combined_majority_df = merged_df

    # total average time per tissue type
    time_per_tissue = time_per_tissue_type(df)
    time_per_tissue_mean = time_per_tissue.mean()

    # average time per tissue type for pathologists
    path_time_per_tissue = time_per_tissue_type(path_df)
    path_time_per_tissue_mean = path_time_per_tissue.mean()

    # average time per tissue type for non experts
    nonpath_time_per_tissue = time_per_tissue_type(nonpath_df)
    nonpath_time_per_tissue_mean = nonpath_time_per_tissue.mean()

    # counts of tissue types across pathologists
    path_tissue_counts = path_df["label"].value_counts()

    # counts of tissue types across non experts
    nonpath_tissue_counts = nonpath_df["label"].value_counts()

    # total kappa between pathologists
    path_kappas = get_total_kappa(path_predictions)
    print(f"Pathologist's kappa: {np.array(path_kappas).mean():.3f}")

    # Total kappa between pathologists and me when they are in agreement
    filtered_combined_df = combined_majority_df[
        combined_majority_df["num_unique"] <= 2
    ].reset_index(drop=True)
    original_to_path_kappa = cohen_kappa_score(
        filtered_combined_df["majority_class"],
        filtered_combined_df["original_label"],
    )
    print(f"Original to path majority kappa: {original_to_path_kappa:.3f}")

    if path_to_patches_file is not None:
        new_to_path_kappa = cohen_kappa_score(
            filtered_combined_df["majority_class"],
            filtered_combined_df["new_annot"],
        )
        print(f"New to path majority kappa: {new_to_path_kappa:.3f}")

    for label in ALL_LABELBOX_LABELS:
        # kappa per tissue type as one vs rest per pathologist combination average
        mean_label_kappa = kappa_for_one_tissue(path_predictions, label)
        print(f"{label} avg kappa: {mean_label_kappa:.3f}")

        label_majority_mask = combined_majority_df["majority_class"] == label
        label_filtered_df = combined_majority_df[label_majority_mask]
        num_images = len(label_filtered_df)
        if num_images != 0:
            # custom certainty for each tissue type between pathologists
            path_prediction_stats(
                combined_majority_df, label_filtered_df, num_images, label
            )

            # agreement for each tissue type between original and pathologists
            original_label_mask = combined_majority_df["original_label"] == label
            original_path_kappa = cohen_kappa_score(
                label_majority_mask, original_label_mask
            )
            print(f"{label} original to path kappa: {original_path_kappa:.3f}")

            # agreement original vs pathologists stats
            original_vs_path_stats(combined_majority_df, label, num_images)
        else:
            print(f"{label}: no agreement between pathologists")

    # Most commonly labelled together tissue types
    pathologist_confusion(combined_majority_df)

    # Pathologist votes against original 'ground truth' confusion matrix
    original_to_path_confusion(combined_majority_df)

    # Pathologist votes against new 'ground truth' confusion matrix
    original_to_path_confusion(combined_majority_df, new_labels=True)


def kappa_for_one_tissue(path_predictions, label):
    indices = list(range(len(path_predictions)))
    indices_combinations = list(combinations(indices, 2))
    all_kappas = []
    for comb in indices_combinations:
        label_map_1 = path_predictions[comb[0]]["label"] == label
        label_map_2 = path_predictions[comb[1]]["label"] == label
        kappa = cohen_kappa_score(label_map_1, label_map_2)
        if np.isnan(kappa):
            kappa = 0
        all_kappas.append(kappa)
    mean_label_kappa = np.array(all_kappas).mean()
    return mean_label_kappa


def path_prediction_stats(combined_majority_df, label_filtered_df, num_images, label):
    num_predictions = label_filtered_df["num_unique"].sum()
    other_majority_df = combined_majority_df[
        combined_majority_df["majority_class"] != label
    ]
    other_majority_df_disagreement = other_majority_df[
        other_majority_df["num_unique"] != 1
    ]
    num_label_in_other_majority = (
        other_majority_df_disagreement["all_preds"]
        .apply(lambda x: x.count(label))
        .sum()
    )
    certainty = (num_images / (num_predictions + num_label_in_other_majority)) * 100
    print(
        f"{label} images in majority {num_images} | "
        f"num disagreed with majority {num_predictions - num_images} | "
        f"num labels in other majorities {num_label_in_other_majority} | "
        f"certainty {certainty:.1f}%"
    )


def original_vs_path_stats(combined_majority_df, label, num_images):
    num_pathologists = len(combined_majority_df["all_preds"][0])
    original_filtered_df = combined_majority_df[
        combined_majority_df["original_label"] == label
    ]
    num_matching_majority = (original_filtered_df["majority_class"] == label).sum()
    count_matching = (
        "==" + original_filtered_df["all_preds"].str.join("==") + "=="
    ).str.count(f"={label}=")
    print(
        f"num of original images matching majority image: "
        f"{num_matching_majority}/{num_images} "
        f"({(num_matching_majority / num_images) * 100:.0f}%)"
    )
    print(
        f"num of original matching at least one pathologist: "
        f"{(count_matching >= 1).sum()}/20 "
        f"({((count_matching >= 1).sum() / 20) * 100:.0f}%)"
    )
    print(
        f"num of original matching at least two pathologist: "
        f"{(count_matching >= 2).sum()}/20 "
        f"({((count_matching >= 2).sum() / 20) * 100:.0f}%)"
    )
    print(
        f"num of original matching at least three pathologist: "
        f"{(count_matching >= 3).sum()}/20 "
        f"({((count_matching >= 3).sum() / 20) * 100:.0f}%)"
    )
    print(
        f"num of original matching num of total pathologist votes: "
        f"{count_matching.sum()}/{20 * num_pathologists} "
        f"({count_matching.sum() / (20 * num_pathologists) * 100:.0f}%)"
    )


def pathologist_confusion(combined_majority_df):
    labelbox_labels = ALL_LABELBOX_LABELS + ["not_listed", "unclear"]
    model_labels = MODEL_LABELS + ["Not Listed", "Unclear"]
    cm = np.empty((len(labelbox_labels), len(labelbox_labels)))
    for i, label in enumerate(labelbox_labels):
        label_majority_mask = combined_majority_df["majority_class"] == label
        label_filtered_df = combined_majority_df[label_majority_mask]
        for j, other_label in enumerate(labelbox_labels):
            num_matching = (
                label_filtered_df["all_preds"]
                .apply(lambda x: x.count(other_label))
                .sum()
            )
            cm[i, j] = num_matching
    row_labels = []
    unique_counts = cm.sum(axis=1)
    total_counts = cm.sum()
    label_proportions = ((unique_counts / total_counts) * 100).round(2)
    for i, label in enumerate(model_labels):
        row_labels.append(f"{label} ({label_proportions[i]}%)")
    unique_counts = cm.sum(axis=1)
    cm_df = (
        pd.DataFrame(
            cm / unique_counts[:, None], columns=model_labels, index=model_labels
        )
        .fillna(0)
        .astype(float)
    )
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.1)
    plt.rcParams["figure.dpi"] = 600
    ax = sns.heatmap(
        cm_df, annot=True, cmap="Blues", square=True, cbar=False, fmt=".2f"
    )
    ax.hlines([9], *ax.get_xlim())
    ax.vlines([9], *ax.get_ylim())
    plt.ylabel("Majority Label")
    plt.xlabel("Proportion of All Labels")
    plt.tight_layout()
    plt.savefig("pathologist_confusion.png")


def original_to_path_confusion(combined_majority_df, new_labels=False, prop=True):
    cm = np.empty((len(ALL_LABELBOX_LABELS), len(ALL_LABELBOX_LABELS)))
    for i, label in enumerate(ALL_LABELBOX_LABELS):
        if new_labels:
            label_original_mask = combined_majority_df["new_annot"] == label
        else:
            label_original_mask = combined_majority_df["original_label"] == label
        label_filtered_df = combined_majority_df[label_original_mask]
        for j, other_label in enumerate(ALL_LABELBOX_LABELS):
            num_matching = (
                label_filtered_df["all_preds"]
                .apply(lambda x: x.count(other_label))
                .sum()
            )
            cm[i, j] = num_matching
    if prop:
        fmt = ".2f"
        row_labels = []
        unique_counts = cm.sum(axis=1)
        total_counts = cm.sum()
        label_proportions = ((unique_counts / total_counts) * 100).round(2)
        for i, label in enumerate(MODEL_LABELS):
            row_labels.append(f"{label} ({label_proportions[i]}%)")
        unique_counts = cm.sum(axis=1)
        cm_df = (
            pd.DataFrame(
                cm / unique_counts[:, None], columns=MODEL_LABELS, index=MODEL_LABELS
            )
            .fillna(0)
            .astype(float)
        )
    else:
        fmt = "g"
        cm_df = pd.DataFrame(cm, columns=MODEL_LABELS, index=MODEL_LABELS)
    plt.figure(figsize=(10, 8))
    plt.rcParams["figure.dpi"] = 600
    sns.heatmap(cm_df, annot=True, cmap="Blues", square=True, cbar=False, fmt=fmt)
    plt.yticks(rotation=0)
    plt.ylabel("Original Tissue Structure Annotation")
    plt.xlabel("Proportion of All Pathologist Labels")
    plt.tight_layout()
    if new_labels:
        plt.savefig("original_to_pathologist_confusion_new.png")
    else:
        plt.savefig("original_vs_pathologist_confusion.png")


def time_per_tissue_type(df):
    return df.groupby("label")["duration"].mean()


def get_images_and_labels_by_person(df, pathologist_only):
    predictions = []
    for person in df["pathologist"].unique():
        if pathologist_only:
            if PATHOLOGIST_MAPPING[person] == 1:
                predictions.append(get_predictions_by_pathologist(df, person))
        else:
            predictions.append(get_predictions_by_pathologist(df, person))
    return predictions


def get_predictions_by_pathologist(df, pathologist):
    predicted_labels = df[df["pathologist"] == pathologist][["image_name", "label"]]
    predicted_labels.sort_values(by="image_name", ignore_index=True, inplace=True)
    return predicted_labels


def get_total_kappa(all_predictions):
    indices = list(range(len(all_predictions)))
    indices_combinations = list(combinations(indices, 2))
    kappas = []
    for comb in indices_combinations:
        kappas.append(
            cohen_kappa_score(
                all_predictions[comb[0]]["label"], all_predictions[comb[1]]["label"]
            )
        )
    return kappas


def get_majority_labels(df):
    unique_counts_df = (
        df.groupby("image_name")["label"].nunique().reset_index(drop=False)
    )
    unique_counts_df["majority_class"] = (
        df.groupby("image_name")["label"].agg(lambda x: pd.Series.mode(x).iat[0]).values
    )
    unique_counts_df.loc[
        unique_counts_df["majority_class"].apply(lambda x: isinstance(x, np.ndarray)),
        "majority_class",
    ] = "none"
    unique_counts_df.columns = ["image_name", "num_unique", "majority_class"]
    unique_counts_df["all_preds"] = (
        df.groupby("image_name").label.apply(list).reset_index()["label"]
    )
    unique_counts_df = unique_counts_df.sort_values(by=["majority_class"])
    return unique_counts_df


def combine_path_with_original(path_df, original_df):
    # Align indicies by image_name for original and grouped path dfs
    sort_path = path_df.sort_values("image_name")
    sort_original = original_df.sort_values("image_name").reset_index(drop=True)
    sort_path["original_label"] = sort_original["label"]
    return sort_path


def _process_labelbox_data(raw_data):
    cleaned_data = []
    for d in raw_data:
        if not d["Skipped"]:
            cleaned_d = {
                "image_name": d["External ID"],
                "pathologist": d["Created By"],
                "label": d["Label"]["classifications"][0]["answer"]["value"],
                "duration": d["Seconds to Label"],
                "has_comment": d["Has Open Issues"],
                "is_pathologist": PATHOLOGIST_MAPPING[d["Created By"]],
            }
            cleaned_data.append(cleaned_d)
    return pd.DataFrame(cleaned_data)


def _process_original_data(original_raw_data, original_subset_raw_data):
    cleaned_original_data = []
    for d in original_raw_data:
        cleaned_d = {
            "image_name": d["image_name"],
            "label": d["tissue_type"],
        }
        cleaned_original_data.append(cleaned_d)
    original_df = pd.DataFrame(cleaned_original_data)

    cleaned_original_subregion_data = []
    for d in original_subset_raw_data:
        cleaned_d = {
            "image_name": d["image_name"],
            "label": d["tissue_type"],
        }
        cleaned_original_subregion_data.append(cleaned_d)
    original_subregion_df = pd.DataFrame(cleaned_original_subregion_data)
    return pd.concat([original_df, original_subregion_df])


def _map_to_labelbox_classes(df_column):
    label_mapping = {
        "Sprout": "villus_sprout",
        "TVilli": "terminal_villi",
        "MIVilli": "mature_intermediary_villi",
        "AVilli": "anchoring_villi",
        "SVilli": "stem_villi",
        "Chorion": "chorion_amnion",
        "Maternal": "basal_plate_septa",
        "Fibrin": "fibrin",
        "Avascular": "avascular_villi",
    }
    df_column = df_column.map(label_mapping)
    return df_column


if __name__ == "__main__":
    typer.run(main)
