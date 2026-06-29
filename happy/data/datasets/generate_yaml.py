import yaml
import os

def generate_dataset_yaml(output_path, dataset_root, class_to_id):
    # Sort classes by index to ensure correct order
    names = [None] * len(class_to_id)
    for cls_name, idx in class_to_id.items():
        names[idx] = cls_name

    data = {
        "path": os.path.abspath(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names
    }

    with open(output_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"dataset.yaml written to {output_path}")


if __name__ == "__main__":
    class_to_id = {
    "nucleus": 0}

    generate_dataset_yaml(
        output_path="projects/multiorgan/dataset/nuclei/ovary/ovary_nuclei.yaml",
        dataset_root="projects/multiorgan/dataset/nuclei/ovary",
        class_to_id=class_to_id
    )
    #TODO: not harded coded