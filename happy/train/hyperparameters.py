from pathlib import Path

import pandas as pd


class Hyperparameters:
    def __init__(
        self,
        exp_name,
        annot_dir,
        dataset_names,
        model_name,
        pre_trained,
        epochs,
        batch,
        learning_rate,
        init_from_inc,
        frozen,
    ):
        self.exp_name = exp_name
        self.annot_dir = annot_dir
        self.dataset_names = dataset_names
        self.model_name = model_name
        self.pre_trained = pre_trained
        self.epochs = epochs
        self.batch = batch
        self.learning_rate = learning_rate
        self.init_from_inc = init_from_inc
        self.frozen = frozen

    def to_csv(self, path):
        d = {
            "exp_name": [self.exp_name],
            "annot_dir": [self.annot_dir],
            "dataset_names": [self.dataset_names],
            "model_name": [self.model_name],
            "pre_trained": [self.pre_trained],
            "epochs": [self.epochs],
            "batch": [self.batch],
            "learning_rate": [self.learning_rate],
            "init_from_inc": [self.init_from_inc],
            "frozen": [self.frozen],
        }
        hp_df = pd.DataFrame(data=d)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        hp_df.to_csv(path / "params.csv", index=False)
