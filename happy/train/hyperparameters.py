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

    @staticmethod
    def load_from_csv(path):
        hp_data = pd.read_csv(path)
        dict_hp_data = hp_data.to_dict(orient="rows")[0]

        hp = Hyperparameters(
            dict_hp_data["exp_name"],
            dict_hp_data["annot_dir"],
            dict_hp_data["dataset_names"],
            dict_hp_data["model_name"],
            dict_hp_data["pre_trained"],
            dict_hp_data["epochs"],
            dict_hp_data["batch"],
            dict_hp_data["learning_rate"],
            dict_hp_data["init_from_inc"],
            dict_hp_data["frozen"],
        )
        return hp

    def resolve_parser_overwrites(self, args):
        if args.exp_name:
            self.exp_name = args.exp_name
        if args.annot_dir:
            self.annot_dir = args.annot_dir
        if args.dataset_names:
            self.dataset_names = args.dataset_names
        if args.model_name:
            self.model_name = args.model_name
        if args.pre_trained:
            self.pre_trained = args.pre_trained
        if args.epochs:
            self.epochs = args.epochs
        if args.batch:
            self.batch = args.batch
        if args.learning_rate:
            self.learning_rate = args.learning_rate
        if args.init_from_inc:
            self.init_from_inc = args.init_from_inc
        if args.frozen:
            self.frozen = args.frozen
