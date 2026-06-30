from abc import ABC, abstractmethod

import pandas as pd

from happy.train.utils import plot_confusion_matrix


class _Appender(ABC):
    @abstractmethod
    def log_batch_loss(self, batch_count, loss):
        pass

    @abstractmethod
    def log_ap(self, split_name, epoch_num, ap):
        pass

    @abstractmethod
    def log_precision(self, split_name, epoch_num, precision):
        pass

    @abstractmethod
    def log_recall(self, split_name, epoch_num, recall):
        pass

    @abstractmethod
    def log_f1(self, split_name, epoch_num, f1):
        pass

    @abstractmethod
    def log_empty(self, split_name, epoch_num, num_empty):
        pass

    @abstractmethod
    def log_accuracy(self, split_name, epoch_num, accuracy):
        pass

    @abstractmethod
    def log_loss(self, split_name, epoch_num, loss):
        pass

    @abstractmethod
    def log_confusion_matrix(self, cm, dataset_name, save_dir):
        pass


class Console(_Appender):
    def log_batch_loss(self, batch_count, loss):
        pass

    def log_ap(self, split_name, epoch_num, ap):
        print(f"{split_name} AP: {ap}")

    def log_precision(self, split_name, epoch_num, precision):
        print(f"{split_name} Precision: {precision}")

    def log_recall(self, split_name, epoch_num, recall):
        print(f"{split_name} Recall: {recall}")

    def log_f1(self, split_name, epoch_num, f1):
        print(f"{split_name} F1: {f1}")

    def log_empty(self, split_name, epoch_num, num_empty):
        print(f"Number of predictions in empty images: {num_empty}")

    def log_accuracy(self, split_name, epoch_num, accuracy):
        print(f"{split_name} accuracy: {accuracy}")

    def log_loss(self, split_name, epoch_num, loss):
        print(f"{split_name} loss: {loss}")

    def log_confusion_matrix(self, cm, dataset_name, save_dir):
        print(f"{dataset_name} confusion matrix:")
        print(cm)


class File(_Appender):
    def __init__(self, dataset_names, metrics):
        self.train_stats = self._setup_train_stats(dataset_names, metrics)

    def log_batch_loss(self, batch_count, loss):
        pass

    def log_ap(self, split_name, epoch_num, ap):
        self._add_to_train_stats(epoch_num, split_name, "AP", ap)

    def log_precision(self, split_name, epoch_num, precision):
        self._add_to_train_stats(epoch_num, split_name, "Precision", precision)

    def log_recall(self, split_name, epoch_num, recall):
        self._add_to_train_stats(epoch_num, split_name, "Recall", recall)

    def log_f1(self, split_name, epoch_num, f1):
        self._add_to_train_stats(epoch_num, split_name, "F1", f1)

    def log_empty(self, split_name, epoch_num, num_empty):
        pass

    def log_accuracy(self, split_name, epoch_num, accuracy):
        self._add_to_train_stats(epoch_num, split_name, "accuracy", accuracy)

    def log_loss(self, split_name, epoch_num, loss):
        self._add_to_train_stats(epoch_num, split_name, "loss", loss)
        
    def log_confusion_matrix(self, cm, dataset_name, save_dir):
        plot_confusion_matrix(cm, dataset_name, save_dir)

    def _setup_train_stats(self, dataset_names, metrics):
        columns = []
        for name in dataset_names:
            for metric in metrics:
                col = f"{name}_{metric}"
                columns.append(col)
        return pd.DataFrame(columns=columns)

    def _add_to_train_stats(self, epoch_num, dataset_name, metric_name, metric):
        column_name = f"{dataset_name}_{metric_name}"
        if not epoch_num in self.train_stats.index:
            row = pd.Series([metric], index=[column_name])
            self.train_stats.loc[len(self.train_stats)] = row
        else:
            self.train_stats.loc[epoch_num][column_name] = metric
