import collections

from happy.logger.appenders import Console, File, Visdom


class Logger:
    def __init__(self, dataset_names, metrics, vis=True, file=True):
        self.loss_hist = collections.deque(maxlen=500)
        self.appenders = self._get_appenders(vis, file, dataset_names, metrics)

    def log_ap(self, split_name, epoch_num, ap):
        for a in self.appenders:
            self.appenders[a].log_ap(split_name, epoch_num, ap)

    def log_precision(self, split_name, epoch_num, precision):
        for a in self.appenders:
            self.appenders[a].log_precision(split_name, epoch_num, precision)

    def log_recall(self, split_name, epoch_num, recall):
        for a in self.appenders:
            self.appenders[a].log_recall(split_name, epoch_num, recall)

    def log_f1(self, split_name, epoch_num, f1):
        for a in self.appenders:
            self.appenders[a].log_f1(split_name, epoch_num, f1)

    def log_empty(self, split_name, epoch_num, num_empty):
        for a in self.appenders:
            self.appenders[a].log_empty(split_name, epoch_num, num_empty)

    def log_accuracy(self, split_name, epoch_num, accuracy):
        for a in self.appenders:
            self.appenders[a].log_accuracy(split_name, epoch_num, round(accuracy, 4))

    def log_loss(self, split_name, epoch_num, loss):
        for a in self.appenders:
            self.appenders[a].log_loss(split_name, epoch_num, round(loss, 4))

    def log_batch_loss(self, batch_count, loss):
        for a in self.appenders:
            self.appenders[a].log_batch_loss(batch_count, round(loss, 4))

    def log_confusion_matrix(self, cm, dataset_name, save_dir):
        for a in self.appenders:
            self.appenders[a].log_confusion_matrix(cm, dataset_name, save_dir)

    def to_csv(self, save_path):
        file_appender = self.appenders["file"]
        file_appender.train_stats.to_csv(save_path)

    def _get_appenders(self, vis, file, dataset_names, metrics):
        appenders = {"console": Console()}
        if vis:
            appenders["visdom"] = Visdom()
        if file:
            appenders["file"] = File(dataset_names, metrics)
        return appenders
