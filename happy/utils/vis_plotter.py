from visdom import Visdom
import numpy as np


class VisdomLinePlotter():
    """Plots to a Visdom server"""

    def __init__(self, env_name="main", port=8998):
        self.viz = Visdom(port=port)
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x_label, y_label, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(
                X=np.array([x, x]),
                Y=np.array([y, y]),
                env=self.env,
                opts=dict(
                    legend=[split_name],
                    title=title_name,
                    xlabel=x_label,
                    ylabel=y_label,
                ),
            )
        else:
            self.viz.line(
                X=np.array([x]),
                Y=np.array([y]),
                env=self.env,
                win=self.plots[var_name],
                name=split_name,
                update="append",
            )
