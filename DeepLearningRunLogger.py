"""
Created by Philippenko, 16th February 2022.
"""
import numpy as np
from matplotlib import pyplot as plt


class DeepLearningRunLogger:
    """Gathers all important information compute during a Deep Learning run."""

    def __init__(self, id: str) -> None:
        super().__init__()
        # self.parameters = parameters
        self.id = id
        self.train_losses = []
        self.test_losses = []
        self.test_accuracies = []

    def update_run(self, train_loss, test_loss, test_acc):
        """Updates train/test losses and test accuracy."""
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_acc)

    def plot(self):
        ### Plot 1 is for train loss, Plot 2 is for test accuracy.
        fig, axes = plt.subplots(2, figsize=(8, 7))
        self.__plot__(self.train_losses, axes[0], yaxis="Train loss")
        self.__plot__(self.test_accuracies, axes[1], yaxis="Test accuracy")
        axes[0].set_xlabel(r"epoch", fontsize=15)
        if self.id:
            plt.savefig('{0}.eps'.format("./pictures/" + self.id), format='eps')
            plt.close()
        else:
            plt.show()

    def __plot__(self, values, ax, yaxis):
        ax.plot(values)
        ax.set_ylabel(yaxis, fontsize=15)
        ax.grid(True)