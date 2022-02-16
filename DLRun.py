"""
Created by Philippenko, 16th February 2022.
"""

class DeepLearningRun:
    """Gathers all important information compute during a Deep Learning run."""

    def __init__(self) -> None:
        super().__init__()
        # self.parameters = parameters
        self.train_losses = []
        self.test_losses = []
        self.test_accuracies = []

    def update_run(self, train_loss, test_loss, test_acc):
        """Updates train/test losses and test accuracy."""
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_acc)