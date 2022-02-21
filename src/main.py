"""
Created by Philippenko, 16th February 2022.
"""
from matplotlib import pyplot as plt

from src.Dataset import Dataset
from src.models.DenseNet import DenseNet
from src.models.LeNet import LeNet

from src.Trainer import Training
from src.models.Resnet import Resnet20
from src.models.VGG import VGG11

BATCH_SIZE = 128


def run_training(network):

    ### Getting the dataset
    dataset = Dataset(batch_size = BATCH_SIZE, dataset_name="cifar10")
    train_loader, full_train_loader, test_loader, time_data_loading = dataset.get_loaders()

    ### Intialization of the trainer
    trainer = Training(network, train_loader, full_train_loader, test_loader, id="cifar10")

    with open(trainer.logs_file, 'a') as f:
        print("Time loading datasets: {:.2e}s".format(time_data_loading), file=f)

    ### Running the training
    run_logger = trainer.run_training()

    return run_logger


def plot(*args):
    ### Plot 1 is for train loss, Plot 2 is for test accuracy.
    fig, axes = plt.subplots(2, figsize=(8, 7))
    for logger in args:
        __plot__(logger.train_losses, axes[0], yaxis="Train loss", label=logger.id)
        __plot__(logger.test_accuracies, axes[1], yaxis="Test accuracy", label=logger.id)
    axes[0].set_xlabel(r"epoch", fontsize=15)
    plt.savefig('{0}.eps'.format("./pictures/perf"), format='eps')
    plt.close()


def __plot__(values, ax, yaxis, label):
    ax.plot(values, label=label)
    ax.set_ylabel(yaxis, fontsize=15)
    ax.grid(True)


if __name__ == '__main__':

    res_densenet = run_training(DenseNet)
    res_resnet20 = run_training(Resnet20)
    res_vgg11 = run_training(VGG11)
    res_lenet = run_training(LeNet)

    plot(res_densenet, res_resnet20, res_vgg11, res_lenet)



