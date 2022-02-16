"""
Created by Philippenko, 16th February 2022.
"""
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
from torchvision.utils import make_grid

from DLRun import DeepLearningRun
from Models import LeNet
from PathDataset import get_path_to_datasets

from pympler import asizeof

NB_EPOCH = 10
LEARNING_RATE = 0.1
MOMENTUM = 0.9
BATCH_SIZE = 128


class Training:

    def __init__(self, network, train_loader, test_loader) -> None:
        super().__init__()
        self.seed_everything()

        ############## Logs file ##############
        self.logs_file = "logs.txt"

        ############## Train/Test data loader ##############
        self.train_loader = train_loader
        self.test_loader = test_loader

        ############## Device: GPU or CPU ##############
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ############## Global model ##############
        self.global_model = network().to(self.device)

        ############## Settings for cuda ##############
        if self.device == 'cuda':
            self.global_model = torch.nn.DataParallel(self.global_model)
        cudnn.benchmark = True if torch.cuda.is_available() else False

        ############## Algorithm used for optimization ##############
        self.optimizer = optim.SGD(self.global_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

        ############## Loss function ##############
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        ############## Class that stores all train/test losses and the test accuracies ##############
        self.run = DeepLearningRun()

        with open(self.logs_file, 'a') as f:
            print(f"=========== NEW RUN ===========", file=f)
            print("Device :", self.device, file=f)
            print("Size of the global model: {:.2e} bits".format(asizeof.asizeof(self.global_model)), file=f)
            print("Size of the optimizer: {:.2e} bits".format(asizeof.asizeof(self.optimizer)), file=f)

    def seed_everything(self):
        # Seed
        seed = 25
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def run_one_epoch(self):
        running_loss = 0.0
        train_loader_iter = iter(self.train_loader)
        nb_inner_iterations = len(self.train_loader)
        for _ in range(int(nb_inner_iterations)):

            ### Getting the next batch a putting it to the right device.
            data, target = next(train_loader_iter)
            data, target = data.to(self.device), target.to(self.device)

            ### Set to zero the parameter gradients
            self.optimizer.zero_grad()

            ### Forward pass
            outputs = self.global_model(data)

            ### Compute the loss
            loss = self.criterion(outputs, target)

            ### Backward pass
            loss.backward()

            ### Optimizer step
            self.optimizer.step()

            ### Update running loss
            running_loss += loss.item()

        return running_loss / nb_inner_iterations


    def compute_test_accuracy_and_loss(self) -> (int, int):
        """Compute test loss/accuracy."""
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                data, target = data

                ### Calculate the output
                output = self.global_model(data)

                ### Computing the test loss
                loss = self.criterion(output, target)
                test_loss += loss.item()

                ### Computing the test accuracy
                # (The class with the highest energy is what we choose as prediction)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        test_accuracy = 100 * correct / total
        test_loss = test_loss / len(self.test_loader)
        return test_loss, test_accuracy

if __name__ == '__main__':

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    batch_size = BATCH_SIZE

    ### We set pin_memory to True if we push the data from the CPU to the GPU, this speed-up the transfer.
    ### Image shape: torch.Size([128, 3, 32, 32])
    trainset = torchvision.datasets.CIFAR10(root=get_path_to_datasets() + '/data', train=True,
                                            download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, pin_memory=True,
                                               shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root=get_path_to_datasets() + './data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, pin_memory=True,
                                             shuffle=False, num_workers=4)

    trainer = Training(LeNet, train_loader, testloader)

    train_losses = []
    test_losses = []
    test_accuracies = []
    for epoch in range(NB_EPOCH):  # loop over the dataset multiple times

        train_loss = trainer.run_one_epoch()

        test_loss, test_accuracy = trainer.compute_test_accuracy_and_loss()

        # train_loss = self.compute_train_loss()
        # test_loss, test_accuracy = self.compute_test_accuracy_and_loss()
        # self.run.update_run(train_loss, test_loss, test_accuracy)


        train_losses.append(train_loss)
        with open(trainer.logs_file, 'a') as f:
            print(f'[Epoch {epoch + 1}] train loss: {train_loss :.3f}\t test loss: {test_loss :.3f}\t '
                  f'test accuracy: {test_accuracy :.3f}', file=f)


    print('Finished Training')


