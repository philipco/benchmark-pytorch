"""
Created by Philippenko, 17th February 2022.
"""
from datetime import datetime
import random

import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim
from pympler import asizeof
from torch.backends import cudnn

from src.Timer import Timer
from src.DeepLearningRunLogger import DeepLearningRunLogger

NB_EPOCH = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9

class Training:

    def __init__(self, network, train_loader, full_train_loader, test_loader, id: str) -> None:
        super().__init__()
        self.seed_everything()
        self.id = id

        self.timer = Timer()
        self.timer.start()

        ############## Logs file ##############
        self.logs_file = "logs.txt"

        ############## Train/Test dataset loader ##############
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.full_train_loader = full_train_loader

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
        self.run_logger = DeepLearningRunLogger(id = self.id)
        self.timer.stop()

        with open(self.logs_file, 'a') as f:
            print(f"============================= NEW RUN " + datetime.now().strftime("%d/%m/%Y at %H:%M:%S") +
                  " =============================", file=f)
            print("learning_rate -> {0}, momentum -> {1}, model -> {2}"
                  .format(LEARNING_RATE, MOMENTUM, type(self.global_model).__name__), file=f)
            print("Device :", self.device, file=f)
            print("Size of the global model: {:.2e} bits".format(asizeof.asizeof(self.global_model)), file=f)
            print("Size of the optimizer: {:.2e} bits".format(asizeof.asizeof(self.optimizer)), file=f)
            print("Time of initialization: {:.2e}s".format(self.timer.time), file=f)

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

    def __update_run__(self, train_loss, test_loss, test_acc):
        self.run_logger.update_run(train_loss, test_loss, test_acc)

    def run_training(self):

        ### Initialization of the loss/accuracy
        train_loss = self.__compute_train_loss__()
        test_loss, test_accuracy = self.__compute_test_accuracy_and_loss__()
        self.__update_run__(train_loss, test_loss, test_accuracy)

        for epoch in range(NB_EPOCH):
            self.timer.start()

            ### Updating the model and computing the train loss
            self.__run_one_epoch__()

            ### Computing the train loss on the full dataset with the new model
            train_loss = self.__compute_train_loss__()

            ### Computing the test loss/accuracy
            test_loss, test_accuracy = self.__compute_test_accuracy_and_loss__()

            ### Update the run with this new epoch
            self.__update_run__(train_loss, test_loss, test_accuracy)

            self.timer.stop()

            ### Save key informations in the logs file
            with open(self.logs_file, 'a') as f:
                print(f'[Epoch {epoch + 1}] train loss: {train_loss :.3f}\t test loss: {test_loss :.3f}\t '
                      f'test accuracy: {test_accuracy :.2f}% \t time: {self.timer.time :.0f}s', file=f)

        return self.run_logger

    def __run_one_epoch__(self):
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

    def __compute_train_loss__(self) -> (int, int):
        """Compute train loss on the full dataset using a batcof size 6000."""
        train_loss = 0.0
        with torch.no_grad():
            for data in self.full_train_loader:
                data, target = data
                data, target = data.to(self.device), target.to(self.device)

                ### Calculate the output
                output = self.global_model(data)

                ### Computing the test loss
                loss = self.criterion(output, target)
                train_loss += loss.item()

        train_loss = train_loss / len(self.full_train_loader)
        return train_loss

    def __compute_test_accuracy_and_loss__(self) -> (int, int):
        """Compute test loss/accuracy."""
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                data, target = data
                data, target = data.to(self.device), target.to(self.device)

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