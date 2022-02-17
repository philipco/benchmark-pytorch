"""
Created by Philippenko, 17th February 2022.
"""
import torch
import torchvision
import torchvision.transforms as transforms

from PathDataset import get_path_to_datasets
from Timer import Timer


class Dataset:

    def __init__(self, batch_size, dataset_name = "cifar10") -> None:
        super().__init__()
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.dataset_class = self.__get_dataset_class__()

    def get_loaders(self):
        timer = Timer()
        timer.start()

        ### We set pin_memory to True if we push the dataset from the CPU to the GPU, this speed-up the transfer.
        pin_memory = True if torch.cuda.is_available() else False

        ### Get train/test transformers to preprocess data
        transform_train, transform_test = self.__data_transfomer__()

        ### Get train loader
        train_set = self.dataset_class(root=get_path_to_datasets() + '/dataset', train=True,
                                       download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, pin_memory=pin_memory,
                                                   shuffle=True, num_workers=4)
        full_train_loader = torch.utils.data.DataLoader(train_set, batch_size=6000, pin_memory=pin_memory,
                                                   shuffle=True, num_workers=4)

        ### Get test loader
        test_set = self.dataset_class(root=get_path_to_datasets() + './dataset', train=False,
                                      download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, pin_memory=pin_memory,
                                                  shuffle=False, num_workers=4)
        timer.stop()
        return train_loader, full_train_loader, test_loader, timer.time

    def __get_dataset_class__(self):
        """Get the class of the given dataset to later load it."""
        if self.dataset_name == "cifar10":
            return torchvision.datasets.CIFAR10
        else:
            raise ValueError("The given dataset is not recognized.")

    def __data_transfomer__(self):

        if self.dataset_name == "cifar10":
            ### Image shape: torch.Size([128, 3, 32, 32])

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

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
        else:
            raise ValueError("The given dataset is not recognized.")

        return transform_train, transform_test