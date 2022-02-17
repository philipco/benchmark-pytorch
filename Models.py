"""
Created by Philippenko, 16th February 2022.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """From https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py."""
    def __init__(self):
        super(LeNet, self).__init__()
        self.output_size = 10
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.output_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.tanh(self.conv1(x))
        out = F.avg_pool2d(out, 2)
        out = self.relu(self.conv2(out))
        out = F.avg_pool2d(out, 2)
        # out = out.view(out.size(0), -1)
        out = self.flatten(out)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out
