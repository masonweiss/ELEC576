# Mason Weiss
# ELEC 576 - Introduction to Deep Learning
# Prof. Ankit Patel, Rice University
# Due: November 6, 2024
# adapted from code snippet assignment_2_part1_cifar10_skeleton.py

# Step 1: Pytorch and Training Metadata

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path
import matplotlib.pyplot as plt

batch_size = 64
test_batch_size = 1000
epochs = 10
lr = 0.001
try_cuda = True
seed = 1000
logging_interval = 10  # how many batches to wait before logging
logging_dir = None

INPUT_SIZE = 28

# 1) setting up the logging

datetime_str = datetime.now().strftime('%b%d_%H-%M-%S')

if logging_dir is None:
    runs_dir = Path("./") / Path(f"runs/part3/")
    runs_dir.mkdir(exist_ok=True)

    logging_dir = runs_dir / Path(f"{datetime_str}")

    logging_dir.mkdir(exist_ok=True)
    logging_dir = str(logging_dir.absolute())

writer = SummaryWriter(log_dir=logging_dir)

# deciding whether to send to the cpu or not if available
if torch.cuda.is_available() and try_cuda:
    cuda = True
    torch.cuda.manual_seed(seed)
else:
    cuda = False
    torch.manual_seed(seed)

# Step 2: Data Setup

# Setting up data - taken from my previous A1 submission for MNIST CNN preparation
transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.01307,), (0.3081,))
])

# from cnn part of A1
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
nclasses = len(torch.unique(train_dataset.train_labels))

# plot one example
print(train_dataset.train_data.size())  # (60000, 28, 28)
print(train_dataset.train_labels.size())  # (60000)
plt.imshow(train_dataset.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_dataset.train_labels[0])
plt.show()

# step 3 creating model moved to after class declaration to iterate through models


class Net(nn.Module):
    def __init__(self, model_type, num_hidden, num_classes):
        super(Net, self).__init__()

        self.model_type = model_type  # RNN, LSTM, or GRU
        if self.model_type == 'LSTM':
            print("==== BUILDING LSTM MODEL ====")
            self.base_model = nn.LSTM(input_size=INPUT_SIZE, hidden_size=num_hidden, num_layers=2, batch_first=True)
        elif self.model_type == 'GRU':
            print("==== BUILDING GRU MODEL ====")
            self.base_model = nn.GRU(input_size=INPUT_SIZE, hidden_size=num_hidden, num_layers=2, batch_first=True)
        else:
            print("==== BUILDING RNN MODEL ====")
            self.base_model = nn.RNN(input_size=INPUT_SIZE, hidden_size=num_hidden, num_layers=2, batch_first=True)
        self.out = nn.Linear(in_features=num_hidden, out_features=num_classes)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, hidden = self.base_model(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


# Step 4: Train/Test


def train(epoch):
    model.train()

    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        data = data.view(-1, 28, 28)

        optimizer.zero_grad()
        output = model(data)  # forward pass
        loss = criterion(output, target)

        loss.backward()  # backprop
        optimizer.step()

        # same code as from A1 for MNIST CNN (for easier comparison)
        if batch_idx % logging_interval == 0:
            print(f'TRAIN - Epoch: {epoch + 1} of {epochs} | Batch {batch_idx} of {len(train_loader)} | Loss: {round(loss.item(), 5)}')
            writer.add_scalar(f'Train/Loss/{mname}', loss.item(), epoch * len(train_loader) + batch_idx)


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():  # disable computing gradients on testing
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()

            data = data.view(-1, 28, 28)

            output = model(data)  # forward
            test_loss += criterion(output, target).item()  # access numeric value

            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f'TEST - Epoch: {epoch + 1} of {epochs} | Accuracy: {round(correct / len(test_dataset) * 100, 5)} % | Loss: {round(test_loss, 5)}\n')

    # log loss and accuracy to tensorboard
    writer.add_scalar(f'Test/Loss/{mname}', test_loss, epoch)
    writer.add_scalar(f'Test/Accuracy/{mname}', correct / len(test_dataset) * 100, epoch)


# Step 3: Creating the Model

nhidden = 64
for mname in ('RNN', 'LSTM', 'GRU'):
    model = Net(model_type=mname, num_hidden=nhidden, num_classes=nclasses)

    if cuda:
        model.cuda()

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    epochs = 10

    # training loop
    for epoch in range(epochs):
        train(epoch)
        test(epoch)

writer.close()

# %tensorboard --logdir runs/part3