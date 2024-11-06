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
from torchvision import datasets, transforms, io  # use io to take in images to tensors
from torch.utils.data import DataLoader, TensorDataset  # make TensorDataset

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# hyperparameters
batch_size = 128
epochs = 10
high_lr = 0.0010
low_lr = 0.0001  # implement linear learning rate decay
lr_peak_idx = 3  # ramp up lr from low to high before this epoch index, then decay after
try_cuda = True
seed = 1000

# Architecture
num_classes = 10

logging_interval = 10  # how many batches to wait before logging
logging_dir = None
grayscale = True
input_size = 28  # data input size

datetime_str = datetime.now().strftime('%b%d_%H-%M-%S')

if logging_dir is None:
    runs_dir = Path("./") / Path(f"runs/part1")
    runs_dir.mkdir(exist_ok = True)

    logging_dir = runs_dir / Path(f"{datetime_str}")

    logging_dir.mkdir(exist_ok = True)
    logging_dir = str(logging_dir.absolute())

writer = SummaryWriter(log_dir=logging_dir)
logging_dir = None

# deciding whether to send to the cpu or not if available
if torch.cuda.is_available() and try_cuda:
    cuda = True
    torch.cuda.manual_seed(seed)
else:
    cuda = False
    torch.manual_seed(seed)

"""# Step 2: Data Setup"""
# code from https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py
# implement pytorch transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # convert to grayscale
    transforms.ConvertImageDtype(torch.float)  # Convert to float
])

# Load training and test data
for dclass in ('Train', 'Test'):
    images = []  # list of images
    labels = []  # list of labels
    num_per_class = 1000 if dclass == 'Train' else 100
    for class_num in range(10):  # Assuming labels are from 0 to 9
        for img_num in range(num_per_class):
            img_path = f'CIFAR10/{dclass}/{str(class_num)}/Image{img_num:05}.png'
            img = io.read_image(img_path)  # decode image previously did not work
            img = transform(img)  # apply transform
            images.append(img)  # add image to list of images, need to reference
        labels += [class_num]*num_per_class

    if dclass == 'Train':
        train_dataset = TensorDataset(torch.stack(images), torch.tensor(labels))  # convert to a pytorch tensor object
    if dclass == 'Test':
        test_dataset = TensorDataset(torch.stack(images), torch.tensor(labels))

# load in randomly from files since they were stored by class
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

def check_data_loader_dim(loader):
    # Checking the dataset
    for images, labels in loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break

check_data_loader_dim(train_loader)
check_data_loader_dim(test_loader)

"""# 3) Creating the Model"""
layer_1_n_filters = 32
layer_2_n_filters = 64
fc_1_n_nodes = 1024
padding = "same"
kernel_size = 5
verbose = True

padding_factor = 2 if padding == "same" else 0
final_length = int((((input_size - kernel_size + 2*padding_factor + 1) / 2) - kernel_size + 2*padding_factor + 1) / 2)
# calculate the dimension of the output of the CNN stage before the MLP layers given the previous 2 convolutional
# layers with the current padding setting, kernel size and maxpooling

if verbose:
    print(f"final_length = {final_length}")


class LeNet5(nn.Module):

    def __init__(self, num_classes, grayscale=False):
        super(LeNet5, self).__init__()

        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        # implement 'same' padding to maintain edge size (padding will be 2 in this case)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=layer_1_n_filters, kernel_size=kernel_size, padding=padding_factor),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=layer_1_n_filters, out_channels=layer_2_n_filters, kernel_size=kernel_size, padding=padding_factor),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(final_length*final_length*layer_2_n_filters*in_channels, fc_1_n_nodes),
            nn.Tanh(),
            nn.Linear(fc_1_n_nodes, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # output will be (batch_size x layer_2_n_filters x final_length x final_length)
        x = x.view(-1, final_length*final_length*layer_2_n_filters*(3-self.grayscale*2))
        # in between CNN and MLP sections, flatten input while retaining batch structure
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

model = LeNet5(num_classes=num_classes, grayscale=True)

if cuda:
    model.cuda()


# Step 4: Train/Test Loop

def train(epoch):
    model.train()

    total_loss = 0
    correct = 0  # rubric considers training accuracy as a part of grade, so I've included it.

    if epoch < lr_peak_idx:
        lr = (high_lr-low_lr)*((epoch - lr_peak_idx)/lr_peak_idx)+high_lr
    else:
        lr = (low_lr-high_lr)*((epoch - lr_peak_idx)/(epochs - lr_peak_idx - 1))+high_lr  # implement LR decay
    if verbose:
        print(f'TRAIN - Learning Rate for epoch {epoch+1} of {epochs}: {round(lr,10)}')

    writer.add_scalar('Train/learning_rate', lr, epoch)  # log for easier visualization

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        logits, probas = model(data)  # forward

        # refer to prev assignment for training block construction
        loss = criterion(logits, target)  # loss computed on logits
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # compute training accuracy
        preds = torch.argmax(probas, dim=1)
        correct += torch.sum(torch.eq(target, preds)).item()  # compare true and pred, take count of where they are same

        # log at logging interval
        if batch_idx % logging_interval == 0:
            print(f'TRAIN - Epoch: {epoch+1} of {epochs} | Batch {batch_idx} of {len(train_loader)} | Loss: {round(loss.item(),5)}')

    print(f'TRAIN - Epoch: {epoch + 1} of {epochs} | Accuracy: {round(correct/100, 5)}% | Total Loss: {round(total_loss,5)}')
    # log training loss & accuracyat each epoch
    writer.add_scalar('Train/accuracy', round(correct/100, 5), epoch)
    writer.add_scalar('Train/loss', total_loss, epoch)


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(size_average = False)  # this throws a warning

    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()

        logits, probas = model(data)

        test_loss = test_loss + criterion(logits, target).item()  # loss numeric value is .item()

        preds = torch.argmax(probas, dim=1)
        correct += torch.sum(torch.eq(target, preds)).item()  # compare true and pred, take count of where they are same

        # compute activations at final test epoch
        if epoch + 1 == 1:
            with torch.no_grad():  # ensure gradients not being updated
                x = model.features[0](data)     # apply first conv layer
                conv1_activation = x.detach().numpy()     # save activation from first conv layer
                x = model.features[1](x)  # apply 1st activation
                x = model.features[2](x)        # apply maxpool
                x = model.features[3](x)        # apply second conv layer
                conv2_activation = x.detach().numpy()   # save activation from second conv layer

                for idx in range(len(conv1_activation)):
                    conv1_activations.append(conv1_activation[idx])
                    conv2_activations.append(conv2_activation[idx])
                    classes.append(np.array([preds[idx], target[idx]]))

    # log each epoch
    print(f'TEST - Epoch: {epoch+1} of {epochs} | Accuracy: {round(correct/1000 * 100,5)} % | Loss: {round(test_loss,5)}\n')

    writer.add_scalar('Test/loss', test_loss, epoch)
    writer.add_scalar('Test/accuracy', correct/10, epoch)


conv1_activations = []
conv2_activations = []
classes = []

for epoch in range(epochs):
    train(epoch)
    test(epoch)

writer.close()

# Visualize the trained network

# i: Visualize the first convolutional layer's weights
first_conv_layer = model.features[0]
filter_weights = first_conv_layer.weight.detach().numpy() # return tensor as numpy object

fig, axes = plt.subplots(ncols=len(filter_weights)//4, nrows=4, figsize=(len(filter_weights)/2, 8))

for f_idx in range(len(filter_weights)):
    f_weight = filter_weights[f_idx][0]  # depth of 1 -> grayscale
    axes[f_idx // 8, f_idx % 8].axis('off')
    axes[f_idx // 8, f_idx % 8].imshow(f_weight, cmap='gray')
    axes[f_idx // 8, f_idx % 8].set_title(f_idx, fontsize=16)

plt.savefig('figures/fig7_c1_features.png')
plt.show()

# ii: Show the statistics of the activations in the convolutional layers on test images
conv1_activations = np.array(conv1_activations)
conv2_activations = np.array(conv2_activations)
classes = np.array(classes)

# statistics for first activation:
print("Statistics for Output from First Conv Layer:")
print("\tMean:", np.round(np.mean(conv1_activations), 5))
print("\tMedian:", np.round(np.median(conv1_activations), 5))
print("\tMinimum:", np.round(np.min(conv1_activations), 5))
print("\tMaximum:", np.round(np.max(conv1_activations), 5))
print("\tStandard Deviation:", np.round(np.std(conv1_activations), 5), '\n')

print("Statistics for Output from Second Conv Layer:")
print("\tMean:", np.round(np.mean(conv2_activations), 5))
print("\tMedian:", np.round(np.median(conv2_activations), 5))
print("\tMinimum:", np.round(np.min(conv2_activations), 5))
print("\tMaximum:", np.round(np.max(conv2_activations), 5))
print("\tStandard Deviation:", np.round(np.std(conv2_activations), 5))

## iii (2b): Plot the feature map for a few sample outputs
samples = [30, 40, 65]

for sample_idx in range(len(samples)):  # plot feature maps for each sample
    for layer_idx in range(2):
        num_rows = 4*(layer_idx+1)  # for conv1, need 4 rows (32 maps), conv2 need 8 rows (64 maps)
        fig, axes = plt.subplots(ncols=8, nrows=num_rows, figsize=(16, 8*(layer_idx+1)))
        for plot_idx in range(32*(layer_idx+1)):
            if layer_idx == 0:
                feature_map = conv1_activations[samples[sample_idx]][plot_idx]
            else:
                feature_map = conv2_activations[samples[sample_idx]][plot_idx]
            axes[plot_idx // 8, plot_idx % 8].axis('off')
            axes[plot_idx // 8, plot_idx % 8].imshow(feature_map, cmap='gray')
            axes[plot_idx // 8, plot_idx % 8].set_title(str(plot_idx), fontsize=16)

        fig_idx = 8+sample_idx*2+layer_idx
        plt.suptitle(f'Feature Map for Conv Layer {layer_idx+1} | Predicted Class: {classes[samples[sample_idx]][0]} | True Class: {classes[samples[sample_idx]][1]}', fontsize=24)
        plt.savefig(f'figures/fig{fig_idx}_c{layer_idx+1}_feature_map.png')
        plt.show()


# %tensorboard --logdir runs/part1
