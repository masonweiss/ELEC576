# Mason Weiss
# ELEC 576 - Introduction to Deep Learning
# Prof. Ankit Patel, Rice University
# Due: October 10, 2024
# adapted from code snippet assignment_1_pytorch_mnist_skeleton.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter

batch_size = 50
test_batch_size = 100
num_epochs = 10
try_cuda = True
seed = 1000
logging_interval_1 = 10   # how many batches to wait before logging loss
logging_interval_2 = 100  # how many batches to wait before logging stats on model components
logging_dir = './log'

#deciding whether to send to the cpu or not if available
if torch.cuda.is_available() and try_cuda:
    cuda = True
    torch.cuda.manual_seed(seed)
else:
    cuda = False
    torch.manual_seed(seed)

# 2) Import data
transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.01307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# 3) Define Model architecture, loss and optimizer
class Net(nn.Module):
    def __init__(self, act_name, initializer):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, out_features=10)  # add fc layer before output
        self.actFun = act_name  # store name of activation function
        self.initializer = initializer
        self.initialize()

    def forward(self, x):
        x = self.conv1(x)
        x = activation(x, self.actFun)
        x = F.max_pool2d(input=x, kernel_size=2)
        x = self.conv2(x)
        x = activation(x, self.actFun)
        x = F.max_pool2d(input=x, kernel_size=2)
        x = x.view(-1, 1024)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x

    def initialize(self):
        if self.initializer == 'xavier_uniform':  # initialize with glorot
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
        elif self.initializer == 'kaiming_uniform':  # initialize with He
            nn.init.kaiming_uniform_(self.conv1.weight)
            nn.init.kaiming_uniform_(self.conv2.weight)
            nn.init.kaiming_uniform_(self.fc1.weight)
            nn.init.kaiming_uniform_(self.fc2.weight)
        else:
            assert False, 'Provide a valid initializer'


def train(epoch):
    # 4(i) Define Training loop
    model.train()
    criterion = nn.CrossEntropyLoss()  # cross entropy loss accounts for softmax and NLL loss

    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)  # remove adjustments for computing log to not compute twice
        loss.backward()
        optimizer.step()

        with torch.no_grad():  # stop computing gradient when propagating through network in log_info
            if batch_idx % logging_interval_1 == 0:
                print(f'Epoch: {epoch+1} of {num_epochs} | Batch {batch_idx} of {len(train_loader)}| Loss: {loss.item()}')
                n_iter = batch_idx + epoch * len(train_loader)
                writer.add_scalar(f'Train/Loss', loss.item(), n_iter)

            if batch_idx % logging_interval_2 == 0:
                n_iter = batch_idx + epoch * len(train_loader)
                log_info(data, n_iter)


def test(epoch):
    # 4(ii) Define testing loop
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()  # cross entropy loss accounts for softmax and NLL loss

    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)

        test_loss += criterion(output, target).item()  # sum up batch loss (later, averaged over all test samples)
        pred = output.argmax(dim=1)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()  # scale target and pred to same dimension and compare

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print(f'TEST - Epoch: {epoch+1} of {num_epochs} | Accuracy: {test_accuracy}| Loss: {test_loss}')

    # Log test/loss and test/accuracy to TensorBoard at every epoch
    n_iter = epoch * len(test_loader)
    writer.add_scalar(f'Test/Loss', test_loss, n_iter)
    writer.add_scalar(f'Test/Accuracy', test_accuracy, n_iter)

def activation(x, act_name):
    if act_name == 'relu':
        return F.relu(x)
    elif act_name == 'tanh':
        return F.tanh(x)
    elif act_name == 'sigmoid':
        return F.sigmoid(x)
    elif act_name == 'leaky-relu':
        return F.leaky_relu(x, 0.1)  # apply leaky relu with negative slope of 0.1
    else:
        assert False, 'Provide a valid activation'


def log_info(data, n_iter):
    layers = [model.conv1, model.conv2, model.fc1, model.fc2]
    layer_names = ['conv1', 'conv2', 'fc1', 'fc2']

    # weight and bias for each tunable layer
    for i in range(len(layers)):
        writer.add_histogram(f'Train/{layer_names[i]}/weight', layers[i].weight, n_iter)
        writer.add_histogram(f'Train/{layer_names[i]}/bias', layers[i].bias, n_iter)
        writer.add_scalar(f'Train/{layer_names[i]}/weight_min', torch.min(layers[i].weight), n_iter)
        writer.add_scalar(f'Train/{layer_names[i]}/bias_min', torch.min(layers[i].bias), n_iter)
        writer.add_scalar(f'Train/{layer_names[i]}/weight_max', torch.max(layers[i].weight), n_iter)
        writer.add_scalar(f'Train/{layer_names[i]}/bias_max', torch.max(layers[i].bias), n_iter)
        writer.add_scalar(f'Train/{layer_names[i]}/weight_mean', torch.mean(layers[i].weight), n_iter)
        writer.add_scalar(f'Train/{layer_names[i]}/bias_mean', torch.mean(layers[i].bias), n_iter)
        writer.add_scalar(f'Train/{layer_names[i]}/weight_std', torch.std(layers[i].weight), n_iter)
        writer.add_scalar(f'Train/{layer_names[i]}/bias_std', torch.std(layers[i].bias), n_iter)

    # net input for each layer
    conv1_input = data
    activation1_output = activation(model.conv1(conv1_input), model.actFun)
    maxpool1_output = F.max_pool2d(input=activation1_output, kernel_size=2)
    conv2_input = maxpool1_output
    activation2_output = activation(model.conv2(conv2_input), model.actFun)
    maxpool2_output = F.max_pool2d(input=activation2_output, kernel_size=2)
    fc1_input = maxpool2_output.view(-1, 1024)
    dropout_input = model.fc1(fc1_input)
    fc2_input = model.drop(dropout_input)
    model_output = model.fc2(fc2_input)

    params = [conv1_input, activation1_output, maxpool1_output, conv2_input, activation2_output,
              maxpool2_output, fc1_input, dropout_input, fc2_input, model_output]
    p_names = ['conv1_input', 'activation1_output', 'maxpool1_output', 'conv2_input', 'activation2_output',
               'maxpool2_output', 'fc1_input', 'dropout_input', 'fc2_input', 'model_output']

    for i in range(len(params)):
        writer.add_histogram(f'Train/{p_names[i]}/values', params[i], n_iter)
        writer.add_scalar(f'Train/{p_names[i]}/min', torch.min(params[i]), n_iter)
        writer.add_scalar(f'Train/{p_names[i]}/max', torch.max(params[i]), n_iter)
        writer.add_scalar(f'Train/{p_names[i]}/mean', torch.mean(params[i]), n_iter)
        writer.add_scalar(f'Train/{p_names[i]}/std', torch.std(params[i]), n_iter)


# iterate through model constructions
for act in ('relu', 'tanh', 'sigmoid', 'leaky-relu'):
    for lr in (1e-2, 1e-3):
        for init in ('xavier_uniform', 'kaiming_uniform'):
            for opti in ('adam', 'sgd', 'sgd_0.5'):
                # 1) setting up the logging
                writer = SummaryWriter(log_dir=logging_dir+f'/final_submission/{act}/{lr}/{init}/{opti}')
                print(f"\nMODEL {act}, {lr}, {init}, {opti}")

                model = Net(act_name=act, initializer=init)
                if opti == 'adam':
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                if opti == 'sgd':
                    optimizer = optim.SGD(model.parameters(), momentum=0)
                else:
                    optimizer = optim.SGD(model.parameters(), momentum=0.5)

                # 5) Perform Training over multiple epochs:
                for epoch_idx in range(num_epochs):
                    train(epoch_idx)
                    test(epoch_idx)

                writer.close()

# To View in Tensorboard:
# %tensorboard --logdir ./runs/final_submission
