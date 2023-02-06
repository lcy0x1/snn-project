import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import snntorch as snn
import snntorch.functional as SF
import torch
import torch.nn as nn
from snntorch import spikegen
from snntorch import surrogate
from snntorch import utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Inputs are latency coded spike streams
# Looking at the effect of different parameters (# of hidden nodes, # of time-steps, tolerance, leak vs. no leak, ramp neurons with and without leak)

# dataloader arguments
batch_size_train = 100
batch_size_test = 100
data_path = '/home/prashansa/SNNTorch/Datasets/'
num_classes = 10  # MNIST has 10 output classes
subset = 1  # Reduce training set of 60,000 images to 6000 images

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(device)

# Define a transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

mnist_train = utils.data_subset(mnist_train, subset)
mnist_test = utils.data_subset(mnist_test, subset)

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size_train, shuffle=False, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size_test, shuffle=False, drop_last=True)

# Network Architecture
num_inputs = 28 * 28
num_hidden = 300
num_outputs = 10

# Temporal Dynamics
num_steps = 10
beta = 1.0
spike_grad = surrogate.fast_sigmoid()


# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden, bias=False)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, reset_mechanism="zero", learn_beta=True)
        self.fc2 = nn.Linear(num_hidden, num_outputs, bias=False)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, reset_mechanism="zero", learn_beta=True)

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        # print(x.size(),x.flatten(1).size())

        for step in range(num_steps):
            cur1 = self.fc1(x[step].flatten(1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

    def init_L1_weights(self):
        self.fc1.weight.data = torch.normal(mean=torch.zeros(num_hidden, num_inputs),
                                            std=np.sqrt(2 / (num_inputs + num_hidden)))
        return

    def init_L2_weights(self):
        self.fc2.weight.data = torch.normal(mean=torch.zeros(num_outputs, num_hidden),
                                            std=np.sqrt(2 / (num_hidden + num_outputs)))
        return


def conv_seq_to_first_spike_time(spk_rec):
    spk_time = (spk_rec.transpose(0, -1) * (torch.arange(0, spk_rec.size(0)).detach().to(device) + 1)).transpose(0, -1)

    """extact first spike time. Will be used to pass into loss function."""
    first_spike_time = torch.zeros_like(spk_time[0])
    for step in range(spk_time.size(0)):
        first_spike_time += (spk_time[step] * ~first_spike_time.bool())  # mask out subsequent spikes

    """override element 0 (no spike) with shadow spike @ final time step, then offset by -1 s.t. first_spike is at t=0."""
    first_spike_time += ~first_spike_time.bool() * (spk_time.size(0))
    first_spike_time -= 1  # fix offset

    return first_spike_time


def test_accuracy(data_loader, net, num_steps):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        data_loader = iter(data_loader)
        for data, targets in data_loader:
            spike_data = spikegen.latency(data, num_steps=num_steps, tau=5, threshold=0.01, clip=True, normalize=True,
                                          linear=True)
            data = spike_data.to(device)
            targets = targets.to(device)
            spk_rec, _ = net(data)

            acc += SF.accuracy_temporal(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc / total


# Load the network onto CUDA if available
net = Net().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=7e-4, betas=(0.9, 0.999))
# loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
loss_fn = SF.mse_temporal_loss(on_target=0, off_target=num_steps, tolerance=0)

num_epochs = 30
loss_hist = []
acc_hist = []

# net.init_L1_weights()
# net.init_L2_weights()

# training loop

train_acc = []
test_acc = []

for epoch in range(num_epochs):
    t1 = dt.datetime.now()
    for i, (data, targets) in enumerate(iter(train_loader)):
        # print(epoch,i)
        spike_data = spikegen.latency(data, num_steps=num_steps, tau=5, threshold=0.01, clip=True, normalize=True,
                                      linear=True)
        data = spike_data.to(device)
        targets = targets.to(device)

        net.train()
        spk_rec, _ = net(data)

        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    t2 = dt.datetime.now()
    a1 = test_accuracy(train_loader, net, num_steps)
    a2 = test_accuracy(test_loader, net, num_steps)
    t3 = dt.datetime.now()
    train_acc.append(a1)
    test_acc.append(a2)
    print(epoch)
    print(f"Training set accuracy: {a1 * 100:.3f}%")
    print(f"Test set accuracy: {a2 * 100:.3f}%")
    print("Time to run one epoch: " + str(t2 - t1))
    print("Time to compute accuracies: " + str(t3 - t2))

print(net.lif1.beta)
print(net.lif2.beta)

print(train_acc)
print(test_acc)

plt.plot(train_acc, label='Training Set', marker='o')
plt.plot(test_acc, label='Test Set', marker='o')
plt.xlabel('Epoch #')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
