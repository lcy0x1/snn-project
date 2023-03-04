import snntorch.functional as SF
import torch
from snntorch import spikegen
from snntorch import surrogate
from snntorch import utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from trainers.trainer import Trainer, OptmParams
from mnist_test.net import Net

# Inputs are latency coded spike streams
# Looking at the effect of different parameters (# of hidden nodes, # of time-steps, tolerance, leak vs. no leak, ramp neurons with and without leak)

# dataloader arguments
batch_size_train = 100
batch_size_test = 100
data_path = './Datasets/'
num_classes = 10  # MNIST has 10 output classes
subset = 1  # Reduce training set of 60,000 images to 6000 images

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

print("Training Set Size:", len(mnist_train))
print("Testing Set Size:", len(mnist_test))

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size_train, shuffle=False, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size_test, shuffle=False, drop_last=True)


def spike_transform(t, data):
    return spikegen.latency(data, num_steps=t.num_steps, tau=5, threshold=0.01, clip=True,
                            normalize=True, linear=True)


optm_param = OptmParams(grad=surrogate.fast_sigmoid(), num_steps=10, lr=7e-4, beta_lo=1 - 1e-1, beta_hi=1 - 1e-3)
net = Net(optm_param, beta=1.0, num_inputs=28 * 28, num_hidden=300, num_outputs=10)
# loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)s
loss_fn = SF.mse_temporal_loss(on_target=0, off_target=optm_param.num_steps, tolerance=0)
trainer = Trainer(net, optm_param, loss_fn, spike_transform)
trainer.train(30, train_loader, test_loader).plot()


def conv_seq_to_first_spike_time(tr, spk_rec):
    spk_time = (spk_rec.transpose(0, -1) * (
            torch.arange(0, spk_rec.size(0)).detach().to(tr.device) + 1)).transpose(0, -1)

    """extact first spike time. Will be used to pass into loss function."""
    first_spike_time = torch.zeros_like(spk_time[0])
    for step in range(spk_time.size(0)):
        first_spike_time += (spk_time[step] * ~first_spike_time.bool())  # mask out subsequent spikes

    """override element 0 (no spike) with shadow spike @ final time step, then offset by -1 s.t. first_spike is at t=0."""
    first_spike_time += ~first_spike_time.bool() * (spk_time.size(0))
    first_spike_time -= 1  # fix offset

    return first_spike_time
