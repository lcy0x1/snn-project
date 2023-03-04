import datetime as dt

import matplotlib.pyplot as plt
import snntorch.functional as SF
import torch
from snntorch import surrogate
from tqdm import trange


class OptmParams:

    def __init__(self, grad=surrogate.fast_sigmoid(), num_steps=30, lr=7e-4, beta_lo=0.9, beta_hi=0.999):
        self.grad = grad
        self.num_steps = num_steps
        self.lr = lr
        self.betas = (beta_lo, beta_hi)

    def get_optm(self, net):
        return torch.optim.Adam(net.parameters(), lr=self.lr, betas=self.betas)


class TrainResult:

    def __init__(self, train_acc, test_acc):
        self.train_acc = train_acc
        self.test_acc = test_acc

    def plot(self):
        plt.plot(self.train_acc, label='Training Set', marker='o')
        plt.plot(self.test_acc, label='Test Set', marker='o')
        plt.xlabel('Epoch #')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


class TrainProgress:

    def __init__(self):
        self.name = ""
        self.index = 0
        self.max = 0
        self.loss = 0
        self.train_acc = 0
        self.test_acc = 0
        self.epochs = None

    def display(self):
        self.epochs.set_description(f"{self.name}{self.index + 1}/{self.max} | loss: {self.loss:.3e} | "
                                    f"Training Set Accuracy: {self.train_acc*100:.2f}% | "
                                    f"Testing Set Accuracy: {self.test_acc*100:.2f}%")

    def set_sub_process(self, name, size):
        self.name = name
        self.max = size


class Trainer:

    def __init__(self, net, optm: OptmParams, loss, transformer):
        self.dtype = torch.float
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Trainer Constructed. Device: ", self.device)
        self.optm = optm
        self.net = net.to(self.device)
        self.optimizer = self.optm.get_optm(self.net)
        self.loss_fn = loss
        self.num_steps = optm.num_steps
        self.transformer = transformer
        self.progress = TrainProgress()

    def test_accuracy(self, data_loader):
        with torch.no_grad():
            total = 0
            acc = 0
            self.net.eval()

            for i, (data, targets) in enumerate(iter(data_loader)):
                spike_data = self.transformer(self, data)
                data = spike_data.to(self.device)
                targets = targets.to(self.device)
                spk_rec, _ = self.net(data)
                acc += SF.accuracy_temporal(spk_rec, targets) * spk_rec.size(1)
                total += spk_rec.size(1)
                self.progress.index = i
                self.progress.display()
        return acc / total

    def train(self, num_epochs, train_loader, test_loader):
        # training loop
        train_acc = []
        test_acc = []

        epochs = trange(num_epochs)
        self.progress.epochs = epochs

        for epoch in epochs:
            self.progress.set_sub_process(f"Step: {epoch} | Image: ", len(train_loader))
            for i, (data, targets) in enumerate(iter(train_loader)):
                spike_data = self.transformer(self, data)
                data = spike_data.to(self.device)
                targets = targets.to(self.device)
                self.net.train()
                spk_rec, _ = self.net(data)
                loss_val = self.loss_fn(spk_rec, targets)
                self.optimizer.zero_grad()
                loss_val.backward()
                self.optimizer.step()
                self.progress.index = i
                self.progress.loss = loss_val
                self.progress.display()

            self.progress.set_sub_process(f"Step: {epoch} | Evaluate Training Set: ", len(train_loader))
            a1 = self.test_accuracy(train_loader)
            self.progress.set_sub_process(f"Step: {epoch} | Evaluate Testing Set: ", len(test_loader))
            a2 = self.test_accuracy(test_loader)
            train_acc.append(a1)
            test_acc.append(a2)
            self.progress.train_acc = a1
            self.progress.test_acc = a2
            self.progress.display()

        print(train_acc)
        print(test_acc)
        return TrainResult(train_acc, test_acc)
