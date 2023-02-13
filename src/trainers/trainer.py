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


class Trainer:

    def __init__(self, net, optm: OptmParams, loss, transformer):
        self.dtype = torch.float
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Trainer Constructed. Device: ", self.device)
        self.optm = optm
        self.net = net.to(self.device)
        self.optimizer = self.optm.get_optm(self.net)
        # loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
        self.loss_fn = loss
        self.num_steps = optm.num_steps
        self.transformer = transformer

    def test_accuracy(self, data_loader):
        with torch.no_grad():
            total = 0
            acc = 0
            self.net.eval()

            data_loader = iter(data_loader)
            for data, targets in data_loader:
                spike_data = self.transformer(self, data)
                data = spike_data.to(self.device)
                targets = targets.to(self.device)
                spk_rec, _ = self.net(data)
                acc += SF.accuracy_temporal(spk_rec, targets) * spk_rec.size(1)
                total += spk_rec.size(1)
        return acc / total

    def train(self, num_epochs, train_loader, test_loader):
        loss_hist = []
        acc_hist = []
        # self.net.init_L1_weights()
        # self.net.init_L2_weights()

        # training loop
        train_acc = []
        test_acc = []

        epochs = trange(num_epochs)

        for epoch in epochs:
            t1 = dt.datetime.now()
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
                epochs.set_description(f"Step: {epoch} | Image: {i + 1}/{len(train_loader)} | loss: {loss_val:.3e}")

            t2 = dt.datetime.now()
            a1 = self.test_accuracy(train_loader)
            a2 = self.test_accuracy(test_loader)
            t3 = dt.datetime.now()
            train_acc.append(a1)
            test_acc.append(a2)
            print(f"Training set accuracy: {a1 * 100:.3f}%")
            print(f"Test set accuracy: {a2 * 100:.3f}%")
            print("Time to run one epoch: " + str(t2 - t1))
            print("Time to compute accuracies: " + str(t3 - t2))

        # print(self.net.lif1.beta)
        # print(self.net.lif2.beta)

        print(train_acc)
        print(test_acc)

        plt.plot(train_acc, label='Training Set', marker='o')
        plt.plot(test_acc, label='Test Set', marker='o')
        plt.xlabel('Epoch #')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
