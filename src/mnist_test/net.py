import numpy as np
import snntorch as snn
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, optm_param, beta, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.optm_param = optm_param

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden, bias=False)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=optm_param.grad, reset_mechanism="zero", learn_beta=True)
        self.fc2 = nn.Linear(num_hidden, num_outputs, bias=False)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=optm_param.grad, reset_mechanism="zero", learn_beta=True)
        self.init_weights()

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        # print(x.size(),x.flatten(1).size())

        for step in range(self.optm_param.num_steps):
            cur1 = self.fc1(x[step].flatten(1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

    def init_weights(self):
        self.fc1.weight.data = torch.normal(mean=torch.zeros(self.num_hidden, self.num_inputs),
                                            std=np.sqrt(2 / (self.num_inputs + self.num_hidden)))
        self.fc2.weight.data = torch.normal(mean=torch.zeros(self.num_outputs, self.num_hidden),
                                            std=np.sqrt(2 / (self.num_hidden + self.num_outputs)))
