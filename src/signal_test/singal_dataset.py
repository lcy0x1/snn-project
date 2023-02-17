import random

import torch as th


class SignalGenerator:

    def __init__(self, delay, in_size, out_size, operation):
        self.operation = operation
        self.delay = delay
        self.in_size = in_size
        self.out_size = out_size

    def generate(self, r, step):
        ans = []
        target = [[0] * self.out_size] * self.delay
        for i in range(step - self.delay):
            entry = []
            for j in range(self.in_size):
                a = r.random() > 0.5
                entry.append(1 if a else 0)
            ans.append(entry)
            target.append(self.operation(entry))
        ans.extend([[0] * self.in_size] * self.delay)
        return th.Tensor(ans), th.Tensor(target)


class SignalDataset:

    def __init__(self, seed, size, step, gen: SignalGenerator):
        self.size = size
        self.step = step
        self.data = []
        self.random = random.Random(seed)
        for i in range(size):
            dat, tar = gen.generate(self.random, step)
            self.data.append((dat, tar))

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self.data)


def gen_and(lst):
    a = lst[0]
    b = lst[1]
    return [1 if a > 0 and b > 0 else 0]


def gen_or(lst):
    a = lst[0]
    b = lst[1]
    return [1 if a > 0 or b > 0 else 0]


def gen_xor(lst):
    a = lst[0]
    b = lst[1]
    return [1 if (a > 0) != (b > 0) else 0]
