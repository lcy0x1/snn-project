import snntorch.functional as SF
from snntorch import surrogate

import signal_test.singal_dataset as sn
from signal_test.net import Net
from trainers.trainer import OptmParams, Trainer

if __name__ == "__main__":
    optm_param = OptmParams(grad=surrogate.fast_sigmoid(), num_steps=10, lr=7e-4, beta_lo=1 - 1e-1, beta_hi=1 - 1e-3)
    gen = sn.SignalGenerator(2, 2, 1, sn.gen_or)
    train_loader = sn.SignalDataset(seed=12345, size=100, step=optm_param.num_steps, gen=gen)
    test_loader = sn.SignalDataset(seed=54321, size=30, step=optm_param.num_steps, gen=gen)
    net = Net(optm_param, beta=1.0, num_inputs=2, num_hidden=10, num_outputs=1)
    # loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)s
    # TODO fix loss function
    loss_fn = SF.mse_temporal_loss(on_target=0, off_target=optm_param.num_steps, tolerance=0)
    trainer = Trainer(net, optm_param, loss_fn, lambda a, b: b)
    trainer.train(30, train_loader, test_loader).plot()
