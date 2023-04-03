import math

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_max, num_cycles, n_warmup_steps, num_training_steps):
        self._optimizer = optimizer
        self.lr_max = lr_max
        self.num_cycles = num_cycles
        self.n_warmup_steps = n_warmup_steps
        self.num_training_steps = num_training_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        if self.n_steps < self.n_warmup_steps:
            if WARMUP_METHOD == 'log':
                return self.lr_max * 0.10 ** (self.n_warmup_steps - self.n_steps)
            else:
                return self.lr_max * 2 ** -(self.n_warmup_steps - self.n_steps)
        else:
            process = float(self.n_steps - self.n_warmup_steps) / float(max(1, self.num_training_steps - self.n_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * process))) * self.lr_max


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr