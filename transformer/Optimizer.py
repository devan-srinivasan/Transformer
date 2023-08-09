"""
Wrapper to adjust learning rate and implement ADAM optimizer
"""

class OptimizerWrapper:
    def __init__(self, model_size, factor, warmup, optimizer) -> None:
        self.optimizer = optimizer
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self.lr = 0
        self.step = 0
    
    def step(self):
        self.step += 1
        self.lr = self.lr()
        for param in self.optimizer.param_groups:
            param['lr'] = self.lr
        self.optimizer.step()

    def lr(self):
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(self.step ** (-0.5), self.step * self.warmup ** (-1.5)))