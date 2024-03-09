from ..grad.val.engine import Value
from typing import List

class optimizer:
    def __init__(self) -> None:
        raise NotImplementedError

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0.0
    
    def step(self):
        raise NotImplementedError


class GD(optimizer):
    """
    Stochastic Gradient Descent
    """
    def __init__(self, parameters: List['Value'], lr: float):
        self.lr = lr
        self.parameters = parameters
        
    def step(self):
        for param in self.parameters:
            param.value -= (param.grad)*(self.lr)


class MBGD(optimizer):
    """
    Momentum Based Gradient Descent
    """
    def __init__(self, parameters: List['Value'], lr: float, gamma: float = 0.9):
        self.lr = lr
        self.parameters = parameters
        self.gamma = gamma

        self.prev_updates = [0.0 for _ in self.parameters]
    
    def step(self):
        for i in range(len(self.parameters)):
            update = (self.parameters[i].grad)*(self.lr) + (self.gamma)*(self.prev_updates[i])
            self.prev_updates[i] = update
            self.parameters[i].value -= update


class SGD(optimizer):
    """
    SGD optimizer with effect of Momentum, dampening, weight decay and Nesterov Accelerated gradient.
    PS: The implementation of Momentum based and Nesterov Acceleration is as done in PyTorch; which themselves are an approximation of the original algorithm.
    """
    def __init__(self,
                 parameters,
                 lr,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nestrov=False
                 ):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nestrov = nestrov
        self.t=0

    def step(self):
        for param in self.parameters:
            update = param.grad
            vel = update
            if(self.weight_decay!=0):
                update = update + (self.weight_decay)*param.value
            if(self.momentum!=0):
                if(self.t>0):
                    vel = (self.momentum)*vel + (1-self.dampening)*update
                else:
                    vel = update
                if(self.nestrov):
                    update = update + self.momentum*vel
                else:
                    update = vel
            if(self.maximize):
                param.value += self.lr*update
            else:
                param.value -= self.lr*update
        self.t += 1
