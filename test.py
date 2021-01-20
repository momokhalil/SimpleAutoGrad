from autograd.node import Tensor
import autograd as ag
import numpy as np
np.random.seed(50)

y = Tensor(5)

# forward prop and loss
def forward(_y):
    return ag.sin(_y) * ag.cos(_y) + _y


grad = ag.grad(forward(y), [y])
