import matplotlib.pyplot as plt
from autograd.node import Tensor
import autograd as ag
import numpy as np
np.random.seed(550)

num = 50
iters = 10000
alpha = 0.0004
losses = np.empty(shape=(iters, ))

# Toy dataset
dim = 4
data = np.random.randn(num, dim+1)
x = Tensor(data[:, :-1])
y = Tensor(np.reshape(data[:, -1], (num, 1)))

# Initialize parameters
units = 24
W1 = Tensor(np.random.randn(dim, units) * np.sqrt(2 / dim))
W2 = Tensor(np.random.randn(units, 1) * np.sqrt(2 / units))
b1 = Tensor(np.zeros((1, units)))
b2 = Tensor(np.zeros((1, 1)))

def dense_relu(_x, _w, _b):
    return ag.ReLU(ag.matmul(_x, _w) + _b)

def dense(_x, _w, _b):
    return ag.matmul(_x, _w) + _b

def mse(_y, label):
    return ag.Sum((_y - label) * (_y - label))

# forward prop and loss
@ag.makegrad(wrtvars=[2, 3, 4, 5])
def forward(_x, _y, _w1, _w2, _b1, _b2):
    z1 = dense_relu(_x, _w1, _b1)
    z2 = dense(z1, _w2, _b2)
    return mse(z2, _y)


# Gradient descent
for i in range(iters):
    dW1, dW2, db1, db2, loss = forward(x, y, W1, W2, b1, b2)

    W1.val -= dW1.val * alpha
    W2.val -= dW2.val * alpha
    b1.val -= db1.val * alpha
    b2.val -= db2.val * alpha

    losses[i] = loss

# Plotting
plt.plot(losses)
plt.show()
plt.close()
