import matplotlib.pyplot as plt
from autograd.node import Tensor
import autograd as ag
import numpy as np
np.random.seed(550)

num = 50
iters = 10000
alpha = 0.0001
losses = np.empty(shape=(iters, ))

# Toy dataset
data = np.random.randn(num, 4)
x = Tensor(data[:, :-1])
y = Tensor(np.reshape(data[:, -1], (num, 1)))

# Initialize parameters
units = 20
W1 = Tensor(np.random.randn(3, units) * np.sqrt(2 / 3))
W2 = Tensor(np.random.randn(units, 1) * np.sqrt(2 / units))
b1 = Tensor(np.zeros((1, units)))
b2 = Tensor(np.zeros((1, 1)))


# forward prop and loss
@ag.makegrad(wrtvars=[2, 3, 4, 5])
def forward(_x, _y, _w1, _w2, _b1, _b2):
    z1 = ag.ReLU(ag.matmul(_x, _w1) + _b1)
    z2 = ag.matmul(z1, _w2) + _b2
    return ag.Sum((z2 - _y) * (z2 - _y))


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











#plt.plot(a.val, graph.val)


#plt.legend(("f", "f '", "f ''", "f '''", "f ''''"), frameon=False)
#plt.title('f = tanh(x) sin(x) (1 - exp(sin(x)))')
#plt.show()
#plt.close()

# Finite difference
#dyda = (f(a + eps, b) - f(a - eps, b)) / eps / 2
#dyda = (f(a + eps, b) - f(a, b)*2 + f(a - eps, b)) / (eps ** 2)

#print('autograd =', graph.val)
#print('finite_d =', dyda.val)
#print('error =', np.absolute(graph.val - dyda.val))
