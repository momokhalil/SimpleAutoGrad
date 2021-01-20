from autograd.node import Tensor
import matplotlib.pyplot as plt
import autograd as ag
import numpy as np
np.random.seed(50)

num = 200
y = Tensor(np.linspace(-5, 5, num).reshape(num, 1))

# forward prop and loss
def f(_y):
    return ag.tanh(_y)


fig = plt.figure(figsize=(10, 7))
plt.plot(y.val, f(y).val)

grad = ag.grad(f(y), [y])[0]
plt.plot(y.val, grad.val)

grad = ag.grad(grad, [y])[0]
plt.plot(y.val, grad.val)

grad = ag.grad(grad, [y])[0]
plt.plot(y.val, grad.val)

grad = ag.grad(grad, [y])[0]
plt.plot(y.val, grad.val)

plt.show()
plt.close()
