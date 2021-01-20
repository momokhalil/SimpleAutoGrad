from autograd.node import Node, Tensor
from collections import defaultdict
from typing import Union, Callable
import numpy as np
import functools


# Reverse broadcasting by averaging over dimensions
def unbroadcast(var_, grad_):
    for axis, (grad_dim, val_dim) in enumerate(zip(grad_.shape, var_.shape)):
        if grad_dim > val_dim:
            grad_.val = np.sum(grad_.val, axis=axis, keepdims=True) / grad_.shape[axis]
    return grad_


# gradient transformation - function
def grad(parent, wrtvars: Union[list, Tensor, None] = None) -> list:
    if wrtvars is None:
        raise ValueError('wrt array must be specified')
    grads = defaultdict(lambda: Tensor(0))
    graph = parent.grad(Tensor(np.ones(shape=parent.val.shape)))
    for node, adjoint in graph:
        grads[node] += unbroadcast(node, adjoint)
        if not isinstance(node, Tensor):
            for child_node, child_grad in node.grad(adjoint):
                graph.append((child_node, child_grad))
    return [grads[wrt] for wrt in wrtvars]


# gradient transformation - decorator
def makegrad(wrtvars: list = None) -> Callable:
    def wrapper(func: Callable[[int, float, np.ndarray, Tensor], Union[Tensor, Node]]) -> Callable:
        @functools.wraps(func)
        def get_grad(*args) -> tuple[list[Tensor], float]:
            if wrtvars is None:
                raise ValueError('wrt array must be specified')
            grads = defaultdict(lambda: Tensor(0))
            parent = func(*args)
            graph = parent.grad(Tensor(np.ones(shape=parent.val.shape)))
            for node, adjoint in graph:
                grads[node] += unbroadcast(node, adjoint)
                if not isinstance(node, Tensor):
                    for child_node, child_grad in node.grad(adjoint):
                        graph.append((child_node, child_grad))
            return *[grads[args[wrt]] for wrt in wrtvars], parent.val
        return get_grad
    return wrapper
