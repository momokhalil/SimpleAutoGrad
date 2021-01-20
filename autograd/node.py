from typing import Union
import numpy as np

class Node:
    eps = 1e-10

    def __init__(self, val_, var_=None):
        self._var = var_
        self._val = val_

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val_):
        self._val = val_

    @property
    def var(self):
        return self._var

    @var.setter
    def var(self, var_):
        self._var = var_

    @property
    def shape(self):
        return self._val.shape

    @staticmethod
    def check_type(other):
        if not isinstance(other, (Tensor, Node)):
            if isinstance(other, int) or isinstance(other, float):
                other = np.double(other)
            other = Tensor(other)
        return other

    def grad(self, adjonit):
        raise NotImplementedError()

    def __add__(self, other: Union['Tensor', 'Node', int, float, np.ndarray]):
        return _add(self, self.check_type(other))

    def plus(self, other: Union['Tensor', 'Node', int, float, np.ndarray]):
        return self.__add__(other)

    def __mul__(self, other: Union['Tensor', 'Node', int, float, np.ndarray]):
        return _mul(self, self.check_type(other))

    def times(self, other: Union['Tensor', 'Node', int, float, np.ndarray]):
        return self.__mul__(other)

    def __pow__(self, other: Union['Tensor', 'Node', int, float, np.ndarray]):
        return _pow(self, self.check_type(other))

    def pow(self, other: Union['Tensor', 'Node', int, float, np.ndarray]):
        return self.__pow__(other)

    def __sub__(self, other: Union['Tensor', 'Node', int, float, np.ndarray]):
        return _add(self, _neg(self.check_type(other)))

    def minus(self, other: Union['Tensor', 'Node', int, float, np.ndarray]):
        return self.__sub__(other)

    def __truediv__(self, other: Union['Tensor', 'Node', int, float, np.ndarray]):
        return _mul(self, _inv(self.check_type(other)))

    def div(self, other: Union['Tensor', 'Node', int, float, np.ndarray]):
        return self.__truediv__(other)

    def __neg__(self):
        return _neg(self)

# tensor class
class Tensor(Node):
    def __init__(self, val):
        super().__init__(val_=val)

    def grad(self, adjoint):
        raise NotImplementedError()

# Private classes ------------------------------------------------------------------------------------------------------
# add node
class _add(Node):
    def __init__(self,
                 a: Union[Tensor, Node],
                 b: Union[Tensor, Node]) -> None:
        super().__init__(val_=a.val + b.val, var_=[a, b])

    def grad(self, adjoint):
        return [(self.var[0], adjoint),
                (self.var[1], adjoint)]

# power node
class _pow(Node):
    def __init__(self,
                 a: Union[Tensor, Node],
                 b: Union[Tensor, Node]) -> None:
        super().__init__(val_=a.val ** b.val, var_=[a, b])

    def grad(self, adjoint):
        return [(self.var[0], self.var[1] * self.var[0] ** (self.var[1] - 1)),
                (self.var[1], (self.var[0] ** self.var[1]) * log(self.var[0]))]

# multiplication node
class _mul(Node):
    def __init__(self,
                 a: Union[Tensor, Node],
                 b: Union[Tensor, Node]) -> None:
        super().__init__(val_=a.val * b.val, var_=[a, b])

    def grad(self, adjoint):
        return [(self.var[0], adjoint * self.var[1]), (self.var[1], adjoint * self.var[0])]

# reciprocal node
class _inv(Node):
    def __init__(self, var: Union[Tensor, Node]) -> None:
        super().__init__(val_=1 / (var.val + Node.eps), var_=var)

    def grad(self, adjoint):
        return [(self.var, -adjoint / (self.var * self.var))]

# negation node
class _neg(Node):
    def __init__(self, var: Union[Tensor, Node]) -> None:
        super().__init__(val_=-var.val, var_=var)

    def grad(self, adjoint):
        return [(self.var, -adjoint)]

# sine node
class sin(Node):
    def __init__(self, var: Union[Tensor, Node]) -> None:
        super().__init__(val_=np.sin(var.val), var_=var)

    def grad(self, adjoint):
        return [(self.var, adjoint * cos(self.var))]

# cosine node
class cos(Node):
    def __init__(self, var: Union[Tensor, Node]) -> None:
        super().__init__(val_=np.cos(var.val),
                         var_=var)

    def grad(self, adjoint):
        return [(self.var, -adjoint * sin(self.var))]

# tangent node
class tan(Node):
    def __init__(self, var: Union[Tensor, Node]) -> None:
        super().__init__(val_=np.tan(var.val), var_=var)

    def grad(self, adjoint):
        return [(self.var, adjoint * sec(self.var) * sec(self.var))]

# secant node
class sec(Node):
    def __init__(self, var: Union[Tensor, Node]) -> None:
        super().__init__(val_=1/(np.cos(var.val) + Node.eps), var_=var)

    def grad(self, adjoint):
        return [(self.var, adjoint * sec(self.var) * tan(self.var))]

# secant node
class csc(Node):
    def __init__(self, var: Union[Tensor, Node]) -> None:
        super().__init__(val_=1/(np.sin(var.val) + Node.eps), var_=var)

    def grad(self, adjoint):
        return [(self.var, -adjoint * csc(self.var) * cot(self.var))]

# secant node
class cot(Node):
    def __init__(self, var: Union[Tensor, Node]) -> None:
        super().__init__(val_=1/(np.tan(var.val) + Node.eps), var_=var)

    def grad(self, adjoint):
        return [(self.var, -adjoint * csc(self.var) * csc(self.var))]

# hyperbolic sine node
class sinh(Node):
    def __init__(self, var: Union[Tensor, Node]) -> None:
        super().__init__(val_=np.sinh(var.val), var_=var)

    def grad(self, adjoint):
        return [(self.var, adjoint * cosh(self.var))]

# hyperbolic cosine node
class cosh(Node):
    def __init__(self, var: Union[Tensor, Node]) -> None:
        super().__init__(val_=np.cosh(var.val), var_=var)

    def grad(self, adjoint):
        return [(self.var, adjoint * sinh(self.var))]

# hyperbolic tan node
class tanh(Node):
    def __init__(self, var: Union[Tensor, Node]) -> None:
        super().__init__(val_=np.tanh(var.val), var_=var)

    def grad(self, adjoint):
        return [(self.var, -adjoint * (tanh(self.var) * tanh(self.var) - 1))]

# exponentiation node
class exp(Node):
    def __init__(self, var: Union[Tensor, Node]) -> None:
        super().__init__(val_=np.exp(var.val), var_=var)

    def grad(self, adjoint):
        return [(self.var, adjoint * exp(self.var))]

# logarithm node
class log(Node):
    def __init__(self, var: Union[Tensor, Node]):
        super().__init__(val_=np.log(var.val + Node.eps), var_=var)

    def grad(self, adjoint):
        return [(self.var, adjoint * recip(self.var))]

# logarithm node
class recip(Node):
    def __init__(self, var: Union[Tensor, Node]):
        super().__init__(val_=1 / (var.val + Node.eps), var_=var)

    def grad(self, adjoint):
        return [(self.var, -adjoint / (self.var * self.var))]

# square root node
class sqrt(Node):
    def __init__(self, var: Union[Tensor, Node]) -> None:
        super().__init__(val_=np.sqrt(var.val), var_=var)

    def grad(self, adjoint):
        return [(self.var, adjoint / sqrt(self.var) / 0.5)]

class matmul(Node):
    def __init__(self,
                 a: Union[Tensor, Node],
                 b: Union[Tensor, Node],) -> None:
        super().__init__(val_=np.matmul(a.val, b.val), var_=[a, b])

    def grad(self, adjoint):
        return [(self.var[0], matmul(adjoint, transpose(self.var[1]))),
                (self.var[1], matmul(transpose(self.var[0]), adjoint))]

class transpose(Node):
    def __init__(self,
                 var: Union[Tensor, Node]) -> None:
        super().__init__(val_=np.transpose(var.val), var_=var)

    def grad(self, adjoint):
        return [(self.var, adjoint)]

class Sum(Node):
    def __init__(self,
                 var: Union[Tensor, Node]) -> None:
        super().__init__(val_=np.sum(var.val), var_=var)

    def grad(self, adjoint):
        return [(self.var, adjoint)]

class ReLU(Node):
    def __init__(self,
                 var: Union[Tensor, Node]) -> None:
        super().__init__(val_=var.val * (var.val > 0), var_=var)

    def grad(self, adjoint):
        return [(self.var, adjoint * (self.val > 0))]
