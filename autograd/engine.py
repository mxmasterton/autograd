import math

class Value:
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0
        self._backprop = lambda: None
        self._prev = set(_children)

    def __repr__(self):
        return "Value(data={})".format(self.data)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backprop():
            self.grad += out.grad
            other.grad += out.grad
        out._backprop = _backprop

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backprop():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backprop = _backprop

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,))

        def _backprop():
            self.grad += (other * self.data**(other - 1)) * out.grad
        out._backprop = _backprop

        return out

    def backprop(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backprop()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other