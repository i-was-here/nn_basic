import numpy as np

class Vector():
    def __init__(self, value, operator='', children=(), label=''):
        assert isinstance(value, np.ndarray), "The value of vector can only be of type numpy.ndarray !"
        assert len(value.shape)==1, "Only 1-D row vectors allowed !"

        if(value.dtype!=np.float32): value = value.astype(np.float32)
        self.value = value
        self.operator = operator
        self.children = children
        self.label = label
        self._backward = lambda: None
        self.grad = np.zeros_like(value, dtype=np.float32)
    
    def __str__(self) -> str:
        return "Value: "+np.array2string(self.value, precision=4, separator=', ')
    
    def __repr__(self) -> str:
        return self.__str__()
    
    @property
    def shape(self):
        return self.value.shape
    
    def __mul__(self, another: 'Vector') -> 'Vector':
        assert isinstance(another, Vector), "Element-wise vector multiplication allowed only b/w 2 vectors ! But got a different types !"
        assert self.shape==another.shape, "Shape mismatch between the input vectors for element-wise multiplication !"

        out = Vector(self.value*another.value, "*", (self, another))

        def _backward():
            self.grad += (out.grad)*(another.value)
            another.grad += (out.grad)*(self.value)
        out._backward = _backward

        return out
    
    def __rmul__(self, another: 'Vector') -> 'Vector':
        return another*self

    def __add__(self, another: 'Vector') -> 'Vector':
        assert isinstance(another, Vector), "Element-wise vector addition allowed only b/w 2 vectors ! But got a different types !"
        assert self.shape==another.shape, "Shape mismatch between the input vectors for element-wise addition !"

        out = Vector(self.value+another.value, "+", (self, another))

        def _backward():
            self.grad += out.grad
            another.grad += out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, another: 'Vector') -> 'Vector':
        return another+self
    
    def __sub__(self, another: 'Vector') -> 'Vector':
        assert isinstance(another, Vector), "Element-wise vector subtraction allowed only b/w 2 vectors ! But got a different types !"
        assert self.shape==another.shape, "Shape mismatch between the input vectors for element-wise subtraction !"

        out = Vector(self.value-another.value, "-", (self, another))

        def _backward():
            self.grad += out.grad
            another.grad += -1*(out.grad)
        out._backward = _backward

        return out
    
    def __rsub__(self, another: 'Vector') -> 'Vector':
        assert isinstance(another, Vector), "Element-wise vector subtraction allowed only b/w 2 vectors ! But got a different types !"
        assert self.shape==another.shape, "Shape mismatch between the input vectors for element-wise subtraction !"

        out = Vector(another.value-self.value, "-", (another, self))

        def _backward():
            self.grad += -1*out.grad
            another.grad += out.grad
        out._backward = _backward

        return out
    
    def __truediv__(self, another: 'Vector') -> 'Vector':
        assert isinstance(another, Vector), "Element-wise vector division allowed only b/w 2 vectors ! But got a different types !"
        assert self.shape==another.shape, "Shape mismatch between the input vectors for element-wise division !"

        out = Vector(self.value/another.value, "/", (self, another))

        def _backward():
            self.grad += (out.grad)/(another.value)
            another.grad += -1*(out.grad*out.value)/another.value
        out._backward = _backward

        return out
    
    def __rtruediv__(self, another: 'Vector') -> 'Vector':
        assert isinstance(another, Vector), "Element-wise vector division allowed only b/w 2 vectors ! But got a different types !"
        assert self.shape==another.shape, "Shape mismatch between the input vectors for element-wise division !"

        out = Vector(another.value/self.value, "/", (another, self))

        def _backward():
            self.grad += -1*(out.grad*out.value)/(self.value)
            another.grad += (out.grad)/(self.value)
        out._backward = _backward

        return out
    
    def sum(self, dim=None):
        # TODO: consider the dim over which sum needs to be done
        
        out = Vector(np.array([float(self.value.sum())]), 'sum', (self,))

        def _backward():
            self.grad += (out.grad*np.ones_like(self.value))
        out._backward = _backward

        return out
    
    def dot_product(self, another: 'Vector') -> 'Vector':
        assert isinstance(another, Vector), "Dot Product allowed only b/w 2 vectors ! But got a different types !"
        assert self.shape==another.shape, "Shape mismatch between the input vectors for Dot Product !"

        out = Vector(np.array([np.matmul(self.value, another.value.T)]), 'dot_prod', (self, another))
        def _backward():
            self.grad += out.grad*(another.value)
            another.grad += out.grad*(self.value)
        out._backward = _backward

        return out
    
    def tanh(self):
        out = Vector(np.tanh(self.value), 'tanh', (self,))

        def _backward():
            self.grad += out.grad*(1-(out.value*out.value))
        out._backward = _backward

        return out
    
    def ReLU(self):
        vec = np.copy(self.value)
        vec[vec<0] = 0.0
        out = Vector(vec, 'ReLU', (self,))

        def _backward():
            derivative = np.ones_like(self.value)
            derivative[self.value<0]=0
            self.grad += out.grad*derivative
        out._backward = _backward

        return out
    
    def backward(self):
        topo = []
        visited = set()
        def topo_sort(root: 'Vector'):
            if(root not in visited):
                visited.add(root)
                topo.append(root)
                for child in root.children:
                    topo_sort(child)
        topo_sort(self)

        self.grad = np.ones_like(self.value, dtype=np.float32)
        for node in topo:
            node._backward()


def Vec_hstack(vectors: list):
    """
    -> Operation on a list of vector. To stack them all horizontally.
    -> Intended use: when all outputs from the neurons are to be combined into a single output vector,
                    this can be used to combine them. It will take care of the backward gradient's flow.
    params:
    `vectors` : List of Vector objects
    """
    children = tuple(vectors)
    out = Vector(np.hstack([vec.value for vec in vectors]))
    out.operator = 'hstack'
    out.children = children
    
    def _backward():
        for child_ind in range(len(children)):
            children[child_ind].grad = out.grad[child_ind:child_ind+1]
    out._backward = _backward

    return out

