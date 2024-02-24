import math
from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.children:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        dot.node(name=str(id(n)), label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.value, n.grad), shape='record')
        if n.operator:
            dot.node(name=str(id(n)) + n.operator, label=n.operator)
            dot.edge(str(id(n)) + n.operator, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2.operator)
    
    return dot

class Value():
    def __init__(self, value, operator='', children=(), label=''):
        self.value = value
        self.operator = operator
        self.children = children
        self.grad = 0
        self.label = label
        self._backward = lambda: None
    
    
    def __repr__(self) -> str:
        return "Value: "+str(self.value)
    
    def __str__(self) -> str:
        return self.__repr__()
    
    
    def __mul__(self, another: 'Value') -> 'Value':
        if(not isinstance(another, Value)): another = Value(another)
        out = Value(self.value*another.value, '*', (self, another))
        
        def _backward():
            self.grad += (another.value)*(out.grad)
            another.grad += (self.value)*(out.grad)
        out._backward = _backward

        return out
    
    def __rmul__(self, another: 'Value') -> 'Value':
        if(not isinstance(another, Value)): another = Value(another)
        out = Value(another.value*self.value, '*', (another, self))

        def _backward():
            self.grad += (out.grad)*(another.value)
            another.grad += (out.grad)*(self.value)
        out._backward = _backward

        return out
    

    def __add__(self, another: 'Value') -> 'Value':
        if(not isinstance(another, Value)): another = Value(another)
        out = Value(self.value+another.value, '+', (self, another))

        def _backward():
            self.grad += out.grad
            another.grad += out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, another: 'Value') -> 'Value':
        if(not isinstance(another, Value)): another = Value(another)
        out = Value(another.value+self.value, '+', (another, self))

        def _backward():
            self.grad += out.grad
            another.grad += out.grad
        out._backward = _backward

        return out
    
    
    def __truediv__(self, another: 'Value') -> 'Value':
        if(not isinstance(another, Value)): another = Value(another)
        out = Value((self.value)/(another.value), '/', (self, another))

        def _backward():
            self.grad += out.grad/(another.value)
            another.grad += ((out.value)*(-1*(out.grad)))/(another.value)
        out._backward = _backward

        return out
    
    def __rtruediv__(self, another: 'Value') -> 'Value':
        if(not isinstance(another, Value)): another = Value(another)
        out = Value((another.value)/(self.value), '/', (another, self))

        def _backward():
            another.grad += out.grad/(self.value)
            self.grad += ((out.value)*(-1*(out.grad)))/(self.value)
        out._backward = _backward
        
        return out
    
    
    def __sub__(self, another: 'Value') -> 'Value':
        if(not isinstance(another, Value)): another = Value(another)
        out = Value(self.value-another.value, '-', (self, another))

        def _backward():
            self.grad += out.grad
            another.grad += -1*(out.grad)
        out._backward = _backward
        
        return out
    
    def __rsub__(self, another: 'Value') -> 'Value':
        if(not isinstance(another, Value)): another = Value(another)
        out = Value(another.value-self.value, '-', (another, self))

        def _backward():
            self.grad += -1*(out.grad)
            another.grad += out.grad
        out._backward = _backward

        return out
    

    def __pow__(self, b):
        assert isinstance(b, (int, float)), "Only allowed datatypes of the power are Int and Float"
        out = Value((self.value)**b, operator='**'+str(b), children=(self, ))

        def _backward():
            self.grad = b*((self.value)**(b-1))*out.grad
        out._backward = _backward

        return out
    
    
    def tanh(self):
        a1 = math.exp(2*(self.value))
        out = Value((a1-1)/(a1+1), 'tanh', (self, ))

        def _backward():
            self.grad += (1 - (out.value)**2)*(out.grad)
        out._backward = _backward

        return out

    def exp(self):
        out = Value(math.exp(self.value), 'exp', (self, ))

        def _backward():
            self.grad += (out.grad)*(out.value)
        out._backward = _backward

        return out
    

    def __neg__(self):
        return self * -1
    
    def backward(self):
        topo = []
        visited = set()
        def topo_sort(root: 'Value'):
            if(root not in visited):
                visited.add(root)
                topo.append(root)
                for child in root.children:
                    topo_sort(child)
        topo_sort(self)

        self.grad = 1.0
        for node in topo:
            node._backward()


if __name__=='__main__':

    # inputs
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    # weights & biases
    w1 = Value(-3.0, label='w1')
    w2 = Value(-1.0, label='w2')
    b  = Value(6.88, label='b')

    # apply linear layer
    x1w1 = x1*w1; x1w1.label='x1w1'
    x2w2 = x2*w2; x2w2.label='x2w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1 + x2w2'
    n = x1w1x2w2 + b; n.label = 'n'

    # loss function
    o = n.tanh(); o.label = 'o'
    o.grad = 1

    # backward process
    o.backward()

    # graphing out network
    graph = draw_dot(o)
    graph.render("./images/grad_engine", format="png", cleanup=True)