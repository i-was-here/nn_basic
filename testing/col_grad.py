import numpy as np
from nnet.grad.col.engine import Vector
from nnet.visualize.draw_graph import draw_dot

if __name__=='__main__':

    # inputs
    # x1 = Value(2.0, label='x1')
    x1 = np.array([[1], [2], [3], [4], [5]])
    x1 = Vector(x1, label='x1')
    x2 = np.array([[1], [2], [3], [4], [5]])
    x2 = Vector(x2, label='x2')

    # weights & biases
    w1 = Vector(np.array([[6], [7], [8], [9], [10]]), label='w1')
    w2 = Vector(np.array([[6], [7], [8], [9], [10]]), label='w2')
    b  = Vector(np.array([[2]]), label='b')

    # apply linear layer
    x1w1 = x1.dot_product(w1); x1w1.label='x1w1'
    x2w2 = x2.dot_product(w2); x2w2.label='x2w2'
    print(x1w1, x2w2)
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1 + x2w2'
    n = x1w1x2w2 + b; n.label = 'n'
    print(n)

    # # loss function
    # o = n.tanh(); o.label = 'o'
    # o.grad = 1

    # backward process
    n.backward()

    # graphing out network
    graph = draw_dot(n)
    graph.render("./images/grad_engine_vector", format="png", cleanup=True)