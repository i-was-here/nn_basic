import numpy as np
from nnet.grad.row.engine import Vector
from nnet.visualize.draw_graph import draw_dot

if __name__=='__main__':

    # inputs
    x1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    x1 = Vector(x1, label='x1')
    x2 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    x2 = Vector(x2, label='x2')

    # weights & biases
    w1 = Vector(np.array([6, 7, 8, 9, 10], dtype=np.float32), label='w1')
    w2 = Vector(np.array([6, 7, 8, 9, 10], dtype=np.float32), label='w2')
    b  = Vector(np.array([2], dtype=np.float32), label='b')

    # apply linear layer
    x1w1 = x1.dot_product(w1); x1w1.label='x1w1'
    x2w2 = x2.dot_product(w2); x2w2.label='x2w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1 + x2w2'
    n = x1w1x2w2 + b; n.label = 'n'

    # loss function
    n_sq = n*n; n_sq.label="n_sq"
    o = n_sq.sum(); o.label = 'o'

    # backward process
    o.backward()

    # graphing out network
    graph = draw_dot(o)
    graph.render("./testing/images/grad_engine_row", format="png", cleanup=True)
