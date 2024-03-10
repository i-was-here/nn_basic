from nnet.grad.val.engine import Value
from nnet.visualize.draw_graph import draw_dot

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