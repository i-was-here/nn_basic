from nnet.grad.col.engine import Vector, Vec_vstack
from nnet.nn.col.nn import MLP
from nnet.visualize.draw_graph import trace, draw_dot

import numpy as np
import time

np.random.seed(42)

HIDDEN_LAYERS = [3, 3, 1]

def col_vector_test(num_exp=100):
    from nnet.nn.col.nn import MLP
    from nnet.optimizers.optimizers import GD
    from nnet.losses.losses_row import mse_loss
    from nnet.grad.col.engine import Vector
    from nnet.visualize.draw_graph import draw_dot
    import numpy as np
    import random
    random.seed(42)

    # hyper params initialization
    epochs = 20
    lr = 0.01

    # data initialization
    x = [
        Vector(np.array([[3.0], [4.0], [5.0],])),
        Vector(np.array([[-5.0], [-2.0], [8.0]])),
        Vector(np.array([[-2.0], [2.5], [-4.5]])),
        Vector(np.array([[2.0], [-1.5], [-4.0]]))
    ]
    y_true = [Vector(np.array([[1]])), Vector(np.array([[-1]])), Vector(np.array([[-1]])), Vector(np.array([[1]]))]
    
    start = time.perf_counter()
    for experiment in range(num_exp):

        # model initialization
        model = MLP(3, HIDDEN_LAYERS)
        optim = GD(model.parameters(), lr)

        # training loop
        for epoch in range(epochs):

            # forward pass / inference
            y_pred = [model(inp) for inp in x]
            loss_val = mse_loss(y_pred, y_true)
            # print(epoch+1, loss_val.value)
            
            # zero grad
            optim.zero_grad()

            # draw graph
            # graph = draw_dot(loss_val)
            # graph.render("testing/images/time_col", format="png", cleanup=True)
            
            # backward pass
            loss_val.backward()

            # update params
            optim.step()


    stop = time.perf_counter()

    # saving graph of nn as image
    # graph = draw_dot(loss_val)
    # graph.render("testing/images/time_col", format="png", cleanup=True)

    print("Average Col Vector way time: ", (stop-start)/num_exp)


if __name__=="__main__":

    col_vector_test(100)
    
    # input_size = 3
    # layers_config = [4, 4, 1]

    # model = MLP(input_size, layers_config)

    # sample_inp = Vector(np.array([[2.0], [3.0], [-1.0]]))
    # x = [
    #     Vector(np.array([[3.0], [4.0], [5.0],])),
    #     Vector(np.array([[-5.0], [-2.0], [8.0]]))
    #     # Vector(np.array([[-2.0], [2.5], [-4.5]])),
    #     # Vector(np.array([[2.0], [-1.5], [-4.0]]))
    # ]

    # L = model(sample_inp)
    # L.backward()
    # graph = draw_dot(L)
    # graph.render("./testing/images/temp1", format="png", cleanup=True)


    