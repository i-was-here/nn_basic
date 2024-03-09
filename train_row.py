from nn.row.nn import MLP
from optimizers.optimizers import GD
from losses.losses_row import mse_loss
from grad.row.engine import draw_dot, Vector
import numpy as np
import random
random.seed(42)


if __name__=="__main__":

    # hyper params initialization
    epochs = 20
    lr = 0.01

    # data initialization
    x = [
        Vector(np.array([3.0, 4.0, 5.0])),
        Vector(np.array([-5.0, -2.0, 8.0])),
        Vector(np.array([-2.0, 2.5, -4.5])),
        Vector(np.array([2.0, -1.5, -4.0]))
    ]
    y_true = [Vector(np.array([1])), Vector(np.array([-1])), Vector(np.array([-1])), Vector(np.array([1]))]
    
    # model initialization
    model = MLP(3, [4, 4, 1])
    optim = GD(model.parameters(), lr)

    # training loop
    for epoch in range(epochs):

        # forward pass / inference
        y_pred = [model(inp) for inp in x]
        loss_val = mse_loss(y_pred, y_true)
        print(epoch+1, loss_val.value)

        # zero grad
        optim.zero_grad()
        
        # backward pass
        loss_val.backward()

        # update params
        optim.step()


    # # saving graph of nn as image
    # graph = draw_dot(loss_val)
    # graph.render("nn", format="png", cleanup=True)