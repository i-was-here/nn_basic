import time

HIDDEN_LAYERS = [6, 6, 1]

def row_vector_test(num_exp=100):
    from nn.row.nn import MLP
    from optimizers.optimizers import GD
    from losses.losses_row import mse_loss
    from grad.row.engine import Vector
    from visualize.draw_graph import draw_dot
    import numpy as np
    import random
    random.seed(42)

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
            
            # backward pass
            loss_val.backward()

            # update params
            optim.step()


        # # saving graph of nn as image
        # graph = draw_dot(loss_val)
        # graph.render("nn", format="png", cleanup=True)
    stop = time.perf_counter()
    print("Average Row Vector way time: ", (stop-start)/num_exp)


def col_vector_test(num_exp=100):
    from nn.col.nn import MLP
    from optimizers.optimizers import GD
    from losses.losses_row import mse_loss
    from grad.col.engine import Vector
    from visualize.draw_graph import draw_dot
    import numpy as np
    import random
    random.seed(42)

    # hyper params initialization
    epochs = 20
    lr = 0.01

    # data initialization
    x = [
        Vector(np.array([[3.0], [4.0], [5.0],])),
        # Vector(np.array([[-5.0], [-2.0], [8.0]])),
        # Vector(np.array([[-2.0], [2.5], [-4.5]])),
        # Vector(np.array([[2.0], [-1.5], [-4.0]]))
    ]
    y_true = [Vector(np.array([[1]]))]#, Vector(np.array([[-1]])), Vector(np.array([[-1]])), Vector(np.array([[1]]))]
    
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
            
            # backward pass
            loss_val.backward()

            # update params
            optim.step()


        # # saving graph of nn as image
        # graph = draw_dot(loss_val)
        # graph.render("nn", format="png", cleanup=True)
    stop = time.perf_counter()
    print("Average Col Vector way time: ", (stop-start)/num_exp)

def value_test(num_exp=100):
    from nn.val.nn import MLP
    from optimizers.optimizers import GD
    from losses.losses import mse_loss
    from visualize.draw_graph import draw_dot
    import random
    random.seed(42)

    # hyper params initialization
    epochs = 20
    lr = 0.01

    # data initialization
    x = [
        [3.0, 4.0, 5.0],
        [-5.0, -2.0, 8.0],
        [-2.0, 2.5, -4.5],
        [2.0, -1.5, -4.0]
    ]
    y_true = [1, -1, -1, 1]

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
            
            # backward pass
            loss_val.backward()

            # update params
            optim.step()


        # # saving graph of nn as image
        # graph = draw_dot(loss_val)
        # graph.render("nn", format="png", cleanup=True)
    stop = time.perf_counter()
    print("Average Value way time: ", (stop-start)/num_exp)

if __name__=="__main__":
    value_test(100)
    print()
    col_vector_test(100)
    print()
    row_vector_test(100)