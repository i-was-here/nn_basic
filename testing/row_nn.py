import numpy as np
from nnet.nn.row.nn import Vector, MLP
from nnet.visualize.draw_graph import draw_dot

if __name__=="__main__":
    
    input_size = 3
    layers_config = [4, 4, 1]

    model = MLP(input_size, layers_config)

    sample_inp = Vector(np.array([2.0, 3.0, -1.0]))
    L = model(sample_inp)
    L.backward()
    graph = draw_dot(L)
    graph.render("./testing/images/nn_test", format="png", cleanup=True)