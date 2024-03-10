from nnet.grad.col.engine import Vector, Vec_vstack
from nnet.nn.col.nn import MLP
from nnet.visualize.draw_graph import trace, draw_dot
import numpy as np

np.random.seed(42)

if __name__=="__main__":
    
    input_size = 3
    layers_config = [4, 4, 1]

    model = MLP(input_size, layers_config)

    sample_inp = Vector(np.array([[2.0], [3.0], [-1.0]]))
    L = model(sample_inp)
    L.backward()
    graph = draw_dot(L)
    graph.render("./testing/images/temp1", format="png", cleanup=True)