from grad.col.engine import Vector, Vec_vstack, trace, draw_dot
from nn.col.nn import MLP
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
    graph.render("./visualize/temp", format="png", cleanup=True)