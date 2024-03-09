from grad.row.engine import Vector, Vec_hstack, trace, draw_dot
import numpy as np

np.random.seed(42)

class Neuron():
    def __init__(self, num_inp):
        self.w = Vector(np.random.random((num_inp)))
        self.b = Vector(np.random.random((1)))
    
    def __call__(self, x: np.ndarray) -> Vector:
        assert len(x.value) == len(self.w.value), "Input is NOT of the correct size ! Incompatible w/ weights"
        val = self.w.dot_product(x)+self.b
        out = val.tanh()
        return out
    
    def parameters(self):
        return [self.w, self.b]

class LinearLayer():
    def __init__(self, num_inp, num_out):
        self.neurons = [Neuron(num_inp) for _ in range(num_out)]
    
    def __call__(self, x):
        outs = Vec_hstack([neuron(x) for neuron in self.neurons])
        return outs
    
    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]

class MLP():
    def __init__(self, inp_size: int, neuron_sizes: list):
        all_neuron_sizes = [inp_size] + neuron_sizes
        self.layers = [LinearLayer(all_neuron_sizes[i], all_neuron_sizes[i+1]) for i in range(len(neuron_sizes))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        # SUS
        return [params for layer in self.layers for params in layer.parameters()]


if __name__=="__main__":
    
    input_size = 3
    layers_config = [4, 4, 1]

    model = MLP(input_size, layers_config)

    sample_inp = Vector(np.array([2.0, 3.0, -1.0]))
    L = model(sample_inp)
    L.backward()
    graph = draw_dot(L)
    graph.render("./images/nn_test", format="png", cleanup=True)