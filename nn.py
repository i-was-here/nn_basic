import random
from grad_engine import Value, draw_dot

class Neuron:
    def __init__(self, num_inp) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_inp)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        assert len(x)==len(self.w), "Input Incompatible with the shape of weights, try changing the `num_inp` param"
        val = sum([wi*xi for wi, xi in zip(self.w, x)], self.b)
        out = val.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, num_inp, num_out):
        self.neurons = [Neuron(num_inp) for _ in range(num_out)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs
    
    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]

class MLP:
    def __init__(self, inp_size: int, neuron_sizes: list):
        all_neurons_sizes = [inp_size] + neuron_sizes
        self.layers = [Layer(all_neurons_sizes[i], all_neurons_sizes[i+1]) for i in range(len(neuron_sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]


if __name__=="__main__":
    
    input_size = 3
    layers_config = [4, 4, 1]

    model = MLP(input_size, layers_config)

    sample_inp = [2.0, 3.0, -1.0]
    L = model(sample_inp)

    graph = draw_dot(L)
    graph.render("nn", format="png", cleanup=True)