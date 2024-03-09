# Deep Learning Module

## This is a very simple implementation of Deep Learning Module which can be used to train neural networks with small architectures.

### Features
- Implemented reverse mode Automatic Differentiation form scratch
- Implemented various optimizers like Momentum based Nesterov Accelerated Gradient Descent
- Optimized the framework by adding in the support for vectors as the atomic unit of parameters.

### Future works
- Incorporating various other Layers like Convolution, BatchNorm, Attention, etc.
- Introduce other state-of-the-art optimizers like Adam, Adagrad.
- Introduce other loss functions like binary cross-entropy.

### Note
> Use the framework after doing an editable install of the project as : `pip install -e .`

### Sample Training
```python
from nnet.nn.val.nn import MLP
from nnet.optimizers.optimizers import SGD
from nnet.losses.losses import mse_loss
import random
random.seed(42)


if __name__=="__main__":

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
    
    # model initialization
    model = MLP(3, [4, 4, 1])
    optim = SGD(model.parameters(), lr, 0.9, nestrov=True)

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
```

### Credits
Motivation: [Andrej Karpathy's repo](https://github.com/karpathy/micrograd)