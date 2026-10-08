# Multilayer-Perceptron-Built-from-Scratch
Multilayer Perceptron implemented from scratch using raw python and numpy. For those who want to learn about the math behind the components of MLP and the optimization strategies in training models.  

# Initialize Model
```python
from nn import MLP
from linear import Linear
from activation import ReLU
from loss import MeanSquareError
from optimizer import Adam

model = MLP(Linear(8, 16),
            ReLU(),
            Linear(16,1))

model.assign_loss_fn(MeanSquareError())
model.assign_optimizer(Adam(epsilon = 1e-8, beta1 = 0.9, beta2 = 0.99))
```