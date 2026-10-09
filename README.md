# Multilayer-Perceptron-Built-from-Scratch
Multilayer Perceptron implemented from scratch using raw python and numpy. For those who want to learn about the math behind the components of MLP and the optimization strategies in training models.  

## Installation Guide
Install the following libraries:
```bash
pip install numpy 
pip install matplotlib
pip install pandas
pip install scikit-learn
```

# Walk Through of Each Module of a Multilayer Perceptron
## Sections:
1. Overview
2. Forward Pass
3. Loss Calculation
4. Backpropagation

## Overview
Multilayer Perceptron (MLP) is the most basic architecture of neural network and deep learning.

Applications Include:
1. Regression models: predict continuous numerical value 
2. Binary Classification models: categorizes sample to two classes 
3. Multi-class Classification models: categorizes sample to multiple classes

### Modules in MLP
1. Structure:\ 
Alternating Linear & Activation Layers, each layer's output is another layer's input in the Forward Pass described below.

2. Flow:
1) Forward Pass: 
input ($x$) -> linear layer 1 ($z_1$) -> activation layer 1 ($h_1$) -> -> linear layer 2 ($z_2$) -> activation layer 2 ($h_2$) -> ...alternating... -> output ($\hat{y}$)

- $x$: features of sample, input of the first linear layer
- $z_i$: output value of the linear layer $i$, input value of the activation layer $i$
- $h_i$: output value of the activation layer $i$, input value of the next linear layer $i + 1$ 
- $\hat{y}$: 
final prediction, could be a $z$ or a $h$ depending on whether the last layer is a linear layer or an activation layer

2) Loss Calculation: 
uses the output $\hat{y}$ to calculate the loss, which measures how "far" is the predicted result $\hat{y}$ from the actual value $y val$.

3) Backpropagation: 
calculate the partial derivative of loss with respect to each of $\hat{y}$, $h_i$, $z_i$. The goal is to use the partial derivative of loss with respect to $z_i$ to calculate the gradient for the weight and bias of each perceptron. 

More math and calculation details of the modules in MLP would be explained below.


## Forward Pass


## Back propagation 

## Putting it all together: nn.py


# Putting It All Together: Initiate and Train a Model
## Sections:
1. Initiate Model
2. Data Preparation
3. Train Model
4. Validate Model

## Initiate Model
```python
from nn import MLP
from linear import Linear
from activation import ReLU
from loss import MeanSquareError
from optimizer import Adam

#initiate model
model = MLP(Linear(8, 16),
            ReLU(),
            Linear(16,1))

#create and assign loss function
model.assign_loss_fn(MeanSquareError())  

#create and assign optimizer object
model.assign_optimizer(Adam(epsilon = 1e-8, beta1 = 0.9, beta2 = 0.99))
```

## Data Preparation
```python
from data import data_preparation

data_file_pth = "..."
x_train, y_train, x_test, y_test = data_preparation(data_file_path = data_file_pth)
```

## Train Model
```python
from train import train
from plots import plot_accumulated_losses

#Hyperparameters
lr = 0.001
epochs = 500
batch_size = 32
decay_rate = 1e-4

#train
train_losses, test_losses = train(model, x_train, y_train, x_test, y_test, lr, epochs, batch_size, decay_rate)

#plot losses
plot_accumulated_losses(train_losses, test_losses)
```
## Validate Model
```python
from validate import test, regression_metrics_check

y_train_pred = test(model, x_train)
y_test_pred = test(model, x_test)
regression_metrics_check(y_train_pred, y_train, "Train")
regression_metrics_check(y_test_pred, y_test, "Test")
```