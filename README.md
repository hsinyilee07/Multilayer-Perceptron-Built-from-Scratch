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
1. Structure:<br>Alternating Linear & Activation Layers, each layer's output is another layer's input in the Forward Pass described below.

2. Flow:
    1) Forward Pass:<br>input ($x$) -> linear layer 1 ($z_1$) -> activation layer 1 ($h_1$) -> -> linear layer 2 ($z_2$) -> activation layer 2 ($h_2$) -> ...alternating... -> output ($\hat{y}$)

    - $x$: features of sample, input of the first linear layer
    - $z_i$: output value of the linear layer $i$, input value of the activation layer $i$
    - $h_i$: output value of the activation layer $i$, input value of the next linear layer $i + 1$ 
    - $\hat{y}$: final prediction, could be a $z$ or a $h$ depending on whether the last layer is a linear layer or an activation layer

    2) Loss Calculation:<br>uses the output $\hat{y}$ to calculate the loss, which measures how "far" is the predicted result $\hat{y}$ from the actual value $y_{val}$.

    3) Backpropagation:<br>calculate the partial derivative of loss with respect to each of $\hat{y}$, $h_i$, $z_i$. The goal is to use the partial derivative of loss with respect to $z_i$ to calculate the gradient for the weight and bias of each perceptron. 

If you are confused, don't worry! This is just an overview, each module in MLP would be explained in detail below.

## Forward Pass
Let us define a model of two hidden layers.

The forward pass would be:

$z_1$ = $x$ @ $w_1$ + $b_1$
<br>$h_1$ = activation($z_1$)
<br>$z_2$ = $h_1$ @ $w_2$ + $b_2$
<br>$h_2$ = activation($z_2$)
<br>$\hat{y}$ = $h_2$ @ $w_3$ + $b_3$

Here $\hat{y}$ serves as $z_3$.

### Refer to linear.py
Linear Class sets up:

$z_i$ = $x$ @ $w_i$ + $b_i$

```python
def __init__(self, input_dim, output_dim):
    self.weights = Random.normal(input_dim, output_dim)
    self.bias = np.zeros(output_dim)
```
The weights is initialized as a 2D numpy array of shape (input_dim, output_dim) with values determined by a Initilization Class (initialization strategy explained later), in this case, it is the Random Class.

- input_dim: number of features in the input $x$ 
- output_dim: how many perceptrons or neurons are there in this layer

The biases is initialized a 1D numpy array of shape (output_dim) of values of zero.

```python
def __call__(self, x):
    self.input = x[:]
    return x @ self.weights + self.bias
```
Stores the input $x$ for backpropagation (explained later), and called during forward pass to calculate $z_i$.

### Refer to activation.py
Activation Class sets up:

$h_i$ = activation($z_i$)

The below is an example for ReLU activation.
```python
def __call__(self, x):
    self.input = x[:]
    return np.maximum(0 , x)
```
Stores the input $x$ for backpropagation (explained later), and called during forward pass to calculate $h_i$.

## Loss Calculation
The loss calculates how "far" the prediction is from the actual value. Different applications use different loss equations.

### Application 1: Regression
Refer to loss.py, MeanSquareError Class
```python
def __call__(self, pred, val):
    self.pred = pred
    self.val = val
    return np.mean((pred - val)**2)
```
${loss}$ = $\frac{1}{2}(\hat{y} - y_{val})^2$ is computed.

## Backpropagation
This is the most confusing part of a Neural Network! But hang in there, we can do it!

### Goal of Backpropagation
Before we go into the mechanism of backpropagation, let us understand what does it do.

The entire goal of training a neural network is telling each perceptron or neuron how to change its weight and bias to yield a prediction $\hat{y}$ that is closest to the actual value $y_{val}$. In other words, it wants to adjust its weight and bias so the loss is minimized.

So what we want to do, is we want to calculate how much, and in what direction (positive or negative) the ${loss}$ changes as the weight $w_i$ and bias $b_i$ changes. And this the definition of $\frac{\partial {loss}}{\partial {w_i}}$ and $\frac{\partial {loss}}{\partial {b_i}}$.

If the partial derivative is positive, it means that the slope of the loss is increasing at the current parameter (weight or bias), so we want to make the current parameter smaller by a certain amount so the loss would decrease. Vice versa, if the partial derivative is negative, it means that the slope of the loss is decreasing at the current parameter, so we want to make the current parameter bigger by a certain amount so the loss would decrease. The certain amount is determined by $\frac{\partial {loss}}{\partial {parameter}}$ multiplied by the learning rate {lr}.

$w_i -= {lr} \times \frac{\partial {loss}}{\partial {w_i}}$

$b_i -= {lr} \times \frac{\partial {loss}}{\partial {b_i}}$

### Mechanism 
Now, let us see how do we compute $\frac{\partial {loss}}{\partial {parameter}}$.

We will still use the model of two hidden layers as example, the shape of each variable is listed:

$$\mathbf{z_1 = x \cdot w_1 + b_1}$$

$z_1$ = $x$ @ $w_1$ + $b_1

shape: $x$ = (a, b), $w_1$ = (b, c), $b_1$ = (c), $z_1$ = (a, c)
- a is number of samples
- b is the number of features of the sample
- c is the number of perpectrons or neurons in this layer

$h_1$ = activation($z_1$)  

shape: $z_1$ = (a, c), $h_1$ = (a, c)

$z_2$ = $h_1$ @ $w_2$ + $b_2$

shape: $h_1$ = (a, c), $w_2$ = (c, d), $b_2$ = (d), $z_2$ = (a, d)
- d is the number of perpectrons or neurons in this layer

$h_2$ = activation($z_2$)

shape: $z_2$ = (a, d), $h_2$ = (a, d)

$\hat{y}$ = $h_2$ @ $w_3$ + $b_3$

shape: $h_2$ = (a, d), $w_3$ = (d, 1), $b_3$ = (e, 1), $\hat{y}$ = (a, 1)

And we will say that this is a regression model, which uses the MeanSquareError as loss function:

${loss}$ = $\frac{1}{2}(\hat{y} - y_{val})^2$

shape: $\hat{y}$ = (a, 1), $y_{val}$ = (a, 1), ${loss}$ = (a, 1)

Backpropagation, unlike foward pass, is computing the partial derivative of ${loss}$ with respect to the output of each layer starting the last layer to the first layer. 

HINT: the shape of $\frac{\partial {loss}}{\partial {parameter}}$ is the shape of the parameter!

1. $\frac{\partial {loss}}{\partial {\hat{y}}}$

${loss}$ = $\frac{1}{2}(\hat{y} - y_{val})^2$

$\frac{\partial {loss}}{\partial {\hat{y}}} = 2 \times (\hat{y} - y_{val})$

2. $\frac{\partial {loss}}{\partial {\hat{h_2}}}$

using the chain rule: $\frac{\partial {loss}}{\partial {\hat{h_2}}}$ = $\frac{\partial {loss}}{\partial {\hat{y}}} \times \frac{\partial {\hat{y}}}{\partial {\hat{h_2}}}$






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