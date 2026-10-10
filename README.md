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

## Sections:
1. CONCEPT & MATH BEHIND THE CODE: Walk Through of Each Module in a Multilayer Perceptron
2. USING THE CODE: Initiate and Train a Multilayer Perceptron

# CONCEPT & MATH BEHIND THE CODE: Walk Through of Each Module in a Multilayer Perceptron
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
Let us define a model of two hidden layers (meaning there are two pairs of Linear and Activation Layer before the final layer).

The forward pass would be:

$z_1$ = $x \cdot w_1 + b_1$ 

$h_1$ = activation($z_1$)  

$z_2$ = $h_1 \cdot w_2 + b_2$ 

$h_2$ = activation($z_2$)  

$\hat{y}$ = $h_2 \cdot w_3 + b_3$  

Here $\hat{y}$ serves as $z_3$.

The visual representation of this model would be:

<p align='center'>
<img width="640" height="474" alt="fcl" src="https://github.com/user-attachments/assets/494bec1c-ae93-4bda-97fb-020c16a1bc81" />
</p>

### Define Each Variable in Detail:
1. $x$
- Input for training
- Shape: 2D-array of (a, b). 
    - a is the number of samples (e.g., number of patients)
    - b is the number of features of each sample (e.g., each patient has 2 features: age, blood pressure level)
- Visual Representation

$$
\begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,b}\\
x_{2,1} & x_{2,2} & \cdots & x_{2,b}\\
\vdots & \vdots & \vdots & \vdots\\
x_{a,1} & x_{a,2} & \cdots & x_{a,b}\\
\end{bmatrix}
$$

2. $z1$
- Output of the first linear layer. Taking $x$ as an input.
- Shape: 2D-array of (a, c). 
    - a is the number of samples
    - c is the number of perceptrons or neurons in the first linear layer.

- Visual Representation

$$
\begin{bmatrix}
z_{1,1} & z_{1,2} & \cdots & z_{1,c}\\
z_{2,1} & z_{2,2} & \cdots & z_{2,c}\\
\vdots & \vdots & \vdots & \vdots\\
z_{a,1} & z_{a,2} & \cdots & z_{a,c}\\
\end{bmatrix}
$$

- How is it calculated? 
Define the two other variable:
    - $w1$:
        - weights of the perceptrons or neurons in the first linear layer
        - shape: 2D-array (b, c)
            - b is the number of features passed from the input
            - c is the number of perceptrons or neurons in the first linear layer
    
    - $b1$:
        - bias of the perceptrons or neurons in the first linear layer
        - shape: 1D-array (c)
            - c is the number of perceptrons or neurons in the first linear layer

    - Visual Representaion of $w1$ and $b1$:

$$
\begin{bmatrix}
w_{1,1} & w_{1,2} & \cdots & w_{1,c}\\
w_{2,1} & z_{2,2} & \cdots & w_{2,c}\\
\vdots & \vdots & \vdots & \vdots\\
w_{b,1} & z_{b,2} & \cdots & w_{b,c}\\
\end{bmatrix}
$$ 

$$
\begin{bmatrix}
b_{1} & b_{2} & \cdots & b_{c}\\
\end{bmatrix}
$$ 

$z$ = $x \cdot w + b$

$$
\begin{bmatrix}
z_{1,1} & z_{1,2} & \cdots & z_{1,c}\\
z_{2,1} & z_{2,2} & \cdots & z_{2,c}\\
\vdots & \vdots & \vdots & \vdots\\
z_{a,1} & z_{a,2} & \cdots & z_{a,c}\\
\end{bmatrix} = \begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,b}\\
x_{2,1} & x_{2,2} & \cdots & x_{2,b}\\
\vdots & \vdots & \vdots & \vdots\\
x_{a,1} & x_{a,2} & \cdots & x_{a,b}\\
\end{bmatrix}
\cdot
\begin{bmatrix}
w_{1,1} & w_{1,2} & \cdots & w_{1,c}\\
w_{2,1} & z_{2,2} & \cdots & w_{2,c}\\
\vdots & \vdots & \vdots & \vdots\\
w_{b,1} & z_{b,2} & \cdots & w_{b,c}\\
\end{bmatrix} + \begin{bmatrix}
b_{1} & b_{2} & \cdots & b_{c}\\
\end{bmatrix}
$$ 

Calculate $z_{2,1}$, which is the output of the second sample at the first perceptron of the first linear layer.

$z_{2,1}$ = $x_{2,1} \times w_{1,1}  + x_{2,2} \times w_{2,1} + ... + x_{2,b}  \times w_{b,1} + b_{1}$

<p align='center'>
<img width="640" height="440" alt="first_linear_layer" src="https://github.com/user-attachments/assets/89dd63d2-f869-4420-ad83-41f9228364d9" />
</p>

3. $h1$
- Output of the first activation layer. Taking $z1$ as an input.
- Shape: 2D-array of (a, c). Same as $z1$.
    - a is the number of samples
    - c is the number of perceptrons or neurons in the first linear layer.

- Visual Representation

$$
\begin{bmatrix}
h_{1,1} & h_{1,2} & \cdots & h_{1,c}\\
h_{2,1} & h_{2,2} & \cdots & h_{2,c}\\
\vdots & \vdots & \vdots & \vdots\\
h_{a,1} & h_{a,2} & \cdots & h_{a,c}\\
\end{bmatrix}
$$

- Calculation

Let $f$ represent the activation function.

$$
\begin{bmatrix}
h_{1,1} & h_{1,2} & \cdots & h_{1,c}\\
h_{2,1} & h_{2,2} & \cdots & h_{2,c}\\
\vdots & \vdots & \vdots & \vdots\\
h_{a,1} & h_{a,2} & \cdots & h_{a,c}\\
\end{bmatrix} = \begin{bmatrix}
f(z_{1,1}) & f(z_{1,2}) & \cdots & f(z_{1,c})\\
f(z_{2,1}) & f(z_{2,2}) & \cdots & f(z_{2,c})\\
\vdots & \vdots & \vdots & \vdots\\
f(z_{a,1}) & f(z_{a,2}) & \cdots & f(z_{a,c})\\
\end{bmatrix}
$$

2. $z2$
- Output of the second linear layer. Taking $h1$ as an input.
- Shape: 2D-array of (a, d). 
    - a is the number of samples
    - d is the number of perceptrons or neurons in the second linear layer.
- $w2$
    - shape (c, d). The number of perceptrons c in the previous layer is the same as the number of features c passed as input to this current layer.
- $b2$
    - shape (d)
- Calculation:

$$
\begin{bmatrix}
z_{1,1} & z_{1,2} & \cdots & z_{1,d}\\
z_{2,1} & z_{2,2} & \cdots & z_{2,d}\\
\vdots & \vdots & \vdots & \vdots\\
z_{a,1} & z_{a,2} & \cdots & z_{a,d}\\
\end{bmatrix} = \begin{bmatrix}
h_{1,1} & h_{1,2} & \cdots & h_{1,c}\\
h_{2,1} & h_{2,2} & \cdots & h_{2,c}\\
\vdots & \vdots & \vdots & \vdots\\
h_{a,1} & h_{a,2} & \cdots & h_{a,c}\\
\end{bmatrix}
\cdot
\begin{bmatrix}
w_{1,1} & w_{1,2} & \cdots & w_{1,d}\\
w_{2,1} & z_{2,2} & \cdots & w_{2,d}\\
\vdots & \vdots & \vdots & \vdots\\
w_{c,1} & z_{c,2} & \cdots & w_{c,d}\\
\end{bmatrix} + \begin{bmatrix}
b_{1} & b_{2} & \cdots & b_{d}\\
\end{bmatrix}
$$ 

Calculate $z_{2,1}$, which is the output of the second sample at the first perceptron of the second linear layer.

$z_{2,1}$ = $h_{2,1} \times w_{1,1}  + h_{2,2} \times w_{2,1} + ... + h_{2,c}  \times w_{c,1} + b_{1}$

<p align='center'>
<img width="640" height="627" alt="second_linear_layer" src="https://github.com/user-attachments/assets/2f1ae415-8d80-4a83-8d63-e92e0044294a" />
</p>


### Refer to linear.py
Linear Class sets up:

$z_i$ = $x \cdot w_i + b_i$

```python
def __init__(self, input_dim, output_dim):
    self.weights = Random.normal(input_dim, output_dim)
    self.bias = np.zeros(output_dim)
```
The weights is initialized as a 2D numpy array of shape (input_dim, output_dim) with values determined by a Initilization Class (initialization strategy explained later in Advanced Technique Section), in this case, it is the Random Class.

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

So what we want to do, is we want to calculate how much, and in what direction (positive or negative) the ${loss}$ changes as the weight $w_i$ and bias $b_i$ changes. And this the definition of $\frac{\partial {loss}}{\partial {w_i}}$ and $\frac{\partial {loss}}{\partial {b_i}}$, or the GRADIENTS.

If the partial derivative is positive, it means that the slope of the loss is increasing at the current parameter (weight or bias), so we want to make the current parameter smaller by a certain amount so the loss would decrease. Vice versa, if the partial derivative is negative, it means that the slope of the loss is decreasing at the current parameter, so we want to make the current parameter bigger by a certain amount so the loss would decrease. The certain amount is determined by $\frac{\partial {loss}}{\partial {parameter}}$ multiplied by the learning rate {lr}.

$w_i -= {lr} \times \frac{\partial {loss}}{\partial {w_i}}$

$b_i -= {lr} \times \frac{\partial {loss}}{\partial {b_i}}$

### Mechanism of Backpropagation
Now, let us see how do we compute $\frac{\partial {loss}}{\partial {parameter}}$.

#### Set up MLP structure
We will still use the model of two hidden layers as example, the shape of each variable is listed:

1. $z_1$ = $x \cdot w_1 + b_1$

shape: $x$ = (a, b), $w_1$ = (b, c), $b_1$ = (c), $z_1$ = (a, c)
- a is number of samples
- b is the number of features of the sample
- c is the number of perpectrons or neurons in this layer

2. $h_1$ = activation($z_1$)  

shape: $z_1$ = (a, c), $h_1$ = (a, c)

3. $z_2$ = $h_1 \cdot w_2 + b_2$

shape: $h_1$ = (a, c), $w_2$ = (c, d), $b_2$ = (d), $z_2$ = (a, d)
- d is the number of perpectrons or neurons in this layer

4. $h_2$ = activation($z_2$)

shape: $z_2$ = (a, d), $h_2$ = (a, d)

5. $\hat{y}$ = $h_2 \cdot w_3 + b_3$

shape: $h_2$ = (a, d), $w_3$ = (d, 1), $b_3$ = (e, 1), $\hat{y}$ = (a, 1)

And we will say that this is a regression model, which uses the MeanSquareError as loss function:

6. ${loss}$ = $\frac{1}{2}(\hat{y} - y_{val})^2$

shape: $\hat{y}$ = (a, 1), $y_{val}$ = (a, 1), ${loss}$ = (a, 1)

#### Calculate Gradient of the Output of Each Layer (${\frac{\partial {loss}}{\partial {z_i}}}$ and ${\frac{\partial {loss}}{\partial {h_i}}}$)
Backpropagation, unlike foward pass, is computing the partial derivative of ${loss}$ with respect to the output of each layer starting the last layer to the first layer. 

HINT: the shape of $\frac{\partial {loss}}{\partial {parameter}}$ is the shape of the parameter!

1. $\frac{\partial {loss}}{\partial {\hat{y}}}$

$\because {loss}$ = $\frac{1}{2}(\hat{y} - y_{val})^2$

$\therefore {\frac{\partial {loss}}{\partial {\hat{y}}} = 2 \times (\hat{y} - y_{val})}$

check shape: $\frac{\partial {loss}}{\partial {\hat{y}}}$ is (a, 1), same as ${\hat{y}}$


2. $\frac{\partial {loss}}{\partial {h_2}}$

Using the chain rule: $\frac{\partial {loss}}{\partial {h_2}}$ = $\frac{\partial {loss}}{\partial {\hat{y}}} \times \frac{\partial {\hat{y}}}{\partial {h_2}}$

$\because {\hat{y}} = {h_2 \cdot w_3 + b_3}$

$\therefore {\frac{\partial {\hat{y}}}{\partial {h_2}}} = {w_3}$

$\therefore {\frac{\partial {loss}}{\partial {h_2}}} = {\frac{\partial {loss}}{\partial {\hat{y}}}} \cdot {w_3}$

But is this correct? Let us check the shape!

(a, 1) $\cdot$ (d, 1) = ERROR $\neq$ (a, d)

What we do is we can transpose $w_3$ so that $w_3^T$ has a shape of (d,1).

(a, 1) $\cdot$ (1, d) = (a, d), , same as $h_2$ 

$\therefore {\frac{\partial {loss}}{\partial {h_2}}} = {\frac{\partial {loss}}{\partial {\hat{y}}}} \cdot {w_3 ^ T}$

Merely using shape check to modify and determine the equation of ${\frac{\partial {loss}}{\partial {h_i}}}$ looks a bit arbitrary. So in the section, In-Depth Understanding of MLP Gradient Calculation, we will look more in-depth into how we get this equation.

3. $\frac{\partial {loss}}{\partial {z_2}}$

Using the chain rule: $\frac{\partial {loss}}{\partial {z_2}}$ = $\frac{\partial {loss}}{\partial {h_2}} \times \frac{\partial {h_2}}{\partial {z_2}}$

$\because$
$h_2$ = activation($z_2$)

$\therefore$
$\frac{\partial {h_2}}{\partial {z_2}}$ = activation_derivative($z_2$)

$\therefore$
$\frac{\partial {loss}}{\partial {z_2}}$ = $\frac{\partial {loss}}{\partial {h_2}}$ $\times$ activation_derivative($z_2$)

Check Shape:

activation_derivative($z_2$) shape same as activation($z_2$), which is (a, d)

(a,d) $\times$ (a,d) = (a, d), same as $z_2$ 


4. $\frac{\partial {loss}}{\partial {h_1}}$

Same as step 2, as ${\hat{y}}$ is technically ${z_3}$.

$\therefore {\frac{\partial {loss}}{\partial {h_1}}} = {\frac{\partial {loss}}{\partial {z_2}}} \cdot {w_2 ^ T}$

5. $\frac{\partial {loss}}{\partial {z_1}}$

Same as step 3. 

$\therefore$
$\frac{\partial {loss}}{\partial {z_1}}$ = $\frac{\partial {loss}}{\partial {h_1}}$ $\times$ activation_derivative($z_1$)

### Calculate Gradient of the Weights and Bias (${\frac{\partial {loss}}{\partial {w_i}}}$, ${\frac{\partial {loss}}{\partial {b_i}}}$).
Now we calculated $\frac{\partial {loss}}{\partial {z_i}}$ and $\frac{\partial {loss}}{\partial {h_i}}$, lets finally calculate $\frac{\partial {loss}}{\partial {w_i}}$ and $\frac{\partial {loss}}{\partial {b_i}}$.

1. $\frac{\partial {loss}}{\partial {w_3}}$ 

Using the chain rule: $\frac{\partial {loss}}{\partial {w_3}}$ = $\frac{\partial {loss}}{\partial {\hat{y}}} \times \frac{\partial {\hat{y}}}{\partial {w_3}}$

$\because$
$\hat{y}$ = $h_2 \cdot w_3 + b_3$

$\therefore$
${\frac{\partial {\hat{y}}}{\partial {w_3}}} = {h_2}$

$\therefore$
${\frac{\partial {loss}}{\partial {w_3}}} = {\frac{\partial {loss}}{\partial {\hat{y}}}} \cdot {h_2}$

But does the shape match?

(a, 1) $\cdot$ (a, d) $\neq$ (d, 1)

So we can transpose ${h_2}$ so ${h_2 ^ T}$ is (d, a).

(d, a) $\cdot$ (a, 1) = (d, 1), same as ${w_3}$

$\therefore$
${\frac{\partial {loss}}{\partial {w_3}}} = {h_2 ^ T} \cdot {\frac{\partial {loss}}{\partial {\hat{y}}}}$

2. $\frac{\partial {loss}}{\partial {b_3}}$ 

Using the chain rule: $\frac{\partial {loss}}{\partial {b_3}}$ = $\frac{\partial {loss}}{\partial {\hat{y}}} \times \frac{\partial {\hat{y}}}{\partial {b_3}}$

$\because$
$\hat{y}$ = $h_2 \cdot w_3 + b_3$

$\therefore$
${\frac{\partial {\hat{y}}}{\partial {b_3}}} = 1$

$\therefore$
${\frac{\partial {loss}}{\partial {b_3}}} = \frac{\partial {loss}}{\partial {\hat{y}}}$

Check Shape:

(a, 1) = (a, 1), same as $b_3$

We DON'T have to do more calculations, because all $w_i$ and $b_i$ are calculated in the same way. And because $\hat{y}$ is $z_3$, we know both $w_i$ and $b_i$ are related to $z_i$.

So the general formula is:

${\frac{\partial {loss}}{\partial {w_i}}} = {h_{i-1} ^ T} \cdot {\frac{\partial {loss}}{\partial {z_i}}}$.      $h_0$ is $x$

${\frac{\partial {loss}}{\partial {b_i}}} = \frac{\partial {loss}}{\partial {z_i}}$

## Summarize Partial Derivative Formula for $z_i$, $h_i$, $w_i$, $b_i$

${\frac{\partial {loss}}{\partial {h_i}}} = {\frac{\partial {loss}}{\partial {z_{i+1}}}} \cdot {w_{i+1} ^ T}$

$\frac{\partial {loss}}{\partial {z_i}}$ = $\frac{\partial {loss}}{\partial {h_i}}$ $\times$ activation_derivative($z_i$)

${\frac{\partial {loss}}{\partial {w_i}}} = {h_{i-1} ^ T} \cdot {\frac{\partial {loss}}{\partial {z_i}}}$.      $h_0$ is $x$

${\frac{\partial {loss}}{\partial {b_i}}} = \frac{\partial {loss}}{\partial {z_i}}$



## Putting it All Together: nn.py
```python
 def backward(self, lr, decay_rate=1e-4):
        x = self.loss_fn.backward()   

        for layer in reversed(self.layers):
            if isinstance(layer, Linear): 
              dL_dweight = layer.input.T @ x
              dL_dbias = np.sum(x)
              layer.update(lr, dL_dweight, dL_dbias)

              x = x @ layer.weights.T  
            elif isinstance(layer, Activation):   
              x = x * layer.take_derivative()              
```

The gradient calculation for weight and bias shown here is simplified than in the actual file because we didn't cover L2 Regularization and use of Optimizer yet. We will cover that in Advanced Technique Section.

BREAKDOWN:
```python
x = self.loss_fn.backward()   
```
Calculates $\frac{\partial {loss}}{\partial {\hat{y}}}$

```python
if isinstance(layer, Linear): 
```
If we are in a layer of $z_i$ = $h_{i-1} \cdot w_i + b_i$, we know the current x is a $\frac{\partial {loss}}{\partial {z_i}}$.

```python
dL_dweight = layer.input.T @ x
dL_dbias = np.sum(x)

x = x @ layer.weights.T  
```

So we can calculate $\frac{\partial {loss}}{\partial {w_i}}$, $\frac{\partial {loss}}{\partial {b_i}}$, and $\frac{\partial {loss}}{\partial {h_{i-1}}}$. The x is updated as $\frac{\partial {loss}}{\partial {h_i}}$.

```python
layer.update(lr, dL_dweight, dL_dbias)
```
Refer to Linear Class in linear.py

```python
def update(self, lr, gradient_weight, gradient_bias):
    self.weights -= lr * gradient_weight
    self.bias -= lr * gradient_bias
```
The end goal of backpropagation is MET!!!!!!

```python
elif isinstance(layer, Activation):   
    x = x * layer.take_derivative() 
```
Vice Versa process. Current x is $\frac{\partial {loss}}{\partial {h_{i-1}}}$, so new x is updated as $\frac{\partial {loss}}{\partial {z_{i-1}}}$.

### In-Depth Understanding of MLP Gradient Calculation 

#### Why does ${\frac{\partial {loss}}{\partial {h_i}}} = {\frac{\partial {loss}}{\partial {z_{i+1}}}} \cdot {w_{i+1} ^ T}$ ?

Lets first simplify the expression to ${\frac{\partial {loss}}{\partial {h}}} = {\frac{\partial {loss}}{\partial {z}}} \cdot {w^ T}$.

${\frac{\partial {loss}}{\partial {h}}}$ is a matrix of shape (m, n), in which:
- m is the number of samples
- n is the number of features passed as input to the linear ${z}$ layer.

$$
\begin{bmatrix}
{\frac{\partial {loss}}{\partial {h_{1, 1}}}} & {\frac{\partial {loss}}{\partial {h_{1, 2}}}} & \cdots & {\frac{\partial {loss}}{\partial {h_{1, n}}}} \\
{\frac{\partial {loss}}{\partial {h_{2, 1}}}} & {\frac{\partial {loss}}{\partial {h_{2, 2}}}} & \cdots & {\frac{\partial {loss}}{\partial {h_{2, n}}}} \\
\vdots & \vdots & \vdots & \vdots \\
{\frac{\partial {loss}}{\partial {h_{m, 1}}}} & {\frac{\partial {loss}}{\partial {h_{m, 2}}}} & \cdots & {\frac{\partial {loss}}{\partial {h_{m, n}}}}  \\
\end{bmatrix}
$$

${\frac{\partial {loss}}{\partial {z}}}$ is a matrix of shape (m, p), in which:
- m is the number of samples
- p is the number of perceptrons or neurons in this linear ${z}$ layer.

$$
\begin{bmatrix}
{\frac{\partial {loss}}{\partial {z_{1, 1}}}} & {\frac{\partial {loss}}{\partial {z_{1, 2}}}} & \cdots & {\frac{\partial {loss}}{\partial {z_{1, p}}}} \\
{\frac{\partial {loss}}{\partial {z_{2, 1}}}} & {\frac{\partial {loss}}{\partial {z_{2, 2}}}} & \cdots & {\frac{\partial {loss}}{\partial {z_{2, p}}}} \\
\vdots & \vdots & \vdots & \vdots \\
{\frac{\partial {loss}}{\partial {z_{m, 1}}}} & {\frac{\partial {loss}}{\partial {z_{m, 2}}}} & \cdots & {\frac{\partial {loss}}{\partial {z_{m, p}}}}  \\
\end{bmatrix}
$$

$w$ is a matrix of shape (n, p), in which:
- n is the number of features passed as input to the linear ${z}$ layer, as defined above
- p is the number of perceptrons or neurons in this linear ${z}$ layer, as defined above

$$
\begin{bmatrix}
w_{1, 1} & w_{1, 2} & \cdots & w_{1, p} \\
w_{2, 1} & w_{2, 2} & \cdots & w_{2, p} \\
\vdots & \vdots & \vdots & \vdots \\
w_{n, 1} & w_{n, 2} & \cdots & w_{n, p} \\
\end{bmatrix}
$$

How is ${\frac{\partial {loss}}{\partial {h}}}$ and ${\frac{\partial {loss}}{\partial {z}}}$ related by ${w^ T}$? 

Let us take one element $\frac{\partial {loss}}{\partial {h_{1, 1}}}$ as an example. 

We know that according to chain rule $\frac{\partial {loss}}{\partial {h_{1, 1}}}$ = $\frac{\partial {loss}}{\partial {z}} \times \frac{\partial {z}}{\partial {h_{1, 1}}}$

Since ${h_{1, 1}}$, the first feature of sample 1, is an input to ALL of the perceptrons or neurons in the linear layer. The chain rule can be rewritten as the following:

$\frac{\partial {loss}}{\partial {h_{1,1}}}$ = $\frac{\partial {loss}}{\partial {z_{1,1}}} \times \frac{\partial {z_{1,1}}}{\partial {h_{1,1}}}$ + $\frac{\partial {loss}}{\partial {z_{1,2}}} \times \frac{\partial {z_{1,2}}}{\partial {h_{1,1}}}$  + ... + $\frac{\partial {loss}}{\partial {z_{1,p}}} \times \frac{\partial {z_{1,p}}}{\partial {h_{1,1}}}$ 

$\because$
$\frac{\partial {z_{1,k}}}{\partial {h_{1,1}}}$ = $w_{1,k}$

$\therefore$
$\frac{\partial {loss}}{\partial {h_{1,1}}}$ = $\frac{\partial {loss}}{\partial {z_{1,1}}} \times w_{1,1}$ + $\frac{\partial {loss}}{\partial {z_{1,2}}} \times w_{1,2}$  + ... + $\frac{\partial {loss}}{\partial {z_{1,p}}} \times w_{1,p}$ 

Let us write an equation for $\frac{\partial {loss}}{\partial {h_{1, 2}}}$ too.

Since ${h_{1, 2}}$, the second feature of sample 1, is an input to ALL of the perceptrons or neurons in the linear layer. The chain rule can be rewritten as the following:

$\frac{\partial {loss}}{\partial {h_{1,2}}}$ = $\frac{\partial {loss}}{\partial {z_{1,1}}} \times \frac{\partial {z_{1,1}}}{\partial {h_{1,2}}}$ + $\frac{\partial {loss}}{\partial {z_{1,2}}} \times \frac{\partial {z_{1,2}}}{\partial {h_{1,2}}}$  + ... + $\frac{\partial {loss}}{\partial {z_{1,p}}} \times \frac{\partial {z_{1,p}}}{\partial {h_{1,2}}}$ 

$\because$
$\frac{\partial {z_{1,k}}}{\partial {h_{1,2}}}$ = $w_{2,k}$

$\therefore$
$\frac{\partial {loss}}{\partial {h_{1,1}}}$ = $\frac{\partial {loss}}{\partial {z_{1,1}}} \times w_{2,1}$ + $\frac{\partial {loss}}{\partial {z_{1,2}}} \times w_{2,2}$  + ... + $\frac{\partial {loss}}{\partial {z_{1,p}}} \times w_{2,p}$ 

So a general equation for $\frac{\partial {loss}}{\partial {h_{1, f}}}$ is:

$\frac{\partial {loss}}{\partial {h_{1,1}}}$ = $\frac{\partial {loss}}{\partial {z_{1,1}}} \times w_{f,1}$ + $\frac{\partial {loss}}{\partial {z_{1,2}}} \times w_{f,2}$  + ... + $\frac{\partial {loss}}{\partial {z_{1,p}}} \times w_{f,p}$ 

So if we want to construct the matrix for the first sample:

$$
\begin{bmatrix}
{\frac{\partial {loss}}{\partial {h_{1, 1}}}} & {\frac{\partial {loss}}{\partial {h_{1, 2}}}} & \cdots & {\frac{\partial {loss}}{\partial {h_{1, n}}}}\\
\end{bmatrix}
$$

It would be:

$$
\begin{bmatrix}
\frac{\partial {loss}}{\partial {z_{1,1}}} \times w_{1,1} + \frac{\partial {loss}}{\partial {z_{1,2}}} \times w_{1,2} + ... + \frac{\partial {loss}}{\partial {z_{1,p}}} \times w_{1,p}
& 
\frac{\partial {loss}}{\partial {z_{1,1}}} \times w_{2,1} + \frac{\partial {loss}}{\partial {z_{1,2}}} \times w_{2,2} + ... + \frac{\partial {loss}}{\partial {z_{1,p}}} \times w_{2,p}
& 
\cdots 
& 
\frac{\partial {loss}}{\partial {z_{1,1}}} \times w_{n,1} + \frac{\partial {loss}}{\partial {z_{1,2}}} \times w_{n,2} + ... + \frac{\partial {loss}}{\partial {z_{1,p}}} \times w_{n,p}\\
\end{bmatrix}
$$

And this equals:

$$
\begin{bmatrix}
{\frac{\partial {loss}}{\partial {z_{1, 1}}}} & {\frac{\partial {loss}}{\partial {z_{1, 2}}}} & \cdots & {\frac{\partial {loss}}{\partial {z_{1, p}}}}\\
\end{bmatrix}
\cdot
\begin{bmatrix}
w_{1,1} & w_{2,1} & \cdots & w_{n,1}\\
w_{1,2} & w_{2,2} & \cdots & w_{n,2}\\
\vdots & \vdots & \vdots & \vdots \\
w_{1,p} & w_{2,p} & \cdots & w_{n,p}\\
\end{bmatrix}
$$


So a matrix with all samples from 1 to m would be:

$$
\begin{bmatrix}
{\frac{\partial {loss}}{\partial {z_{1, 1}}}} & {\frac{\partial {loss}}{\partial {z_{1, 2}}}} & \cdots & {\frac{\partial {loss}}{\partial {z_{1, p}}}} \\
{\frac{\partial {loss}}{\partial {z_{2, 1}}}} & {\frac{\partial {loss}}{\partial {z_{2, 2}}}} & \cdots & {\frac{\partial {loss}}{\partial {z_{2, p}}}} \\
\vdots & \vdots & \vdots & \vdots \\
{\frac{\partial {loss}}{\partial {z_{m, 1}}}} & {\frac{\partial {loss}}{\partial {z_{m, 2}}}} & \cdots & {\frac{\partial {loss}}{\partial {z_{m, p}}}}  \\
\end{bmatrix}
\cdot
\begin{bmatrix}
w_{1,1} & w_{2,1} & \cdots & w_{n,1}\\
w_{1,2} & w_{2,2} & \cdots & w_{n,2}\\
\vdots & \vdots & \vdots & \vdots \\
w_{1,p} & w_{2,p} & \cdots & w_{n,p}\\
\end{bmatrix}
$$

Note that $w^{T}$ is:

$$
\begin{bmatrix}
w_{1,1} & w_{2,1} & \cdots & w_{n,1}\\
w_{1,2} & w_{2,2} & \cdots & w_{n,2}\\
\vdots & \vdots & \vdots & \vdots \\
w_{1,p} & w_{2,p} & \cdots & w_{n,p}\\
\end{bmatrix}
$$

Therefore:

${\frac{\partial {loss}}{\partial {h}}} = {\frac{\partial {loss}}{\partial {z}}} \cdot {w^ T}$ 


# USING THE CODE: Initiate and Train a MLP
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
