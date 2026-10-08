import numpy as np

#class Initializatoin
rng = np.random.default_rng()

class Initialization:
    pass

class Kaiming(Initialization):   #for linear layers before relu activation
    def normal(input_dim, output_dim):
        weights = rng.normal(loc=0.0, scale = np.sqrt(2.0/input_dim), size = (input_dim, output_dim))
        return weights
    
    def uniform(input_dim, output_dim):
        bound = np.sqrt(6.0 / input_dim)
        weights = rng.uniform(-bound, bound, size = (input_dim, output_dim))
        return weights

class Random(Initialization):   #for linear layers in shallow models
    def normal(input_dim, output_dim):
        weights = rng.normal(loc=0.0, scale = 1, size = (input_dim, output_dim))       # this works too: weights = rng.standard_normal((input_dim, output_dim))
        return weights


#class Linear
class Linear:
    def __init__(self, input_dim, output_dim):
        #initialize weights and bias
        # self.weights = Kaiming.uniform(input_dim, output_dim)
        self.weights = Random.normal(input_dim, output_dim)
        self.bias = np.zeros(output_dim)

        #initialize for optimizer
        self.weights_m = np.zeros((input_dim, output_dim))
        self.weights_v = np.zeros((input_dim, output_dim))
        self.bias_m = np.zeros(output_dim)
        self.bias_v = np.zeros(output_dim)

        self.input = None

    def __call__(self, x):
        self.input = x[:]
        return x @ self.weights + self.bias

    def update(self, lr, gradient_weight, gradient_bias):
        self.weights -= lr * gradient_weight
        self.bias -= lr * gradient_bias
        
#Class Activation Function
class Activation:
    def __init__(self):
        self.input = None
    
class ReLU(Activation):
  def __call__(self, x):
      self.input = x[:]
      return np.maximum(0 , x)

  def take_derivative(self):
      return (self.input > 0).astype(float)

#Class Loss
class Loss:
    def __init__(self):
        self.pred = None
        self.val = None

class MeanSquareError(Loss):
    def __call__(self, pred, val):
      self.pred = pred
      self.val = val
      return np.mean((pred - val)**2)

    def backward(self):
        return 2 * (1/self.pred.shape[0]) * (self.pred - self.val)  #dL/dlast_layer because pred is the output of the last layer (z or h); wrote this part wrong, used np.mean before, but np.mean gives a value, not a (a,1) array

#Class Optimizer
class Optimizer:
    pass

class Adam(Optimizer):
  def __init__(self, epsilon = 1e-8, beta1 = 0.9, beta2 = 0.99):
      self.t = 1
      self.epsilon = epsilon
      self.beta1 = beta1
      self.beta2 = beta2

  def __call__(self, p_grad, p_m, p_v):
      #1) p_m and p_v are the weighted weight of the historical weight with curret wieght
      new_p_m = self.beta1 * p_m + (1-self.beta1) * p_grad
      new_p_v = self.beta2 * p_v + (1-self.beta2) * p_grad ** 2

      #2) adjust p_m and p_v to prevent cold start
      p_m_hat = new_p_m / (1 - self.beta1 ** self.t) 
      p_v_hat = new_p_v / (1 - self.beta2 ** self.t)

      #Adjust the gradient using formula
      updated_gradient = p_m_hat / (p_v_hat ** 0.5 + self.epsilon)

      return updated_gradient, new_p_m, new_p_v

  def increment_t(self):   #USE IN TRAIN LOOP
      self.t += 1
    
#class MLP
class MLP:
    def __init__(self, *args):
        self.layers = args
        self.loss_fn = None
        self.optimizer = None

    def assign_loss_fn(self, loss_obj):
        self.loss_fn = loss_obj

    def assign_optimizer(self, optimizer):
        self.optimizer = optimizer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def loss(self, pred, val):
        return self.loss_fn(pred, val)

    def backward(self, lr, decay_rate=1e-4):
        x = self.loss_fn.backward() #x is dL/dlast layer

        for layer in reversed(self.layers):
            if isinstance(layer, Linear):  #x is currently dL/dz
              #1. GRADIENT CALCULATION & UPDATE FOR WEIGHT & BIAS
              #1)calculate normal gradient
              dL_dweight = layer.input.T @ x
              dL_dbias = np.sum(x)

              #2)adjust gradient of weight with L2 Regularization (loss + 1/2 * decay_rate * weight**2)
              dL_dweight += decay_rate * layer.weights

              #3)adjust gradient using optimizer
              dL_dweight, layer.weights_m, layer.weights_v = self.optimizer(dL_dweight, layer.weights_m, layer.weights_v)
              dL_dbias, layer.bias_m, layer.bias_v = self.optimizer(dL_dbias, layer.bias_m, layer.bias_v)

              #4)update gradient to the actual weight
              layer.update(lr, dL_dweight, dL_dbias)

              #2. GRADIENT CALCULATION FOR OUTPUT OF THE ACTIVATION LAYER BEFORE THIS LAYER
              x = x @ layer.weights.T   #x is now dL/dh

            elif isinstance(layer, Activation):   #x is currently dL/dh
              #1. GRADIENT CALCULATION FOR OUTPUT OF THE LINEAR LAYER BEFORE THIS LAYER
              x = x * layer.take_derivative()     #x is now dL/dz
                   


