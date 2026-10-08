import numpy as np
from linear import Linear
from activation import Activation

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
        #dL/dz means dLoss/d output of linear layer; dL/dh means dLoss/d output of 
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
                   


