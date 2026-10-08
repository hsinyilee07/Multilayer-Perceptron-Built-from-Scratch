import numpy as np

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
