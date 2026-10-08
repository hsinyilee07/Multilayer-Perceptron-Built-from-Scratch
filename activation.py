import numpy as np

class Activation:
    def __init__(self):
        self.input = None
    
class ReLU(Activation):
  def __call__(self, x):
      self.input = x[:]
      return np.maximum(0 , x)

  def take_derivative(self):
      return (self.input > 0).astype(float)