import numpy as np

#Initializatoin Methods
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


#Linear Class
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