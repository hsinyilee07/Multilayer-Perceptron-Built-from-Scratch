# Multilayer-Perceptron-Built-from-Scratch
Multilayer Perceptron implemented from scratch using raw python and numpy. For those who want to learn about the math behind the components of MLP and the optimization strategies in training models.  

# Initialize Model
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

# Data Preparation
```python
from data import data_preparation

data_file_pth = "..."
x_train, y_train, x_test, y_test = data_preparation(data_file_path = data_file_pth)
```

# Train Model
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

#Validate Model
```python
from validate import test, regression_metrics_check

y_train_pred = test(model, x_train)
y_test_pred = test(model, x_test)
regression_metrics_check(y_train_pred, y_train, "Train")
regression_metrics_check(y_test_pred, y_test, "Test")
```