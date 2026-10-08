import matplotlib.pyplot as plt

def plot_accumulated_losses(train_losses_history, test_mses_history):
    plt.plot(train_losses_history, label='Train MSE', color = "blue")
    plt.plot(test_mses_history, label='Test MSE', color = "red")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Cumulative Training and Test Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
