import numpy as np

def test(model, x):
    y_pred = model.forward(x)
    return y_pred

def regression_metrics_check(y_pred, y_val, data_catgory):
    #MSE: Mean Squared Error
    mse = np.mean((y_pred - y_val)**2)

    #RMSE: Root Mean Sqared Error
    rmse = np.sqrt(mse)

    #MAE: Mean Absolute Error (coefficient of deterination)
    mae = np.mean(abs(y_pred - y_val))

    #r-squared
    residuals = np.sum((y_pred - y_val) ** 2)
    y_mean = np.mean(y_val)
    squares = np.sum((y_val - y_mean) ** 2)
    r_squared = 1 - (residuals/squares)

    print(f"{data_catgory} Mean Squared Error: {mse :.4f}")
    print(f"{data_catgory} Root Mean Squared Error: {rmse :.4f}")
    print(f"{data_catgory} Mean Absolute Error: {mae :.4f}")
    print(f"{data_catgory} R-Squared: {r_squared :.4f}")

    return mse, rmse, mae, r_squared