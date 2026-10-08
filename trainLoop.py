def train(model, x_train, y_train, x_test, y_test, learning_rate, epochs, batch_size = 32, decay_rate = 1e-4): 
  train_losses = []
  test_losses = []

  num_train_samples = x_train.shape[0]
  num_batches = num_train_samples // batch_size


  for epoch in range(1, epochs+1): 
    train_loss_epoch = 0

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        if end_idx > num_train_samples:
            end_idx = num_train_samples

        x_batch = x_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]

        # Train on train set
        y_pred_train = model.forward(x_batch)
        loss_train = model.loss(y_pred_train, y_batch)

        model.backward(learning_rate, decay_rate)
        train_loss_epoch += loss_train  

    avg_train_loss_epoch = train_loss_epoch/num_batches
    train_losses.append(avg_train_loss_epoch)
    model.optimizer.increment_t()

    # Evaluate on test set
    y_pred_test = model.forward(x_test)
    loss_test = model.loss(y_pred_test, y_test)
    test_losses.append(loss_test)

    print(f'Epoch {epoch}, Train Loss: {avg_train_loss_epoch:.4f}, Test Loss: {loss_test:.4f}')
  
  return train_losses, test_losses