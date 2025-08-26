def train(model, X_train, Y_train, epochs, learning_rate):
    for epoch in range(epochs):
        # Forward pass
        A = X_train
        caches_for_relu = []   # we'll use a simple pattern: conv->relu->pool->...; so we re-run forward with activations here
        # To keep it simple: do forward with activations inline:
        A = model.forward(X_train)  # final logits (note: our model architecture includes ReLU layers explicitly)
        loss = cross_entropy_loss(A, Y_train)

        # Backward pass
        dA = cross_entropy_grad(A, Y_train)
        model.backward(dA, learning_rate)

        print(f"Epoch {epoch+1}/{epochs}  Loss: {loss:.5f}")
