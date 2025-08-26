def relu(Z):
    return np.maximum(0, Z)

def relu_backward(dA, Z):
    dZ = dA.copy()
    dZ[Z <= 0] = 0
    return dZ

def softmax(Z):
    shift = Z - np.max(Z, axis=1, keepdims=True)
    exps = np.exp(shift)
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy_loss(Z, Y):
    # Z: logits shape (m, C); 
    # Y: (m,1) integer labels 0..C-1
    m = Z.shape[0]
    P = softmax(Z)
    C = Z.shape[1]
    Y_onehot = np.eye(C)[Y.reshape(-1)]
    loss = -np.mean(np.sum(Y_onehot * np.log(P + 1e-15), axis=1))
    return loss

def cross_entropy_grad(Z, Y):
    m = Z.shape[0]
    P = softmax(Z)
    C = Z.shape[1]
    Y_onehot = np.eye(C)[Y.reshape(-1)]
    dZ = (P - Y_onehot) / m
    return dZ
