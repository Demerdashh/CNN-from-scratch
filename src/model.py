class ReLULayer:
    def __init__(self):
        self.Z = None
    def forward(self, X):
        self.Z = X
        return np.maximum(0, X)
    def backward(self, dA, learning_rate=None):
        dZ = dA.copy()
        dZ[self.Z <= 0] = 0
        return dZ

# Build model
model = SequentialModel()

# conv->relu->pool
model.add(ConvLayer(filters=8, kernel_size=3, stride=1, padding=1))
model.add(ReLULayer())
model.add(PoolLayer(pool_size=2, mode="max", stride=2))

# conv->relu->pool
model.add(ConvLayer(filters=16, kernel_size=3, stride=1, padding=1))
model.add(ReLULayer())
model.add(PoolLayer(pool_size=2, mode="max", stride=2))

# FlattenLayer
model.add(FlattenLayer())

# Fullyconnected->relu
model.add(FullyConnectedLayer(units=64))
model.add(ReLULayer())

# classification layer (output layer)
model.add(FullyConnectedLayer(units=2))  # logits for 2 classes
