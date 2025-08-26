# conv
class ConvLayer:
    def __init__(self, filters, kernel_size, stride=1, padding=0):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = None  # Initialize weights later
        self.b = None  # Initialize biases later
        self.cache_conv = None # Initialize ConvCache later
        
    def forward(self, A_prev):
        # lazy init weights using A_prev channels
        if self.W is None:
            n_C_prev = A_prev.shape[-1]
            f = self.kernel_size
            n_C = self.filters
            # Xavier / He-ish
            scale = np.sqrt(2. / (f * f * n_C_prev))
            self.W = np.random.randn(f, f, n_C_prev, n_C) * scale
            self.b = np.zeros((1,1,1,n_C))
        Z, self.cache_conv = conv_forward(A_prev, self.W, self.b, {'stride': self.stride, 'pad': self.padding})
        return Z

    def backward(self, dZ, learning_rate):
        m = dZ.shape[0]
        dA_prev, dW, db = conv_backward(dZ, self.cache_conv)
        # update (average gradients by m)
        self.W -= learning_rate * (dW / m)
        self.b -= learning_rate * (db / m)
        return dA_prev

# pool
class PoolLayer:
    def __init__(self, pool_size, mode="max", stride=None):
        self.pool_size = pool_size
        self.mode = mode
        self.stride = stride if stride else pool_size
        self.cache_pool = None
        
    def forward(self, A_prev):
        ####self.cache = A_prev  # Cache input for backward pass
        Z, self.cache_pool = pool_forward(A_prev, {"f": self.pool_size, "stride": self.stride}, self.mode)
        return Z

    def backward(self, dA, learning_rate= None):
        dA_prev = pool_backward(dA, self.cache_pool, self.mode)
        return dA_prev

#flatten
class FlattenLayer:
    def __init__(self):
        self.orig_shape = None

    def forward(self, A_prev):
        self.orig_shape = A_prev.shape
        m = A_prev.shape[0]
        return A_prev.reshape(m, -1)

    def backward(self, dA, learning_rate=None):
        return dA.reshape(self.orig_shape)
# fullyconnected
class FullyConnectedLayer:
    def __init__(self, units):
        self.units = units
        self.W = None  # Initialize weights later
        self.b = None  # Initialize biases later
        self.cache = None
        
    def forward(self, A_prev):
         # A_prev shape (m, d)
        if self.W is None:
            d = A_prev.shape[1]
            # Xavier init
            self.W = np.random.randn(d, self.units) * np.sqrt(2. / d)
            self.b = np.zeros((1, self.units))
        self.cache = A_prev
        Z = A_prev.dot(self.W) + self.b
        return Z

    def backward(self, dZ, learning_rate):
        m = dZ.shape[0]
        A_prev = self.cache
        
        dW = A_prev.T.dot(dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = dZ.dot(self.W.T)
        
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        return dA_prev
