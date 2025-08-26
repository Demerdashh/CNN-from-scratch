############### FORWARD-PROP ###################
def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape  # Retrive dimensions from A_prev'shape

    (f, f, n_C_prev, n_C) = W.shape # Retrieve dimensions from W's shape   ((f -- is the filter window size))

    stride = hparameters['stride'] # Retrieve stride from "hparameters"
    pad = hparameters['pad'] # Retrieve pad from "hparameters"

    # Computing the dimensions of the CONV output volume using the formula given above.
    n_H = int(1 + (n_H_prev - f) / stride)  # height 
    n_W = int(1 + (n_H_prev - f) / stride)  # width


    Z = np.zeros((m, n_H, n_W, n_C))  # Initialize the output volume Z with zeros.

    
    A_prev_pad = zero_pad(A_prev, pad)  #  A_prev_pad by padding A_prev

    

    for i in range(m):                                    # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]                        # select ith training example's padded activation
        
        for h in range(n_H):                              # loop over vertical axis of the output volume
            vert_start = h * stride                        # the vertical start of the current "slice"
            vert_end = vert_start + f                     # the vertical  end of the current "slice"

            for w in range(n_W):                      # loop over horizontal axis of the output volume
                horiz_start = w * stride              # the horizontal start of the current "slice"
                horiz_end = horiz_start + f           # the horizontal end of the current "slice"

                for c in range(n_C):                  # loop over channels (= #filters) of the output volume
                        
                    # Use the corners to define the (3D) slice of a_prev_pad
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
                    weights = W[:,:,:,c]
                        
                    biases = b[0,0,0,c]
                        
                    Z[i, h, w, c] = np.sum(a_slice_prev * weights) + biases
                        
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache

#################################
#################################

def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """

    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)

    
    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C_prev))

    for i in range(m):                                 # loop over the training examples
        
        for h in range(n_H):                           # loop on the vertical axis of the output volume
            vert_start = h * stride                    # the vertical start of the current "slice"
            vert_end = vert_start + f                  # the vertical end of the current "slice"
            
            for w in range(n_W):                       # loop on the horizontal axis of the output volume
                horiz_start = w * stride               # the horizontal start of the current "slice"
                horiz_end = horiz_start + f            # the horizontal end of the current "slice"
                
                for c in range(n_C_prev):                   # loop over the channels of the output volume
                    
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c.
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    # Compute the pooling operation on the slice.
                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == 'average':
                        A[i, h, w, c] = np.mean(a_prev_slice)

     # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)
    
    return A, cache




############### BACK-PROP ###################

def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function
    
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()
    
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """

        # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache
    
    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C)= W.shape
    
    # Retrieve information from "hparameters"
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape

    # Initialize dA_prev, dW, db
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)


    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume
                    
                    # the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]

        # Set the ith training example's dA_prev to the unpadded da_prev_pad 
        dA_prev[i, :, :, :] = da_prev_pad[ pad:-pad, pad:-pad, :]

    
    return dA_prev, dW, db

#################################
#################################

def pool_backward(dA, cache, mode="max"):
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    # Retrieve information from cache
    (A_prev, hparameters) = cache
    
    # Retrieve hyperparameters from "hparameters"
    stride = hparameters["stride"]
    f = hparameters["f"]
    
    # Retrieve dimensions from A_prev's shape and dA's shape
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    # Initialize dA_prev with zeros
    dA_prev = np.zeros_like(A_prev)
    
    for i in range(m):  # Loop over the training examples
        
        # Select training example from A_prev
        a_prev = A_prev[i]
        
        for h in range(n_H):  # Loop on the vertical axis
            for w in range(n_W):  # Loop on the horizontal axis
                for c in range(n_C):  # Loop over the channels (depth)
                    
                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        
                        # Use the corners to define the current slice from a_prev
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        
                        # Create the mask from a_prev_slice
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += mask * dA[i, h, w, c]
                        
                    elif mode == "average":
                        
                        # Get the value da from dA
                        da = dA[i, h, w, c]
                        
                        # Define the shape of the filter as fxf
                        shape = (f, f)
                        
                        # Distribute it to get the correct slice of dA_prev
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.ones(shape) * da / (f * f)


    
    return dA_prev
