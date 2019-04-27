from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    """newdim = 1
    i = 1
    while i < len(x.shape):
        newdim *= x.shape[i]
        i += 1
    x_trans = np.reshape(x, newshape = (x.shape[0], newdim))
    z = np.matmul(x_trans, w) + b
    out = z"""
    x_trans = x.reshape(x.shape[0], -1)
    #print(x_trans.shape)
    #print(w.shape)
    #print(b.shape)
    #mat = np.matmul(x_trans, w)
    out = np.matmul(x_trans, w) + b.T
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    """newdim = 1
    i = 1
    while i < len(x.shape):
        newdim *= x.shape[i]
        i += 1
    x_trans = np.reshape(x, newshape = (x.shape[0], newdim))
    dw = np.matmul(x_trans.transpose(), dout)
    dx = np.matmul(dout, w.transpose())
    dx = np.reshape(dx, newshape = x.shape)
    ones = np.ones(shape = (x.shape[0],))
    db = np.matmul(dout.transpose(), ones)"""
    x_trans = x.reshape(x.shape[0], -1)
    dx = np.dot(dout, np.transpose(w))
    #print(dx.shape)
    dx = dx.reshape(x.shape)
    dw = np.dot(np.transpose(x_trans), dout)
    db = np.sum(dout, axis = 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    result = lambda y: y * (y > 0).astype(float)
    out = result(x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    backrelu = lambda d, y: d * (y >= 0)
    dx = backrelu(dout, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """

    loss = 0.0
    dx = None
    ###########################################################################
    # TODO: Implement the softmax loss                                        #
    ###########################################################################
    """for i in range(x.shape[0]):
        top = np.exp(y[i] - np.max(x[i]))
        bottom = np.sum(np.exp(x[i] - np.max(x[i])))
        loss += np.log(top/bottom)
    dx = np.sum([loss - i for i in y])"""
    top = np.exp(x - np.amax(x, axis = 1, keepdims = True))
    top = top / np.sum(top, axis = 1, keepdims = True)
    loss = np.sum(np.log(top[np.arange(x.shape[0]), y])) / x.shape[0]
    dx = top.copy()
    dx[np.arange(x.shape[0]), y] -= 1
    dx = dx/x.shape[0]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
