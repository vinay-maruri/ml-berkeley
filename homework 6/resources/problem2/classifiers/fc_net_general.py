from builtins import range
from builtins import object
import numpy as np

from layers import *


class FullyConnectedNetGeneral(object):
    """
    A fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of [H, ...], and perform classification over C classes.

    The architecure should be like affine - relu - affine - softmax for a one
    hidden layer network, and affine - relu - affine - relu- affine - softmax for
    a two hidden layer network, etc.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim, hidden_dim=[10, 5], num_classes=8,
                 weight_scale=0.1):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: A list of integer giving the sizes of the hidden layers
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        """
        self.params = {}
        self.hidden_dim = hidden_dim
        ############################################################################
        # TODO: Initialize the weights and biases of the net. Weights              #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        weights_1 = np.random.normal(loc = 0.0, scale = weight_scale, size = (input_dim, hidden_dim[0]))
        weights_2 = np.random.normal(loc = 0.0, scale = weight_scale, size = (hidden_dim[0], hidden_dim[0]))
        weights_3 = np.random.normal(loc = 0.0, scale = weight_scale, size = (hidden_dim[0], hidden_dim[0]))
        weights_4 = np.random.normal(loc = 0.0, scale = weight_scale, size = (hidden_dim[0], hidden_dim[0]))
        biases_1 = np.zeros(shape = hidden_dim[0])
        biases_2 = np.zeros(shape = hidden_dim[0])
        biases_3 = np.zeros(shape = hidden_dim[0])
        biases_4 = np.zeros(shape = hidden_dim[0])
        self.params['W1'] = weights_1
        self.params['W2'] = weights_2
        self.params['W3'] = weights_3
        self.params['W4'] = weights_4
        self.params['b1'] = biases_1
        self.params['b2'] = biases_2
        self.params['b3'] = biases_3
        self.params['b4'] = biases_4
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the net, computing the              #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # architecture: input - fc layer - ReLU activation - fc layer - softmax loss.
        out1, cache1 = affine_forward(X, self.params['W1'], self.params['b1'])
        rout1, rcache1 = relu_forward(out1)
        out2, cache2 = affine_forward(rout1, self.params['W2'], self.params['b2'])
        rout2, rcache2 = relu_forward(out2)
        out3, cache3 = affine_forward(rout2, self.params['W3'], self.params['b3'])
        rout3, rcache3 = relu_forward(out3)
        out4, cache4 = affine_forward(rout3, self.params['W4'], self.params['b4'])
        scores = out4
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the net. Store the loss            #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k].                                                          #
        ############################################################################
        #passed all the way forward, now we need to traverse in reverse i.e. (second
        #fc layer first).
        soft_loss, upstreamd = softmax_loss(out4, y)
        dx4, dw4, db4 = affine_backward(upstreamd, cache4)
        rback1 = relu_backward(dx4, rcache3)
        dx3, dw3, db3 = affine_backward(rback1, cache3)
        rback2 = relu_backward(dx3, rcache2)
        dx2, dw2, db2 = affine_backward(rback2, cache2)
        rback3 = relu_backward(dx2, rcache1)
        dx1, dw1, db1 = affine_backward(rback3, cache1)
        grads['W1'] = dw1
        grads['W2'] = dw2
        grads['W3'] = dw3
        grads['W4'] = dw4
        grads['b1'] = db1
        grads['b2'] = db2
        grads['b3'] = db3
        grads['b4'] = db4
        loss += (np.sum(self.params['W1']*self.params['W1']) + np.sum(self.params['W2']*self.params['W2']) + np.sum(self.params['W3']*self.params['W3']) + np.sum(self.params['W4']*self.params['W4']))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
