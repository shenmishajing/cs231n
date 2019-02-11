from random import shuffle

import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in range(X.shape[0]):
        score = X[i].dot(W)
        expscore = np.exp(score)
        p = expscore / np.sum(expscore)
        loss -= np.log(p[y[i]])
        for j in range(W.shape[1]):
            if (j == y[i]):
                dW[:, j] += (p[y[i]] - 1) * X[i]
            else:
                dW[:, j] += p[j] * X[i]
    loss /= X.shape[0]
    dW /= X.shape[0]

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = np.exp(X.dot(W))
    p = scores / scores.sum(axis=1).reshape(-1, 1)
    loss = np.sum(-np.log(p[range(X.shape[0]), y])) / X.shape[0] + np.sum(
        reg * W * W)

    p[range(X.shape[0]), y] -= 1
    dW = X.T.dot(p) / X.shape[0] + 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
