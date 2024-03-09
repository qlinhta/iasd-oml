###############################################################################
#            THIS CODE HAS BEEN TAKEN FROM LAB 4 OF THE COURSE                 #
#            "OPTIMIZATION FOR MACHINE LEARNING" IASD                          #
###############################################################################

import numpy as np
from numpy.linalg import norm
from scipy.linalg import svdvals
from scipy.linalg import toeplitz
from numpy.random import multivariate_normal, randn


class RegPb(object):
    def __init__(self, X, y, lbda=0, loss='l2'):
        self.X = X
        self.y = y
        self.n, self.d = X.shape
        self.loss = loss
        self.lbda = lbda

    # Objective value
    def fun(self, w):
        if self.loss == 'l2':
            return norm(self.X.dot(w) - self.y) ** 2 / (2. * self.n) + self.lbda * norm(w) ** 2 / 2.
        elif self.loss == 'logit':
            yXw = self.y * self.X.dot(w)
            return np.mean(np.log(1. + np.exp(-yXw))) + self.lbda * norm(w) ** 2 / 2.

    # Partial objective value
    def f_i(self, i, w):
        if self.loss == 'l2':
            return norm(self.X[i].dot(w) - self.y[i]) ** 2 / (2.) + self.lbda * norm(w) ** 2 / 2.
        elif self.loss == 'logit':
            yXwi = self.y[i] * np.dot(self.X[i], w)
            return np.log(1. + np.exp(- yXwi)) + self.lbda * norm(w) ** 2 / 2.

    # Full gradient computation
    def grad(self, w):
        if self.loss == 'l2':
            return self.X.T.dot(self.X.dot(w) - self.y) / self.n + self.lbda * w
        elif self.loss == 'logit':
            yXw = self.y * self.X.dot(w)
            aux = 1. / (1. + np.exp(yXw))
            return - (self.X.T).dot(self.y * aux) / self.n + self.lbda * w

    # Hessian (Quyen Linh TA)
    def hess(self, w):
        if self.loss == 'l2':
            return self.X.T.dot(self.X) / self.n + self.lbda * np.eye(self.d)
        elif self.loss == 'logit':
            yXw = self.y * self.X.dot(w)
            aux = 1. / (1. + np.exp(yXw))
            D = np.diag(aux * (1. - aux))
            return self.X.T.dot(D).dot(self.X) / self.n + self.lbda * np.eye(self.d)

    # Partial gradient
    def grad_i(self, i, w):
        x_i = self.X[i]
        if self.loss == 'l2':
            return (x_i.dot(w) - self.y[i]) * x_i + self.lbda * w
        elif self.loss == 'logit':
            grad = - x_i * self.y[i] / (1. + np.exp(self.y[i] * x_i.dot(w)))
            grad += self.lbda * w
            return grad

    # Lipschitz constant for the gradient
    def lipgrad(self):
        if self.loss == 'l2':
            L = norm(self.X, ord=2) ** 2 / self.n + self.lbda
        elif self.loss == 'logit':
            L = norm(self.X, ord=2) ** 2 / (4. * self.n) + self.lbda
        return L

    # ''Strong'' convexity constant (could be zero if self.lbda=0)
    def cvxval(self):
        if self.loss == 'l2':
            s = svdvals(self.X)
            mu = min(s) ** 2 / self.n
            return mu + self.lbda
        elif self.loss == 'logit':
            return self.lbda


def simu_linmodel(w, n, std=1., corr=0.5):
    d = w.shape[0]
    cov = toeplitz(corr ** np.arange(0, d))
    X = multivariate_normal(np.zeros(d), cov, size=n)
    noise = std * randn(n)
    y = X.dot(w) + noise
    return X, y

###############################################################################
#            THIS CODE HAS BEEN TAKEN FROM LAB 4 OF THE COURSE                 #
#            "OPTIMIZATION FOR MACHINE LEARNING" IASD                          #
###############################################################################