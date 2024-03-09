from sklearn.datasets import load_svmlight_file
import numpy as np

class Instance:
    def __init__(self, dataset_path=None, X=None, y=None):
        if dataset_path is not None:
            self.X, self.y = self.load_dataset(dataset_path)
        elif X is not None and y is not None:
            self.X = X
            self.y = y
        else:
            raise ValueError("Must provide either dataset_path or both X and y.")

        self.n = self.X.shape[0]
        self.d = self.X.shape[1]
        self.lbda = 1. / self.n ** (0.5)

    def load_dataset(self, dataset_path):
        features_sparse, labels = load_svmlight_file(dataset_path)
        features_dense = features_sparse.toarray()
        return features_dense, labels

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def single_loss(self, weights, index):
        feature_vector = self.X[index]
        label = self.y[index]
        prediction = self.sigmoid(np.dot(feature_vector, weights))
        return (label - prediction) ** 2

    def single_gradient(self, weights, index):
        feature_vector = self.X[index]
        label = self.y[index]
        exp_weight_feature_dot = np.exp(np.dot(feature_vector, weights))
        gradient_factor = -2 * exp_weight_feature_dot * ((exp_weight_feature_dot * (label - 1)) + label)
        gradient_denominator = (1 + exp_weight_feature_dot) ** 3
        return (gradient_factor / gradient_denominator) * feature_vector

    def single_hessian(self, weights, index):
        feature_vector = self.X[index]
        label = self.y[index]
        exp_weight_feature_dot = np.exp(np.dot(feature_vector, weights))
        hessian_numerator = 2 * exp_weight_feature_dot * (
                exp_weight_feature_dot ** 2 * (label - 1) + 2 * exp_weight_feature_dot - label)
        hessian_denominator = (1 + exp_weight_feature_dot) ** 4
        unregularized_hessian = (hessian_numerator / hessian_denominator) * np.outer(feature_vector, feature_vector)
        regularization_term = 2 * self.lbda * np.eye(self.d)
        return unregularized_hessian + regularization_term

    def grad_i(self, i, w):
        return self.single_gradient(w, i)

    def f_i(self, i, w):
        return self.single_loss(w, i)

    def fun(self, w):
        return np.mean([self.f_i(i, w) for i in range(self.n)])

    def grad(self, w):
        return np.mean([self.grad_i(i, w) for i in range(self.n)], axis=0)

    def hess_i(self, i, w):
        return self.single_hessian(w, i)

    def hess(self, w):
        return np.mean([self.hess_i(i, w) for i in range(self.n)], axis=0)