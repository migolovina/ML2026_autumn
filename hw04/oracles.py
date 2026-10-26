
import numpy as np
import scipy
import scipy.sparse
from scipy.special import expit

class BaseSmoothOracle(object):
    def func(self, x): raise NotImplementedError
    def grad(self, x): raise NotImplementedError
    def hess(self, x): raise NotImplementedError
    def func_directional(self, x, d, alpha): return np.squeeze(self.func(x + alpha * d))
    def grad_directional(self, x, d, alpha): return np.squeeze(self.grad(x + alpha * d).dot(d))

class QuadraticOracle(BaseSmoothOracle):
    def __init__(self, A, b):
        if not scipy.sparse.issparse(A) and not np.allclose(A, A.T): raise ValueError
        self.A = A
        self.b = b
    def func(self, x): return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)
    def grad(self, x): return self.A.dot(x) - self.b
    def hess(self, x): return self.A 

class LogRegL2Oracle(BaseSmoothOracle):
    def __init__(self, A, b, regcoef):
        self.A = A
        self.b = b
        self.regcoef = regcoef
        self._m = A.shape[0]
    def func(self, x):
        z = self.b * self.A.dot(x)
        return np.sum(np.logaddexp(0.0, -z)) / self._m + 0.5 * self.regcoef * np.dot(x, x)
    def grad(self, x):
        z = self.b * self.A.dot(x)
        sigma_neg_z = expit(-z)
        return -self.A.T.dot(self.b * sigma_neg_z) / self._m + self.regcoef * x
    def hess(self, x):
        z = self.b * self.A.dot(x)
        sigma_neg_z = expit(-z)
        s = sigma_neg_z * (1.0 - sigma_neg_z)
        if scipy.sparse.issparse(self.A):
            weighted_A = self.A.multiply(s[:, None])
        else:
            weighted_A = self.A * s[:, None]
        H = self.A.T.dot(weighted_A) / self._m
        return H + self.regcoef * np.eye(x.shape[0])

def grad_finite_diff(func, x, eps=1e-8):
    x = np.asarray(x, dtype=float)
    n = x.size
    g = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n); e_i[i] = eps
        g[i] = (func(x + e_i) - func(x)) / eps
    return g

def hess_finite_diff(func, x, eps=1e-5):
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n))
    f_x = func(x)
    f_xi = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n); e_i[i] = eps
        f_xi[i] = func(x + e_i)
    for i in range(n):
        for j in range(i, n):
            e_i = np.zeros(n); e_i[i] = eps
            e_j = np.zeros(n); e_j[j] = eps
            val = (func(x + e_i + e_j) - f_xi[i] - f_xi[j] + f_x) / (eps * eps)
            H[i, j] = val; H[j, i] = val
    return H
