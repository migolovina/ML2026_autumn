import numpy as np
from numpy.linalg import LinAlgError
import scipy
import scipy.optimize
import scipy.linalg
from datetime import datetime
from collections import defaultdict


class LineSearchTool(object):
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        if self._method == 'Constant':
            return self.c

        phi0 = oracle.func_directional(x_k, d_k, 0.0)
        dphi0 = oracle.grad_directional(x_k, d_k, 0.0)

        if self._method == 'Armijo':
            alpha = self.alpha_0 if previous_alpha is None else previous_alpha
            while True:
                phi = oracle.func_directional(x_k, d_k, alpha)
                if not np.isfinite(phi) or phi <= phi0 + self.c1 * alpha * dphi0:
                    return alpha
                alpha *= 0.5

        elif self._method == 'Wolfe':
            res = scipy.optimize.line_search(
                oracle.func, oracle.grad, x_k, d_k,
                c1=self.c1, c2=self.c2
            )
            alpha = res.alpha if hasattr(res, 'alpha') else res[0]
            
            if alpha is None:
                alpha = self.alpha_0
                while True:
                    phi = oracle.func_directional(x_k, d_k, alpha)
                    if not np.isfinite(phi) or phi <= phi0 + self.c1 * alpha * dphi0:
                        return alpha
                    alpha *= 0.5
            return alpha
        return None


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    if line_search_options is None:
        line_search_options = {}
    line_search_tool = LineSearchTool(**line_search_options)
    x_k = np.copy(x_0)

    t_start = datetime.now()
    def _time():
        return (datetime.now() - t_start).total_seconds()

    def _record(x, f, g):
        if trace:
            history['time'].append(_time())
            history['func'].append(f)
            history['grad_norm'].append(np.linalg.norm(g))
            if x.size <= 2:
                history['x'].append(np.copy(x))

    try:
        g0 = oracle.grad(x_k)
        f_k = oracle.func(x_k)
    except Exception:
        return x_k, 'computational_error', history

    if g0 is None or not np.all(np.isfinite(g0)) or not np.isfinite(f_k):
        return x_k, 'computational_error', history

    g0_norm_sq = np.dot(g0, g0)
    if trace:
        _record(x_k, f_k, g0)

    alpha_prev = None
    for it in range(max_iter):
        g_k = oracle.grad(x_k)
        if g_k is None or not np.all(np.isfinite(g_k)):
            return x_k, 'computational_error', history
        grad_norm_sq = np.dot(g_k, g_k)
        
        if grad_norm_sq <= tolerance * g0_norm_sq:
            if display:
                print(f"iter {it}: f={f_k:.6f}, ||g||={np.sqrt(grad_norm_sq):.6e} (success)")
            return x_k, 'success', history

        d_k = -g_k
        alpha0 = 2.0 * alpha_prev if alpha_prev is not None else None
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=alpha0)
        
        if alpha is None:
            return x_k, 'computational_error', history

        x_new = x_k + alpha * d_k
        f_new = oracle.func(x_new)
        g_new = oracle.grad(x_new)
        
        if (f_new is None or g_new is None
                or not np.isfinite(f_new)
                or not np.all(np.isfinite(g_new))):
            return x_k, 'computational_error', history

        x_k = x_new
        alpha_prev = alpha
        f_k = f_new
        g_k = g_new
        
        if trace:
            _record(x_k, f_new, g_new)
        if display:
            print(f"iter {it}: f={f_new:.6f}, ||g||={np.linalg.norm(g_new):.6e}, alpha={alpha:.3e}")

        # Проверяем критерий в конце итерации (на случай, если max_iter=1 и мы уже сошлись)
        if np.dot(g_k, g_k) <= tolerance * g0_norm_sq:
            if display:
                print(f"iter {it+1}: f={f_k:.6f}, ||g||={np.linalg.norm(g_k):.6e} (success)")
            return x_k, 'success', history

    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    if line_search_options is None:
        line_search_options = {}
    line_search_tool = LineSearchTool(**line_search_options)
    x_k = np.copy(x_0)

    t_start = datetime.now()
    def _time():
        return (datetime.now() - t_start).total_seconds()

    def _record(x, f, g):
        if trace:
            history['time'].append(_time())
            history['func'].append(f)
            history['grad_norm'].append(np.linalg.norm(g))
            if x.size <= 2:
                history['x'].append(np.copy(x))

    try:
        g0 = oracle.grad(x_k)
        f_k = oracle.func(x_k)
    except Exception:
        return x_k, 'computational_error', history

    if g0 is None or not np.all(np.isfinite(g0)) or not np.isfinite(f_k):
        return x_k, 'computational_error', history

    g0_norm_sq = np.dot(g0, g0)
    if trace:
        _record(x_k, f_k, g0)

    for it in range(max_iter):
        g_k = oracle.grad(x_k)
        if g_k is None or not np.all(np.isfinite(g_k)):
            return x_k, 'computational_error', history
        grad_norm_sq = np.dot(g_k, g_k)
        
        if grad_norm_sq <= tolerance * g0_norm_sq:
            if display:
                print(f"iter {it}: f={f_k:.6f}, ||g||={np.sqrt(grad_norm_sq):.6e} (success)")
            return x_k, 'success', history

        H = oracle.hess(x_k)
        try:
            H_dense = np.atleast_2d(np.asarray(H, dtype=float))
            H_dense = 0.5 * (H_dense + H_dense.T)
            
            c = scipy.linalg.cho_factor(H_dense, lower=False, check_finite=True)
            d_k = -scipy.linalg.cho_solve(c, g_k)
        except (LinAlgError, ValueError):
            return x_k, 'newton_direction_error', history
            
        if d_k is None or not np.all(np.isfinite(d_k)):
            return x_k, 'newton_direction_error', history

        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)
        if alpha is None:
            return x_k, 'computational_error', history

        x_new = x_k + alpha * d_k
        f_new = oracle.func(x_new)
        g_new = oracle.grad(x_new)
        
        if (f_new is None or g_new is None
                or not np.isfinite(f_new)
                or not np.all(np.isfinite(g_new))):
            return x_k, 'computational_error', history

        x_k = x_new
        f_k = f_new
        g_k = g_new
        if trace:
            _record(x_k, f_new, g_new)
        if display:
            print(f"iter {it}: f={f_new:.6f}, ||g||={np.linalg.norm(g_new):.6e}, alpha={alpha:.3e}")

        # Проверяем критерий в конце итерации
        if np.dot(g_k, g_k) <= tolerance * g0_norm_sq:
            if display:
                print(f"iter {it+1}: f={f_k:.6f}, ||g||={np.linalg.norm(g_k):.6e} (success)")
            return x_k, 'success', history

    return x_k, 'iterations_exceeded', history
