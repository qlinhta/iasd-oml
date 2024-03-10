import numpy as np
from tqdm import tqdm


class Optimizer:
    def __init__(self, objective_function, gradient_function, hessian_function, initial_point, verbose=True):
        self.objective_function = objective_function
        self.gradient_function = gradient_function
        self.hessian_function = hessian_function
        self.current_point = np.array(initial_point, dtype=np.float64)
        self.verbose = verbose
        self.stats = {
            'iterations': 0,
            'function_evaluations': 0,
            'gradient_evaluations': 0,
            'hessian_evaluations': 0,
            'function_values': [],
            'gradient_norms': [],
            'step_sizes': [],
            'line_search_step': 0,
            'trajectory': [initial_point],
            'error_messages': []
        }
        self.cache = {
            'function_values': {},
            'gradients': {},
            'hessians': {}
        }

    def solve(self, tolerance=1e-4, max_iterations=1000):
        if self.verbose:
            print("Starting optimization!")
        for iteration in range(max_iterations):
            try:
                if self.verbose:
                    print(f"Iteration {iteration + 1}:")
                step_size = self._step(tolerance)
                if self._has_converged(tolerance):
                    if self.verbose:
                        print("Convergence achieved!")
                    break
            except np.linalg.LinAlgError as error:
                self.stats['error_messages'].append(str(error))
                if self.verbose:
                    print(f"Error: {str(error)} - Exiting due to singular Hessian.")
                break
        return self.current_point, self.stats

    def _step(self, tolerance):
        raise NotImplementedError("Subclass must implement abstract method")

    def evaluate_objective_function(self, point):
        point_tuple = tuple(point)
        if point_tuple not in self.cache['function_values']:
            self.stats['function_evaluations'] += 1
            self.cache['function_values'][point_tuple] = self.objective_function(point)
        return self.cache['function_values'][point_tuple]

    def evaluate_gradient(self, point):
        point_tuple = tuple(point)
        if point_tuple not in self.cache['gradients']:
            self.stats['gradient_evaluations'] += 1
            self.cache['gradients'][point_tuple] = self.gradient_function(point)
        return self.cache['gradients'][point_tuple]

    def evaluate_hessian(self, point):
        point_tuple = tuple(point)
        if self.hessian_function is not None and point_tuple not in self.cache['hessians']:
            self.stats['hessian_evaluations'] += 1
            self.cache['hessians'][point_tuple] = self.hessian_function(point)
        return self.cache['hessians'].get(point_tuple, None)

    def _has_converged(self, tolerance):
        gradient = self.evaluate_gradient(self.current_point)
        gradient_norm = np.linalg.norm(gradient)
        self.stats['gradient_norms'].append(gradient_norm)
        function_value = self.evaluate_objective_function(self.current_point)
        self.stats['function_values'].append(function_value)
        if self.verbose:
            print(f"\tCurrent point: {self.current_point}")
            print(f"\tFunction value: {function_value}")
            print(f"\tGradient norm: {gradient_norm}")
        return gradient_norm < tolerance


class NewtonsMethodOptimizer(Optimizer):
    def __init__(self, objective_function, gradient_function, hessian_function, initial_point, verbose=True):
        super().__init__(objective_function, gradient_function, hessian_function, initial_point, verbose)

    def _step(self, tolerance):
        gradient = self.evaluate_gradient(self.current_point)
        hessian = self.evaluate_hessian(self.current_point)
        delta_point = np.linalg.solve(hessian, -gradient)
        self.current_point += delta_point
        self.stats['iterations'] += 1
        self.stats['step_sizes'].append(np.linalg.norm(delta_point))
        self.stats['trajectory'].append(self.current_point.tolist())


class GlobalizedNewtonsOptimizer(NewtonsMethodOptimizer):
    def __init__(self, objective_function, gradient_function, hessian_function, initial_point, c=0.0001, theta=0.5,
                 verbose=True):
        super().__init__(objective_function, gradient_function, hessian_function, initial_point, verbose)
        self.c = c
        self.theta = theta

    def _compute_lambda_k(self, hessian):
        lambda_min = np.min(np.linalg.eigvals(hessian))
        return 2 * max(-lambda_min, 1e-10)

    def _line_search(self, search_direction):
        alpha = 1
        while True:
            new_point = self.current_point + alpha * search_direction
            self.stats['line_search_step'] += 1
            if self.evaluate_objective_function(new_point) < self.evaluate_objective_function(
                    self.current_point) + self.c * alpha * np.dot(search_direction,
                                                                  self.evaluate_gradient(self.current_point)):
                break
            alpha *= self.theta
        return alpha

    def _step(self, tolerance):
        current_gradient = self.evaluate_gradient(self.current_point)
        hessian = self.evaluate_hessian(self.current_point)
        lambda_k = self._compute_lambda_k(hessian)
        adjusted_hessian = hessian + lambda_k * np.eye(len(current_gradient))
        search_direction = -np.linalg.solve(adjusted_hessian, current_gradient)
        step_size = self._line_search(search_direction)
        self.current_point += step_size * search_direction
        self.stats['iterations'] += 1
        self.stats['trajectory'].append(self.current_point.tolist())


class QuasiNewtonBFGSOptimizer(Optimizer):
    def __init__(self, objective_function, gradient_function, initial_point, c=0.0001, theta=0.5, verbose=True):
        super().__init__(objective_function, gradient_function, None, initial_point, verbose)
        self.c = c
        self.theta = theta
        self.inverse_hessian_approximation = np.eye(len(initial_point))

    def _update_inverse_hessian_approximation(self, sk, yk):
        rho = 1.0 / np.dot(yk, sk)
        identity_matrix = np.eye(len(sk))
        self.inverse_hessian_approximation = (identity_matrix - rho * np.outer(sk,
                                                                               yk)) @ self.inverse_hessian_approximation @ (
                                                     identity_matrix - rho * np.outer(yk, sk)) + rho * np.outer(sk,
                                                                                                                sk)

    def _line_search(self, search_direction):
        alpha = 1
        while True:
            potential_new_point = self.current_point + alpha * search_direction
            self.stats['line_search_step'] += 1
            if self.evaluate_objective_function(potential_new_point) < self.evaluate_objective_function(
                    self.current_point) + self.c * alpha * np.dot(search_direction,
                                                                  self.evaluate_gradient(self.current_point)):
                break
            alpha *= self.theta
        return alpha

    def _step(self, tolerance):
        current_gradient = self.evaluate_gradient(self.current_point)
        search_direction = -np.dot(self.inverse_hessian_approximation, current_gradient)
        step_size = self._line_search(search_direction)
        sk = step_size * search_direction
        self.current_point += sk
        new_gradient = self.evaluate_gradient(self.current_point)
        yk = new_gradient - current_gradient
        if np.dot(sk, yk) > 0:
            self._update_inverse_hessian_approximation(sk, yk)
        self.stats['iterations'] += 1
        self.stats['trajectory'].append(self.current_point.tolist())


class LBFGSOptimizer(Optimizer):
    def __init__(self, objective_function, gradient_function, initial_point, memory_size=5, c=0.0001, theta=0.5,
                 verbose=True):
        super().__init__(objective_function, gradient_function, None, initial_point, verbose)
        self.memory_size = memory_size
        self.c = c
        self.theta = theta
        self.sk_memory = []
        self.yk_memory = []

    def _update_memory(self, sk, yk):
        if self.memory_size > 0:
            if len(self.sk_memory) >= self.memory_size:
                self.sk_memory.pop(0)
                self.yk_memory.pop(0)
            self.sk_memory.append(sk)
            self.yk_memory.append(yk)

    def _compute_search_direction(self, current_gradient):
        if self.memory_size == 0:
            return -current_gradient
        else:
            q = current_gradient
            alphas = []
            rhos = []

            for i in reversed(range(len(self.sk_memory))):
                sk = self.sk_memory[i]
                yk = self.yk_memory[i]
                rho = 1.0 / np.dot(yk, sk)
                rhos.insert(0, rho)
                alpha = rho * np.dot(sk, q)
                alphas.insert(0, alpha)
                q = q - alpha * yk

            z = q

            for i in range(len(self.sk_memory)):
                sk = self.sk_memory[i]
                yk = self.yk_memory[i]
                rho = rhos[i]
                beta = rho * np.dot(yk, z)
                z = z + sk * (alphas[i] - beta)

            return -z

    def _line_search(self, search_direction):
        alpha = 1
        while True:
            potential_new_point = self.current_point + alpha * search_direction
            self.stats['line_search_step'] += 1
            if self.evaluate_objective_function(potential_new_point) < self.evaluate_objective_function(
                    self.current_point) + self.c * alpha * np.dot(search_direction,
                                                                  self.evaluate_gradient(self.current_point)):
                break
            alpha *= self.theta
        return alpha

    def _step(self, tolerance):
        current_gradient = self.evaluate_gradient(self.current_point)
        search_direction = self._compute_search_direction(current_gradient)
        step_size = self._line_search(search_direction)
        sk = step_size * search_direction
        self.current_point += sk
        new_gradient = self.evaluate_gradient(self.current_point)
        yk = new_gradient - current_gradient
        if self.memory_size > 0 and np.dot(sk, yk) > 0:
            self._update_memory(sk, yk)
        self.stats['iterations'] += 1
        self.stats['trajectory'].append(self.current_point.tolist())


class BaseOptimizer:
    def __init__(self, problem, theta=0.5, c=1e-4, alpha=0.01, verbose=False):
        self.problem = problem
        self.theta = theta
        self.c = c
        self.alpha = alpha
        self.verbose = verbose
        self.stats = {'losses': [], 'epochs': []}

    def armijo_line_search(self, w, d, grad_f_w, sample_indices=None):
        alpha = self.alpha
        f_w = self.problem.fun(w) if sample_indices is None else np.mean(
            [self.problem.f_i(i, w) for i in sample_indices])
        for _ in range(100):
            w_new = w + alpha * d
            f_w_new = self.problem.fun(w_new) if sample_indices is None else np.mean(
                [self.problem.f_i(i, w_new) for i in sample_indices])
            condition = f_w + self.c * alpha * np.dot(grad_f_w.T, d)
            if f_w_new <= condition:
                break
            alpha *= self.theta
        return alpha

    def optimize(self, w_init, tol=1e-6):
        raise NotImplementedError("This method should be implemented by subclasses.")


class SubsamplingNewton(BaseOptimizer):
    def __init__(self, problem, sample_size_grad, sample_size_hess, epochs, theta=0.9, c=1e-4, alpha=0.001,
                 lambda_reg=1. / 1000 ** (0.5), verbose=False):
        super().__init__(problem, theta, c, alpha, verbose)
        self.sample_size_grad = sample_size_grad
        self.sample_size_hess = sample_size_hess
        self.epochs = epochs
        self.lambda_reg = lambda_reg
        self.verbose = verbose

    def hess_i(self, i, w):
        if hasattr(self.problem, 'hess_i'):
            return self.problem.hess_i(i, w)
        else:
            X_i = self.problem.X[i]
            n_features = X_i.shape[0]
            H_i = 2 * np.outer(X_i, X_i) + 2 * self.lambda_reg * np.eye(n_features)
            return H_i

    def optimize(self, w_init, tol=1e-6, use_line_search=True):
        self.stats = {'losses': [], 'epochs': []}
        w = w_init.copy()
        n_samples = self.problem.n
        batch_per_epoch = max(n_samples // min(self.sample_size_grad, self.sample_size_hess), 1)

        for epoch in range(self.epochs):
            with tqdm(total=batch_per_epoch, desc=f"Epoch {epoch + 1}", disable=not self.verbose) as epoch_progress:
                for batch in range(batch_per_epoch):
                    grad_indices = np.random.choice(n_samples, self.sample_size_grad, replace=False)
                    hess_indices = np.random.choice(n_samples, self.sample_size_hess, replace=False)

                    grad = np.mean([self.problem.grad_i(i, w) for i in grad_indices], axis=0)
                    H = np.mean([self.hess_i(i, w) for i in hess_indices], axis=0)

                    d = -np.linalg.solve(H, grad)

                    if use_line_search:
                        alpha = self.armijo_line_search(w, d, grad, grad_indices)
                    else:
                        alpha = self.alpha

                    w += alpha * d
                    epoch_progress.update(1)

                current_loss = self.problem.fun(w)
                self.stats['losses'].append(current_loss)
                self.stats['epochs'].append(epoch + 1)
                epoch_progress.set_postfix({'Loss': f'{current_loss:.4f}'})

        return w, self.stats


class StochasticBFGS(BaseOptimizer):
    def __init__(self, problem, sample_size, epochs, theta=0.5, c=1e-4, alpha=0.01, verbose=False):
        super().__init__(problem, theta, c, alpha, verbose)
        self.sample_size = sample_size
        self.epochs = epochs

    def optimize(self, w_init, tol=1e-6, use_line_search=True):
        self.stats = {'losses': [], 'epochs': []}
        w = w_init.copy()
        I = np.eye(len(w))
        H = I

        n_samples = self.problem.n
        batch_per_epoch = max(n_samples // self.sample_size, 1)

        for epoch in range(self.epochs):
            with tqdm(total=batch_per_epoch, desc=f"Epoch {epoch + 1}", disable=not self.verbose) as epoch_progress:
                for _ in range(batch_per_epoch):
                    sample_indices = np.random.choice(n_samples, self.sample_size, replace=False)
                    grad = np.mean([self.problem.grad_i(i, w) for i in sample_indices], axis=0)
                    w_prev = w.copy()
                    grad_prev = grad.copy()
                    d = -H.dot(grad)

                    if use_line_search:
                        alpha = self.armijo_line_search(w, d, grad, sample_indices)
                    else:
                        alpha = self.alpha

                    w += alpha * d
                    s = alpha * d
                    grad = np.mean([self.problem.grad_i(i, w) for i in sample_indices], axis=0)
                    y = grad - grad_prev
                    rho = 1.0 / np.dot(y.T, s)

                    if np.dot(y.T, s) > 0:
                        H = (I - rho * np.outer(s, y)).dot(H).dot(I - rho * np.outer(y, s)) + rho * np.outer(s, s)

                    epoch_progress.update(1)

                current_loss = np.mean([self.problem.f_i(i, w) for i in sample_indices])
                epoch_progress.set_postfix({'Loss': f'{current_loss:.4f}'})
                self.stats['losses'].append(current_loss)
                self.stats['epochs'].append(epoch + 1)
        return w, self.stats


class StochasticLBFGS(BaseOptimizer):
    def __init__(self, problem, sample_size, epochs, m=5, theta=0.5, c=1e-4, alpha=0.01, verbose=False):
        super().__init__(problem, theta, c, alpha, verbose)
        self.sample_size = sample_size
        self.epochs = epochs
        self.m = m

    def optimize(self, w_init, tol=1e-6, use_line_search=True):
        self.stats = {'losses': [], 'epochs': []}
        w = w_init.copy()
        n = len(w)
        if self.m > 0:
            S = np.zeros((n, self.m))
            Y = np.zeros((n, self.m))
            rho = np.zeros(self.m)
        alpha_i = np.zeros(max(self.m, 1))
        q = np.zeros(n)

        n_samples = self.problem.n
        batch_per_epoch = max(n_samples // self.sample_size, 1)
        k = 0

        for epoch in range(self.epochs):
            with tqdm(total=batch_per_epoch, desc=f"Epoch {epoch + 1}", disable=not self.verbose) as epoch_progress:
                for _ in range(batch_per_epoch):
                    sample_indices = np.random.choice(n_samples, self.sample_size, replace=False)
                    grad = np.mean([self.problem.grad_i(i, w) for i in sample_indices], axis=0)

                    if self.m > 0:
                        q[:] = grad
                        for i in range(min(k, self.m)):
                            idx = (k - i - 1) % self.m
                            alpha_i[idx] = rho[idx] * S[:, idx].dot(q)
                            q -= alpha_i[idx] * Y[:, idx]

                        if k > 0:
                            gamma = S[:, (k - 1) % self.m].dot(Y[:, (k - 1) % self.m]) / Y[:, (k - 1) % self.m].dot(
                                Y[:, (k - 1) % self.m])
                            H0 = gamma * np.eye(n)
                        else:
                            H0 = np.eye(n)

                        r = H0.dot(q)
                        for i in range(min(k, self.m)):
                            idx = (k - i - 1) % self.m
                            beta = rho[idx] * Y[:, idx].dot(r)
                            r += S[:, idx] * (alpha_i[idx] - beta)

                        d = -r
                    else:
                        d = -grad

                    step_size = self.alpha if not use_line_search else self.armijo_line_search(w, d, grad,
                                                                                               sample_indices)
                    s = step_size * d
                    w += s

                    if self.m > 0:
                        new_grad = np.mean([self.problem.grad_i(i, w) for i in sample_indices], axis=0)
                        y = new_grad - grad
                        if np.dot(y, s) > 0:
                            rho[k % self.m] = 1.0 / np.dot(y, s)
                            S[:, k % self.m] = s
                            Y[:, k % self.m] = y
                            k += 1

                    epoch_progress.update(1)

                current_loss = np.mean([self.problem.f_i(i, w) for i in sample_indices])
                epoch_progress.set_postfix({'Loss': f'{current_loss:.4f}'})
                self.stats['losses'].append(current_loss)
                self.stats['epochs'].append(epoch + 1)
                epoch_progress.close()

        return w, self.stats
