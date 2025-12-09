import numpy as np
import jax.numpy as jnp

class YieldModel_RBF:
    def __init__(self, alpha_points, yields, kernel="gaussian", epsilon=1.0, ridge=1e-6):
        """
        alpha_points: (K, D)
        yields:       (K,)
        kernel:       'gaussian', 'linear', 'cubic', 'mq'
        epsilon:      length scale
        ridge:        small regularization added to diagonal to avoid singular matrix
        """
        self.alpha_points = np.asarray(alpha_points, float)
        self.logy = np.log(np.asarray(yields, float) + 1e-12)
        self.epsilon = float(epsilon)
        self.kernel = kernel
        self.ridge = float(ridge)

        K, D = self.alpha_points.shape

        # Build RBF kernel matrix Φ
        diff = self.alpha_points[:, None, :] - self.alpha_points[None, :, :]  # (K, K, D)
        r = np.linalg.norm(diff, axis=-1)                                     # (K, K)
        Phi = self._rbf(r)

        # Regularize diagonal to avoid singularity
        Phi_reg = Phi + ridge * np.eye(K)

        # Solve for weights w in Φ_reg w ≈ logy  (use lstsq for extra robustness)
        self.weights, *_ = np.linalg.lstsq(Phi_reg, self.logy, rcond=None)

    # --------------------------------------------------------------
    def _rbf(self, r):
        if self.kernel == "gaussian":
            return np.exp(-(r / self.epsilon)**2)
        elif self.kernel == "linear":
            return r
        elif self.kernel == "cubic":
            return r**3
        elif self.kernel == "mq":  # multiquadric
            return np.sqrt(1 + (r / self.epsilon)**2)
        else:
            raise ValueError(f"Unknown RBF kernel {self.kernel}")

    # --------------------------------------------------------------
    def logN(self, alpha):
        alpha = np.asarray(alpha, float).reshape(1, -1)  # (1, D)
        diff = alpha - self.alpha_points                 # (K, D)
        r = np.linalg.norm(diff, axis=1)                 # (K,)
        vals = self._rbf(r)
        logN = np.dot(self.weights, vals)                # scalar
        return logN

    def N(self, alpha):
        return np.exp(self.logN(alpha))

    def ratio(self, base_alpha, goal_alpha):
        return np.exp(self.logN(goal_alpha) - self.logN(base_alpha))
    

import numpy as np
import jax.numpy as jnp
from itertools import combinations_with_replacement


class YieldModel_Poly:
    """
    ND polynomial model for log N(alpha).

    Fits a multivariate polynomial in ND alpha:
    log N(alpha) ≈ sum_m c_m * prod_d alpha_d^{p_{m,d}}

    Good for *small* degree (1 or 2) and low-ish dimension.
    """

    def __init__(self, alpha_points, yields, degree=1, normalize=True):
        """
        alpha_points: (K, D)  array of ND alpha locations
        yields:       (K,)    yields/counts at those locations
        degree:       max total polynomial degree (1=linear, 2=quadratic, ...)
        normalize:    if True, standardize alpha before fitting (recommended)
        """
        alpha_points = np.asarray(alpha_points, float)
        yields = np.asarray(yields, float)
        assert alpha_points.ndim == 2, "alpha_points must be (K, D)"
        assert yields.shape[0] == alpha_points.shape[0], "K mismatch"

        self.degree = degree
        self.D = alpha_points.shape[1]
        self.normalize = normalize

        # Optional: standardize alpha for better conditioning
        if normalize:
            self.mean_ = alpha_points.mean(axis=0, keepdims=True)
            self.std_ = alpha_points.std(axis=0, keepdims=True) + 1e-12
            alpha_scaled = (alpha_points - self.mean_) / self.std_
        else:
            self.mean_ = np.zeros((1, self.D))
            self.std_ = np.ones((1, self.D))
            alpha_scaled = alpha_points

        # Build monomial exponents for all terms up to given degree
        self.exponents = self._build_exponents(self.D, degree)

        # Design matrix Phi: (K, n_terms)
        Phi = self._build_design_matrix(alpha_scaled, self.exponents)

        # Fit linear model in log-space: Phi * coeffs = log(y)
        logy = np.log(yields + 1e-12)
        # Use least squares (works fine for small n_terms)
        self.coeffs, *_ = np.linalg.lstsq(Phi, logy, rcond=None)

    # ---------- basis construction ----------

    @staticmethod
    def _build_exponents(D, degree):
        """
        Return list of exponent tuples e = (e1,...,eD) for all monomials
        with total degree <= 'degree'.
        """
        exps = []
        # Use stars-and-bars via combinations_with_replacement
        # for all degrees from 0..degree
        for total_deg in range(degree + 1):
            # distribute 'total_deg' indistinguishable balls into D bins
            # combinations_with_replacement chooses positions of separators
            for comb in combinations_with_replacement(range(D), total_deg):
                e = [0] * D
                for idx in comb:
                    e[idx] += 1
                exps.append(tuple(e))
        return exps  # list of length n_terms, each an (D,) tuple

    @staticmethod
    def _monomials(alpha_scaled, exponents):
        """
        alpha_scaled: (K, D)
        exponents:    list of exponent tuples, len = n_terms
        returns:      (K, n_terms) monomial matrix
        """
        K, D = alpha_scaled.shape
        n_terms = len(exponents)
        Phi = np.ones((K, n_terms), dtype=float)
        for j, e in enumerate(exponents):
            # compute prod_d alpha_d^{e_d}
            # skip j=0 if exponent is all zeros (constant term); Phi[:,0]==1
            if any(p != 0 for p in e):
                term = np.ones(K, dtype=float)
                for d, p in enumerate(e):
                    if p != 0:
                        term *= alpha_scaled[:, d] ** p
                Phi[:, j] = term
        return Phi

    def _build_design_matrix(self, alpha_scaled, exponents):
        return self._monomials(alpha_scaled, exponents)

    # ---------- evaluation ----------

    def _prepare_alpha(self, alpha):
        alpha = np.asarray(alpha, float)
        if alpha.ndim == 1:
            alpha = alpha[None, :]  # (1, D)
        assert alpha.shape[1] == self.D, "Wrong alpha dimension"
        if self.normalize:
            alpha = (alpha - self.mean_) / self.std_
        return alpha

    def logN(self, alpha):
        """
        alpha: shape (D,) or (N, D)
        returns: log N(alpha), scalar or (N,)
        """
        alpha_scaled = self._prepare_alpha(alpha)          # (N, D)
        Phi = self._build_design_matrix(alpha_scaled, self.exponents)  # (N, n_terms)
        logN = Phi @ self.coeffs                           # (N,)
        return logN if logN.shape[0] > 1 else logN[0]

    def N(self, alpha):
        return np.exp(self.logN(alpha))

    def ratio(self, base_alpha, goal_alpha):
        """
        N(goal_alpha) / N(base_alpha)
        """
        logN_base = self.logN(base_alpha)
        logN_goal = self.logN(goal_alpha)
        return np.exp(logN_goal - logN_base)