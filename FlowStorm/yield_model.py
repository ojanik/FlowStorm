import numpy as np
from scipy.spatial.distance import cdist, pdist
from itertools import combinations_with_replacement


class YieldModel_RBF:
    """
    RBF model for log-density / relative yield over alpha-space.

    With uniform alpha proposal (SnowStorm), q(alpha) ∝ efficiency(alpha),
    so ratios q(goal)/q(base) give yield ratios (up to a constant that cancels).
    """

    def __init__(
        self,
        alpha_points,
        log_targets,
        kernel="gaussian",
        epsilon=None,
        ridge=1e-6,
        normalize_alpha=True,
    ):
        """
        alpha_points: (K, D) anchor locations in alpha-space
        log_targets:  (K,)   targets in log-space (e.g. log density at anchors)
        kernel:       'gaussian' or 'mq' recommended
        epsilon:      RBF length scale. If None, uses median pairwise distance.
        ridge:        diagonal regularization for stability
        normalize_alpha: standardize alpha dims before fitting/eval (recommended)
        """
        self.kernel = kernel
        self.ridge = float(ridge)
        self.normalize_alpha = bool(normalize_alpha)

        A = np.asarray(alpha_points, float)
        t = np.asarray(log_targets, float).reshape(-1)

        if A.ndim != 2:
            raise ValueError("alpha_points must be (K, D)")
        if t.shape[0] != A.shape[0]:
            raise ValueError("log_targets must have shape (K,) matching alpha_points")

        self.K, self.D = A.shape

        # Standardize alpha for better distance geometry
        if self.normalize_alpha:
            self.mean_ = A.mean(axis=0, keepdims=True)
            self.std_ = A.std(axis=0, keepdims=True) + 1e-12
            A = (A - self.mean_) / self.std_
        else:
            self.mean_ = np.zeros((1, self.D))
            self.std_ = np.ones((1, self.D))

        self.alpha_points = A
        self.log_targets = t

        # Choose epsilon automatically if needed
        if epsilon is None:
            d = pdist(self.alpha_points)
            eps = np.median(d) if d.size else 1.0
            if eps <= 0:
                eps = 1.0
            epsilon = eps
        self.epsilon = float(epsilon)

        # Build kernel matrix
        r = cdist(self.alpha_points, self.alpha_points)  # (K,K)
        Phi = self._rbf(r)

        Phi_reg = Phi + self.ridge * np.eye(self.K)

        # Solve for weights
        self.weights, *_ = np.linalg.lstsq(Phi_reg, self.log_targets, rcond=None)

    # ---------------- kernel ----------------
    def _rbf(self, r):
        if self.kernel == "gaussian":
            return np.exp(- (r / self.epsilon) ** 2)
        elif self.kernel == "mq":
            return np.sqrt(1.0 + (r / self.epsilon) ** 2)
        else:
            raise ValueError(f"Unknown kernel {self.kernel}. Use 'gaussian' or 'mq'.")

    # ---------------- eval ----------------
    def logq(self, alpha):
        """Log density (relative), alpha: (D,) or (N,D) -> scalar or (N,)"""
        a = np.asarray(alpha, float)
        if a.ndim == 1:
            a = a[None, :]
        if a.shape[1] != self.D:
            raise ValueError(f"alpha dim mismatch: got {a.shape[1]}, expected {self.D}")

        a = (a - self.mean_) / self.std_
        r = cdist(a, self.alpha_points)         # (N,K)
        vals = self._rbf(r)                     # (N,K)
        out = vals @ self.weights               # (N,)
        return out[0] if out.shape[0] == 1 else out

    def q(self, alpha):
        return np.exp(self.logq(alpha))

    # Backwards-compatible names
    def logN(self, alpha):  # interpret as log relative yield
        return self.logq(alpha)

    def N(self, alpha):
        return self.q(alpha)

    def ratio(self, base_alpha, goal_alpha):
        """q(goal)/q(base)"""
        return np.exp(self.logq(goal_alpha) - self.logq(base_alpha))

    # ---------------- fitting from accepted-only samples ----------------
    @classmethod
    def fit_from_samples(
        cls,
        alphas_accepted,
        centers=None,
        n_centers=50,
        center_method="kmeans",
        bandwidth=None,
        kernel="gaussian",
        epsilon=None,
        ridge=1e-6,
        normalize_alpha=True,
        seed=0,
    ):
        """
        Fit a density-on-alpha model from accepted-only alpha samples.

        alphas_accepted: (N,D) accepted alpha values

        centers: optional (K,D) anchor points where we estimate density targets.
                 If None, we choose centers from the samples (kmeans or random subset).

        bandwidth: KDE bandwidth in standardized alpha-space.
                   If None, uses median pairwise distance / sqrt(D).

        Returns: YieldModel_RBF instance trained on (centers, log density at centers).
        """
        A = np.asarray(alphas_accepted, float)
        if A.ndim != 2:
            raise ValueError("alphas_accepted must be (N,D)")
        N, D = A.shape

        rng = np.random.default_rng(seed)

        # Choose centers
        if centers is None:
            if center_method == "random":
                idx = rng.choice(N, size=min(n_centers, N), replace=False)
                centers = A[idx]
            elif center_method == "kmeans":
                # lightweight sklearn usage
                from sklearn.cluster import KMeans
                K = min(n_centers, N)
                km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(A)
                centers = km.cluster_centers_
            else:
                raise ValueError("center_method must be 'kmeans' or 'random'")
        centers = np.asarray(centers, float)

        # Standardize using centers+data (more stable geometry)
        if normalize_alpha:
            mean = A.mean(axis=0, keepdims=True)
            std = A.std(axis=0, keepdims=True) + 1e-12
            A_s = (A - mean) / std
            C_s = (centers - mean) / std
        else:
            A_s = A
            C_s = centers

        # Pick KDE bandwidth if not given
        if bandwidth is None:
            d = pdist(C_s) if C_s.shape[0] > 1 else np.array([1.0])
            bw = np.median(d)
            if bw <= 0:
                bw = 1.0
            bandwidth = bw / np.sqrt(D)

        # KDE-like log density targets at centers (up to constant)
        # log q(c) = log mean_i exp(-||c-a_i||^2/(2 bw^2))  (Gaussian kernel)
        d2 = cdist(C_s, A_s, metric="sqeuclidean")  # (K,N)
        log_kernel = -0.5 * d2 / (bandwidth ** 2)

        # log-mean-exp for stability
        m = log_kernel.max(axis=1, keepdims=True)
        logq = (m + np.log(np.mean(np.exp(log_kernel - m), axis=1, keepdims=True))).reshape(-1)

        # Fit RBF on these (center, logq) pairs
        model = cls(
            alpha_points=centers,
            log_targets=logq,
            kernel=kernel,
            epsilon=epsilon,
            ridge=ridge,
            normalize_alpha=normalize_alpha,
        )
        return model
    



class YieldModel_Poly:
    """Multivariate polynomial model for log-density / relative yield over alpha-space."""

    def __init__(self, alpha_points, log_targets, degree=1, normalize=True):
        alpha_points = np.asarray(alpha_points, float)
        log_targets = np.asarray(log_targets, float).reshape(-1)

        if alpha_points.ndim != 2:
            raise ValueError("alpha_points must be (K,D)")
        if log_targets.shape[0] != alpha_points.shape[0]:
            raise ValueError("log_targets must be (K,) matching alpha_points")

        self.degree = int(degree)
        self.D = alpha_points.shape[1]
        self.normalize = bool(normalize)

        if self.normalize:
            self.mean_ = alpha_points.mean(axis=0, keepdims=True)
            self.std_ = alpha_points.std(axis=0, keepdims=True) + 1e-12
            A = (alpha_points - self.mean_) / self.std_
        else:
            self.mean_ = np.zeros((1, self.D))
            self.std_ = np.ones((1, self.D))
            A = alpha_points

        self.exponents = self._build_exponents(self.D, self.degree)
        Phi = self._design(A)
        self.coeffs, *_ = np.linalg.lstsq(Phi, log_targets, rcond=None)

    @staticmethod
    def _build_exponents(D, degree):
        exps = []
        for total_deg in range(degree + 1):
            for comb in combinations_with_replacement(range(D), total_deg):
                e = [0] * D
                for idx in comb:
                    e[idx] += 1
                exps.append(tuple(e))
        return exps

    def _design(self, A):
        K, D = A.shape
        n_terms = len(self.exponents)
        Phi = np.ones((K, n_terms), dtype=float)
        for j, e in enumerate(self.exponents):
            if any(p != 0 for p in e):
                term = np.ones(K, dtype=float)
                for d, p in enumerate(e):
                    if p != 0:
                        term *= A[:, d] ** p
                Phi[:, j] = term
        return Phi

    def logq(self, alpha):
        a = np.asarray(alpha, float)
        if a.ndim == 1:
            a = a[None, :]
        a = (a - self.mean_) / self.std_
        Phi = self._design(a)
        out = Phi @ self.coeffs
        return out[0] if out.shape[0] == 1 else out

    def q(self, alpha):
        return np.exp(self.logq(alpha))

    # Backwards-compatible
    def logN(self, alpha):
        return self.logq(alpha)

    def N(self, alpha):
        return self.q(alpha)

    def ratio(self, base_alpha, goal_alpha):
        return np.exp(self.logq(goal_alpha) - self.logq(base_alpha))
    

class YieldModel_Spline:
    """
    Tensor-product cubic spline model for log-density / relative yield over alpha-space.

    Assumes alpha_points lie on a full Cartesian product grid.
    Uses RegularGridInterpolator with method='cubic' by default.
    """

    def __init__(
        self,
        alpha_points,
        log_targets,
        method="cubic",
        bounds_error=False,
        fill_value=None,
        ridge=0.0,
        normalize_alpha=True,
        ridge_passes=1,
    ):
        """
        alpha_points: (K, D) grid points (full grid, order doesn't matter)
        log_targets:  (K,)   values at those grid points (same order as alpha_points)
        method:       'linear', 'nearest', 'slinear', 'cubic', 'quintic' (SciPy)
        bounds_error: if True, querying outside grid raises error
        fill_value:   if not None, used outside grid; if None and bounds_error=False,
                      extrapolates (SciPy behavior depends on version; often linear extrap)
        ridge:        optional light smoothing strength (0 disables). Implemented as
                      repeated separable 1D smoothing passes on the grid.
        normalize_alpha: standardize alpha dims before building grid (recommended)
        ridge_passes: number of smoothing passes if ridge > 0
        """
        from scipy.interpolate import RegularGridInterpolator

        A = np.asarray(alpha_points, float)
        t = np.asarray(log_targets, float).reshape(-1)

        if A.ndim != 2:
            raise ValueError("alpha_points must be (K, D)")
        if t.shape[0] != A.shape[0]:
            raise ValueError("log_targets must have shape (K,) matching alpha_points")

        self.method = method
        self.bounds_error = bool(bounds_error)
        self.fill_value = fill_value
        self.normalize_alpha = bool(normalize_alpha)
        self.ridge = float(ridge)
        self.ridge_passes = int(ridge_passes)

        self.K, self.D = A.shape

        # Normalize for better conditioning / consistent scaling
        if self.normalize_alpha:
            self.mean_ = A.mean(axis=0, keepdims=True)
            self.std_ = A.std(axis=0, keepdims=True) + 1e-12
            A = (A - self.mean_) / self.std_
        else:
            self.mean_ = np.zeros((1, self.D))
            self.std_ = np.ones((1, self.D))

        # --- build grid axes from unique coordinate values per dimension ---
        axes = []
        for d in range(self.D):
            vals = np.unique(A[:, d])
            if vals.size < 2:
                raise ValueError(f"Dimension {d} has <2 unique grid values; can't spline.")
            axes.append(np.sort(vals))

        self.axes = axes
        shape = tuple(ax.size for ax in axes)
        if np.prod(shape) != self.K:
            raise ValueError(
                "alpha_points do not form a full Cartesian grid: "
                f"expected prod(grid_sizes)={int(np.prod(shape))}, got K={self.K}"
            )

        # --- map (x1,...,xD) -> grid index, fill grid array ---
        # Build index lookups for each axis value -> integer index
        index_maps = []
        for ax in axes:
            # exact float matching expected since points are on the grid
            index_maps.append({v: i for i, v in enumerate(ax.tolist())})

        grid = np.empty(shape, dtype=float)
        for row, val in zip(A, t):
            idx = tuple(index_maps[d][row[d]] for d in range(self.D))
            grid[idx] = val

        # Optional light smoothing ("ridge") on the grid
        if self.ridge > 0:
            grid = self._separable_smooth(grid, lam=self.ridge, passes=self.ridge_passes)

        self.grid_values = grid

        # Interpolator
        self._interp = RegularGridInterpolator(
            points=self.axes,
            values=self.grid_values,
            method=self.method,
            bounds_error=self.bounds_error,
            fill_value=self.fill_value,
        )

    @staticmethod
    def _separable_smooth(arr, lam=1e-3, passes=1):
        """
        Very lightweight smoothing: repeated 1D smoothing along each axis.
        This is not a full spline smoothing solve; it's a practical ridge-like stabilizer.
        """
        out = np.array(arr, copy=True)
        # simple tri-diagonal-like local averaging: out <- (1-2a)*x + a*shift(-1)+a*shift(+1)
        # a chosen from lam in a stable way
        a = lam / (1.0 + 2.0 * lam)

        for _ in range(max(1, int(passes))):
            for axis in range(out.ndim):
                x = out
                xm = np.roll(x, 1, axis=axis)
                xp = np.roll(x, -1, axis=axis)

                # avoid wraparound influence: keep boundaries unchanged
                slicer0 = [slice(None)] * out.ndim
                slicerN = [slice(None)] * out.ndim
                slicer0[axis] = 0
                slicerN[axis] = -1

                sm = (1 - 2 * a) * x + a * xm + a * xp
                sm[tuple(slicer0)] = x[tuple(slicer0)]
                sm[tuple(slicerN)] = x[tuple(slicerN)]
                out = sm
        return out

    # ---------------- eval ----------------
    def logq(self, alpha):
        """alpha: (D,) or (N,D) -> scalar or (N,)"""
        a = np.asarray(alpha, float)
        if a.ndim == 1:
            a = a[None, :]
        if a.shape[1] != self.D:
            raise ValueError(f"alpha dim mismatch: got {a.shape[1]}, expected {self.D}")

        a = (a - self.mean_) / self.std_
        out = self._interp(a)  # (N,)
        return out[0] if out.shape[0] == 1 else out

    def q(self, alpha):
        return np.exp(self.logq(alpha))

    # Backwards-compatible names
    def logN(self, alpha):
        return self.logq(alpha)

    def N(self, alpha):
        return self.q(alpha)

    def ratio(self, base_alpha, goal_alpha):
        """q(goal)/q(base)"""
        return np.exp(self.logq(goal_alpha) - self.logq(base_alpha))
    

import numpy as np


class YieldModel_SmoothingSpline:
    """
    Grid-based smoothing spline (penalized least squares with second differences)
    + tensor-product interpolation for evaluation.

    Objective on the full grid (vectorized):
        argmin_f ||f - y||^2 + lam * sum_d ||D2_d f||^2

    This is a discrete analogue of a smoothing spline / thin-plate penalty on a grid.
    """

    def __init__(
        self,
        alpha_points,
        log_targets,
        lam=1e-2,
        method="cubic",
        normalize_alpha=True,
        bounds_error=False,
        fill_value=None,
        cg_maxiter=2000,
        cg_tol=1e-8,
        use_precond=True,
    ):
        """
        alpha_points: (K, D) full grid points
        log_targets:  (K,)   values at those points
        lam:          smoothness strength (bigger => smoother). Typical range 1e-4..1e2
        method:       interpolation method for evaluation ('cubic' recommended)
        normalize_alpha: standardize alpha dims before building grid
        bounds_error/fill_value: passed to RegularGridInterpolator
        cg_maxiter/tol: conjugate gradient solve settings
        use_precond: use a simple diagonal preconditioner
        """
        from scipy.interpolate import RegularGridInterpolator
        from scipy.sparse.linalg import cg, LinearOperator

        A = np.asarray(alpha_points, float)
        y = np.asarray(log_targets, float).reshape(-1)

        if A.ndim != 2:
            raise ValueError("alpha_points must be (K, D)")
        if y.shape[0] != A.shape[0]:
            raise ValueError("log_targets must have shape (K,) matching alpha_points")

        self.lam = float(lam)
        self.method = method
        self.normalize_alpha = bool(normalize_alpha)
        self.bounds_error = bool(bounds_error)
        self.fill_value = fill_value

        self.K, self.D = A.shape

        # Normalize coordinates (helps conditioning and consistent lam meaning)
        if self.normalize_alpha:
            self.mean_ = A.mean(axis=0, keepdims=True)
            self.std_ = A.std(axis=0, keepdims=True) + 1e-12
            A = (A - self.mean_) / self.std_
        else:
            self.mean_ = np.zeros((1, self.D))
            self.std_ = np.ones((1, self.D))

        # Build axes for each dimension
        axes = []
        for d in range(self.D):
            vals = np.unique(A[:, d])
            if vals.size < 2:
                raise ValueError(f"Dimension {d} has <2 unique grid values.")
            axes.append(np.sort(vals))
        self.axes = axes

        shape = tuple(ax.size for ax in axes)
        if int(np.prod(shape)) != self.K:
            raise ValueError(
                "alpha_points do not form a full Cartesian grid: "
                f"expected prod(grid_sizes)={int(np.prod(shape))}, got K={self.K}"
            )
        self.shape = shape
        self.Num = int(np.prod(shape))

        # Fill grid Y in axis order
        index_maps = [{v: i for i, v in enumerate(ax.tolist())} for ax in axes]
        Y = np.empty(shape, dtype=float)
        for row, val in zip(A, y):
            idx = tuple(index_maps[d][row[d]] for d in range(self.D))
            Y[idx] = val
        self._Y = Y  # observed grid

        # We'll solve for F (smoothed grid)
        b = Y.reshape(-1)

        # Define linear operator Aop: (I + lam * L) x
        # where L = sum_d (D2_d^T D2_d) applied along each axis.
        def apply_L(vec):
            f = vec.reshape(shape)
            out = np.zeros_like(f)

            for axis in range(self.D):
                out += self._apply_D2tD2_axis(f, axis)

            return out.reshape(-1)

        def matvec(vec):
            return vec + self.lam * apply_L(vec)

        Aop = LinearOperator((self.Num, self.Num), matvec=matvec, dtype=float)

        # Simple diagonal preconditioner approximation: diag(I + lam*diag(L))
        M = None
        if use_precond:
            diagL = self._diag_D2tD2_sum(shape)
            Mdiag = 1.0 + self.lam * diagL
            def psolve(v):
                return v / Mdiag
            M = LinearOperator((self.Num, self.Num), matvec=psolve, dtype=float)

        # Solve with Conjugate Gradient (SPD system)
        x0 = b.copy()
        sol, info = cg(Aop, b, x0=x0, maxiter=cg_maxiter, M=M)
        if info != 0:
            # info>0: did not converge in maxiter, info<0: breakdown
            raise RuntimeError(f"CG did not converge (info={info}). Try smaller lam, looser tol, or fewer dims.")

        F = sol.reshape(shape)
        self.grid_values = F  # smoothed grid

        # Build interpolator over smoothed grid
        self._interp = RegularGridInterpolator(
            points=self.axes,
            values=self.grid_values,
            method=self.method,
            bounds_error=self.bounds_error,
            fill_value=self.fill_value,
        )

    # ----- discrete penalty helpers -----

    @staticmethod
    def _apply_D2tD2_axis(f, axis):
        """
        Apply (D2^T D2) along one axis with natural-like boundary handling.
        Uses 2nd differences on interior; yields a 5-point stencil in 1D.
        """
        # Move axis to front
        x = np.moveaxis(f, axis, 0)  # shape: (m, ...)
        m = x.shape[0]
        out = np.zeros_like(x)

        if m < 3:
            # can't define a second difference; return 0 penalty contribution
            return np.moveaxis(out, 0, axis)

        # Interior indices 2..m-3 get full 5-point stencil:
        # (D2^T D2) u at i = u[i-2] - 4u[i-1] + 6u[i] - 4u[i+1] + u[i+2]
        out[2:m-2] += (x[0:m-4] - 4*x[1:m-3] + 6*x[2:m-2] - 4*x[3:m-1] + x[4:m])

        # Near-boundary adjustments (consistent with using D2 only where defined: i=0..m-3)
        # These come from expanding D2^T D2 with "missing" terms at edges.
        # i = 0: from j=0 only => out0 += 1*u0 -2*u1 +1*u2
        out[0] += (1*x[0] - 2*x[1] + 1*x[2])
        # i = 1: from j=0,1 => out1 += -2*u0 +5*u1 -4*u2 +1*u3
        out[1] += (-2*x[0] + 5*x[1] - 4*x[2] + 1*x[3])
        # i = m-2: symmetric
        out[m-2] += (1*x[m-4] - 4*x[m-3] + 5*x[m-2] - 2*x[m-1])
        # i = m-1: symmetric
        out[m-1] += (1*x[m-3] - 2*x[m-2] + 1*x[m-1])

        return np.moveaxis(out, 0, axis)

    @staticmethod
    def _diag_D2tD2_1d(m):
        """Diagonal of D2^T D2 for length m (matches the stencil above)."""
        if m < 3:
            return np.zeros(m, dtype=float)
        d = np.zeros(m, dtype=float)
        d[0] = 1
        d[1] = 5
        d[2:m-2] = 6
        d[m-2] = 5
        d[m-1] = 1
        return d

    @classmethod
    def _diag_D2tD2_sum(cls, shape):
        """
        Diagonal of sum_d (D2_d^T D2_d) on the full grid (vectorized, flattened).
        This is used for a simple Jacobi preconditioner.
        """
        D = len(shape)
        diag = np.zeros(shape, dtype=float)
        for axis, m in enumerate(shape):
            d1 = cls._diag_D2tD2_1d(m)
            # broadcast along other dims
            shp = [1]*D
            shp[axis] = m
            diag += d1.reshape(shp)
        return diag.reshape(-1)

    # ----- API matching your RBF -----

    def logq(self, alpha):
        """alpha: (D,) or (N,D) -> scalar or (N,)"""
        a = np.asarray(alpha, float)
        if a.ndim == 1:
            a = a[None, :]
        if a.shape[1] != self.D:
            raise ValueError(f"alpha dim mismatch: got {a.shape[1]}, expected {self.D}")

        a = (a - self.mean_) / self.std_
        out = self._interp(a)
        return out[0] if out.shape[0] == 1 else out

    def q(self, alpha):
        return np.exp(self.logq(alpha))

    def logN(self, alpha):
        return self.logq(alpha)

    def N(self, alpha):
        return self.q(alpha)

    def ratio(self, base_alpha, goal_alpha):
        return np.exp(self.logq(goal_alpha) - self.logq(base_alpha))