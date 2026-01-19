import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist, pdist
from scipy.special import gammaln


from FlowStorm.yield_model import YieldModel_RBF


def make_adaptive_anchors_density(
    alpha_all,
    n_anchors=200,
    seed=0,
    # ---- adaptive placement (pilot disagreement) ----
    n_pilot=60,
    pilot_method="kmeans",      # "kmeans" or "random"
    n_candidates=100_000,
    weight_mix=0.9,             # 0..1, keep some global coverage
    weight_power=1.0,           # >1 emphasizes peaks
    # ---- pilot KDE target (only used to build the pilot RBFs) ----
    bandwidth=None,             # in standardized space; None => median(pdist(centers))/sqrt(D)
    kernel="gaussian",
    eps_smooth_scale=2.0,
    eps_rough_scale=0.7,
    ridge_smooth=1e-3,
    ridge_rough=1e-6,
    normalize_alpha=True,
    # ---- final target: area-normalized mass/volume ----
    pseudo_count=1.0,           # avoids log(0) for empty-ish cells
    k_center=5,                 # kth nearest center used to set a "cell radius"
    min_radius=1e-6,            # clamp radius
):
    """
    Adaptive anchors + area-normalized log-density targets.

    Anchors:
      - weighted KMeans on candidates with weights from pilot-model disagreement

    Targets:
      - assign each sample to nearest anchor => count_i
      - estimate per-anchor cell volume_i from neighbor-center distances
      - logq_i = log(count_i + pseudo_count) - log(volume_i)  (minus constant ok)

    Returns:
      centers : (n_anchors, D)
      logq    : (n_anchors,)
    """
    from FlowStorm.yield_model import YieldModel_RBF  # avoid circular import at module load

    X = np.asarray(alpha_all, float)
    if X.ndim != 2:
        raise ValueError("alpha_all must be (N,D)")
    N, D = X.shape
    rng = np.random.default_rng(seed)

    # ---------- standardize geometry ----------
    if normalize_alpha:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-12
        Xs = (X - mean) / std
    else:
        mean = np.zeros((1, D))
        std = np.ones((1, D))
        Xs = X

    # ---------- helper: KDE-style logq at centers (for pilot only) ----------
    def kde_logq_at_centers(Cs):
        bw = bandwidth
        if bw is None:
            d = pdist(Cs) if Cs.shape[0] > 1 else np.array([1.0])
            bw0 = np.median(d)
            if bw0 <= 0:
                bw0 = 1.0
            bw = bw0 / np.sqrt(D)

        d2 = cdist(Cs, Xs, metric="sqeuclidean")
        logK = -0.5 * d2 / (bw ** 2)

        m = logK.max(axis=1, keepdims=True)
        logq = (m + np.log(np.mean(np.exp(logK - m), axis=1, keepdims=True))).reshape(-1)
        logq = logq - np.mean(logq)
        return logq

    # ---------- pilot anchors ----------
    n_pilot = int(min(max(2, n_pilot), N))
    if pilot_method == "random":
        idx = rng.choice(N, size=n_pilot, replace=False)
        centers_pilot = X[idx]
    elif pilot_method == "kmeans":
        km0 = KMeans(n_clusters=n_pilot, n_init=10, random_state=seed).fit(X)
        centers_pilot = km0.cluster_centers_
    else:
        raise ValueError("pilot_method must be 'kmeans' or 'random'")

    Cp = (centers_pilot - mean) / std
    logq_pilot = kde_logq_at_centers(Cp)

    # Fit two pilot RBFs (rough vs smooth) to create a "needs resolution" score
    m_base = YieldModel_RBF(
        alpha_points=centers_pilot,
        log_targets=logq_pilot,
        kernel=kernel,
        epsilon=None,
        ridge=ridge_rough,
        normalize_alpha=normalize_alpha,
    )
    eps0 = m_base.epsilon

    m_smooth = YieldModel_RBF(
        centers_pilot, logq_pilot,
        kernel=kernel,
        epsilon=eps_smooth_scale * eps0,
        ridge=ridge_smooth,
        normalize_alpha=normalize_alpha,
    )
    m_rough = YieldModel_RBF(
        centers_pilot, logq_pilot,
        kernel=kernel,
        epsilon=eps_rough_scale * eps0,
        ridge=ridge_rough,
        normalize_alpha=normalize_alpha,
    )

    # ---------- candidates + weights ----------
    n_candidates = int(min(max(5000, n_candidates), N))
    cand_idx = rng.choice(N, size=n_candidates, replace=False)
    CAND = X[cand_idx]

    score = np.abs(m_rough.logq(CAND) - m_smooth.logq(CAND)).reshape(-1) + 1e-12
    if weight_power != 1.0:
        score = score ** float(weight_power)

    # keep global coverage
    w = (1.0 - float(weight_mix)) + float(weight_mix) * (score / np.mean(score))
    w = np.clip(w, 1e-12, None)

    # ---------- weighted KMeans => final anchors ----------
    km = KMeans(n_clusters=int(n_anchors), n_init=10, random_state=seed)
    km.fit(CAND, sample_weight=w)
    centers = km.cluster_centers_

    # ---------- compute area-normalized targets ----------
    # 1) Assign every sample to nearest center => counts per anchor
    # (for big N you can subsample here if needed)
    nn = NearestNeighbors(n_neighbors=1).fit(centers)
    labels = nn.kneighbors(X, return_distance=False).reshape(-1)
    counts = np.bincount(labels, minlength=int(n_anchors)).astype(float)

    # 2) Estimate per-anchor "cell volume" from kth nearest center distance (in standardized space)
    Cs = (centers - mean) / std
    k_center = int(max(2, min(k_center, len(Cs))))
    nnc = NearestNeighbors(n_neighbors=k_center).fit(Cs)
    dcc, _ = nnc.kneighbors(Cs)          # includes self (0)
    r = np.maximum(dcc[:, -1], float(min_radius))

    # volume of D-ball: V = V_D * r^D
    logV_unit = (D / 2) * np.log(np.pi) - gammaln(D / 2 + 1)
    log_volume = logV_unit + D * np.log(r)

    # 3) log density proxy (constant offsets irrelevant)
    logq = np.log(counts + float(pseudo_count)) - log_volume
    logq = logq - np.mean(logq)

    return centers, logq

def make_snowstorm_anchors_density(alpha_all, n_anchors=15, seed=0, bandwidth=None):
    alpha_all = np.asarray(alpha_all, float)
    km = KMeans(n_clusters=n_anchors, n_init=10, random_state=seed).fit(alpha_all)
    centers = km.cluster_centers_

    # standardize for stable distance scaling
    mean = alpha_all.mean(axis=0, keepdims=True)
    std  = alpha_all.std(axis=0, keepdims=True) + 1e-12
    A = (alpha_all - mean) / std
    C = (centers - mean) / std

    # choose bandwidth if not given
    if bandwidth is None:
        # median pairwise distance between centers / sqrt(D)
        from scipy.spatial.distance import pdist
        d = pdist(C) if len(C) > 1 else np.array([1.0])
        bw = np.median(d)
        if bw <= 0: bw = 1.0
        bandwidth = bw / np.sqrt(C.shape[1])

    # KDE log-density targets at centers (up to constant)
    d2 = cdist(C, A, metric="sqeuclidean")
    logK = -0.5 * d2 / (bandwidth**2)

    m = logK.max(axis=1, keepdims=True)
    logq = (m + np.log(np.mean(np.exp(logK - m), axis=1, keepdims=True))).reshape(-1)

    return centers, logq

def make_uniform_grid_anchors_density(
    alpha_all,
    n_anchors=51,
    bounds=None,
    min_per_dim=2,
    pseudo_count=1.0,
):
    """
    Uniform-grid anchors + log-density targets from accepted-only samples.

    Returns (centers, logq) just like make_snowstorm_anchors_density.

    Idea:
      - Build equal-volume ND grid cells (uniform spacing).
      - Count accepted events per cell.
      - Convert counts -> log density up to a constant:
            q(cell) ∝ count / volume
        Since all volumes are equal, logq ∝ log(count).
      - We return logq (up to an additive constant), which is all you need for ratios.

    Parameters
    ----------
    alpha_all : (N, D)
        Accepted alpha samples.
    n_anchors : int
        Target ~ number of grid cells (actual = n_bins**D).
    bounds : None or (D,2)
        Use known uniform box bounds if possible (recommended).
    min_per_dim : int
        Minimum bins per dimension.
    pseudo_count : float
        Additive smoothing to avoid log(0). Use 0.5 or 1.0 typically.

    Returns
    -------
    centers : (K, D)
        Grid cell centers.
    logq : (K,)
        Log density proxy at each center (up to additive constant).
    """
    A = np.asarray(alpha_all, float)
    if A.ndim != 2:
        raise ValueError("alpha_all must be (N, D)")
    N, D = A.shape

    if bounds is None:
        lo = A.min(axis=0)
        hi = A.max(axis=0)
    else:
        b = np.asarray(bounds, float)
        if b.shape != (D, 2):
            raise ValueError(f"bounds must be shape (D,2), got {b.shape}")
        lo = b[:, 0]
        hi = b[:, 1]

    # Choose bins per dimension so total cells ~ n_anchors
    n_bins = int(np.round(n_anchors ** (1.0 / D)))
    n_bins = max(n_bins, min_per_dim)
    n_bins_per_dim = np.full(D, n_bins, dtype=int)

    # Edges and centers
    edges = [np.linspace(lo[d], hi[d], n_bins_per_dim[d] + 1) for d in range(D)]
    centers_1d = [0.5 * (e[1:] + e[:-1]) for e in edges]

    mesh = np.meshgrid(*centers_1d, indexing="ij")
    centers = np.stack([m.reshape(-1) for m in mesh], axis=1)  # (K,D)
    K = centers.shape[0]

    # Bin indices for each sample
    idxs = []
    for d in range(D):
        idx = np.digitize(A[:, d], edges[d]) - 1
        idx = np.clip(idx, 0, n_bins_per_dim[d] - 1)
        idxs.append(idx)
    idxs = np.stack(idxs, axis=1)  # (N,D)

    # Flatten ND index to 1D
    strides = np.cumprod([1] + list(n_bins_per_dim[::-1]))[::-1][1:]
    flat = (idxs * strides).sum(axis=1)

    counts = np.bincount(flat, minlength=K).astype(float)

    # Equal-volume cells => log density differs by constant from log(count)
    logq = np.log(counts + float(pseudo_count))

    # (Optional) remove a constant offset for nicer numbers; ratios unchanged
    logq = logq - np.mean(logq)

    return centers, logq