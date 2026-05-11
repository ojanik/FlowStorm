"""
FlowSurface — jammy_flows (PyTorch) backend.

Three preprocessing modes for angular dimensions, controlled by constructor args:

1. DEFAULT (no special args)
   Pass (cos_zenith, azimuth) directly as Euclidean dims.
   Simple but has azimuth wrap-around discontinuity.

2. SPHERICAL  (s2_dim_pairs)
   Use jammy_flows S2 manifold. Geometrically correct but subflows are
   statistically independent (product manifold) and the 'n' layers + conditional
   inputs have a known CUDA scatter/gather bug — not recommended for now.
   s2_dim_pairs : list of (cos_zenith_col, azimuth_col) in raw x.
                  Converted to (theta, phi) with phi in [0, 2pi].

3. CARTESIAN  (cartesian_az_dims)
   Replace each azimuth column with (sin_az, cos_az). This removes the
   wrap-around discontinuity and keeps the flow purely Euclidean so all
   cross-dimension correlations are preserved. cos_zenith stays as-is
   (already smooth and bounded). x_dim expands by len(cartesian_az_dims).
   cartesian_az_dims : list of azimuth column indices in raw x to expand.
   normalize_dims    : column indices in CONVERTED space to standardise.
                       None -> auto: all non-sin/cos columns.

Modes 2 and 3 are mutually exclusive.

Example — energy + reco direction + true direction:
  inputs: [log10_reco_energy, cos_reco_zenith, reco_azimuth,
           log10_true_energy, cos_true_zenith,  true_azimuth]

  Cartesian (recommended):
    cartesian_az_dims = [2, 5]
    normalize_dims    = [0, 4]   # only the two energy cols (in converted space)
    -> flow sees 8-dim Euclidean: e8
       [log10_reco_e, cos_reco_zen, sin_reco_az, cos_reco_az,
        log10_true_e, cos_true_zen, sin_true_az, cos_true_az]

  Spherical (currently has CUDA bug with conditional inputs):
    s2_dim_pairs    = [(1, 2), (4, 5)]
    normalize_dims  = [0, 3]
    flow_manifold   = "e1+s2+e1+s2"
    flow_layers_str = "pppppppp+nnnnnnnn+pppppppp+nnnnnnnn"
"""

import json
import numpy as np
from pathlib import Path

import torch
import torch.optim as optim

import jammy_flows
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Default stability options for jammy_flows gaussianization layers
# ---------------------------------------------------------------------------
_DEFAULT_FLOW_OPTIONS: dict = {
    "g": {
        "fit_normalization": 0,
        "upper_bound_for_widths": 1.0,
        "lower_bound_for_widths": 0.01,
    },
    "t": {
        "cov_type": "full",
    },
}

_DEFAULT_YIELD_OPTIONS: dict = {}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


class FlowSurface:

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        x,
        alpha,
        weights=None,
        yield_weights=None,
        seed: int = 187,
        flow_kwargs=None,
        yield_kwargs=None,
        eps: float = 1e-12,
        flow_manifold: str = None,
        flow_layers_str: str = None,
        s2_dim_pairs: list = None,
        cartesian_az_dims: list = None,
        normalize_dims: list = None,
        flow_options_overwrite: dict = None,
        yield_options_overwrite: dict = None,
    ):
        if s2_dim_pairs and cartesian_az_dims:
            raise ValueError("s2_dim_pairs and cartesian_az_dims are mutually exclusive.")

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        flow_defaults  = dict(flow_layers=8)
        yield_defaults = dict(flow_layers=16)
        flow_kwargs    = {**flow_defaults,  **(flow_kwargs  or {})}
        yield_kwargs   = {**yield_defaults, **(yield_kwargs or {})}

        x     = _to_tensor(x)
        alpha = _to_tensor(alpha)

        x_dim_raw = int(x.shape[1])
        alpha_dim = int(alpha.shape[1])

        self._x_dim_raw         = x_dim_raw
        self._alpha_dim         = alpha_dim
        self._flow_kwargs       = dict(flow_kwargs)
        self._yield_kwargs      = dict(yield_kwargs)
        self._eps               = float(eps)
        self._flow_manifold     = flow_manifold
        self._flow_layers_str   = flow_layers_str
        self._s2_dim_pairs      = list(s2_dim_pairs)      if s2_dim_pairs      is not None else []
        self._cartesian_az_dims = list(cartesian_az_dims) if cartesian_az_dims is not None else []
        self._flow_options      = flow_options_overwrite  if flow_options_overwrite  is not None \
                                      else _DEFAULT_FLOW_OPTIONS
        self._yield_options     = yield_options_overwrite if yield_options_overwrite is not None \
                                      else _DEFAULT_YIELD_OPTIONS

        self._raw_to_conv, self._conv_to_raw, x_dim_conv = \
            self._build_cartesian_col_map(x_dim_raw)

        self._x_dim = x_dim_conv

        self._normalize_dims = list(range(self._x_dim))

        x = self._convert_s2_coords(x)
        x = self._convert_cartesian_az(x)


        self._x_conv_min = x.min(0).values.to(self.device)
        self._x_conv_max = x.max(0).values.to(self.device)

        x_mean = torch.zeros(self._x_dim)
        x_std  = torch.ones(self._x_dim)
        xl = self._logit_x(x)
        for d in self._normalize_dims:
            x_mean[d] = xl[:, d].mean()
            s = xl[:, d].std()
            x_std[d]  = s if s > 0 else 1.0

        self.x_mean = x_mean.to(self.device)
        self.x_std  = x_std.to(self.device)

        self.alpha_min   = alpha.min(0).values.to(self.device)
        self.alpha_max   = alpha.max(0).values.to(self.device)
        self.alpha_range = torch.where(
            (self.alpha_max - self.alpha_min) <= 0,
            torch.ones_like(self.alpha_max),
            self.alpha_max - self.alpha_min,
        ).to(self.device)

        _u = ((alpha.to(self.device) - self.alpha_min) / self.alpha_range
              )
        _u = (1 - 2 * self._eps) * _u + self._eps

        _logit = torch.log(_u / (1.0 - _u))
        self.alpha_logit_mean = _logit.mean(0)
        _logit_std = _logit.std(0)
        self.alpha_logit_std  = torch.where(
            _logit_std > 0, _logit_std, torch.ones_like(_logit_std)
        ).to(self.device)
        self._use_logit_alpha = True

        self._alpha_scale = (1.0 / (0.25 * self.alpha_range * self.alpha_logit_std)).to(self.device)


        self.x_train     = self.transform_x(x).to(self.device).double()
        self.alpha_train = self.transform_alpha(alpha).to(self.device).double()
        self.weights = _to_tensor(weights).to(self.device).double() if weights is not None else None

        if yield_weights is False:
            self.yield_weights = None
        elif yield_weights is not None:
            self.yield_weights = _to_tensor(yield_weights).to(self.device).double()
        else:
            self.yield_weights = self.weights

        n_flow_layers  = flow_kwargs["flow_layers"]
        n_yield_layers = yield_kwargs["flow_layers"]

        if flow_manifold is not None:
            manifold_str = flow_manifold
        else:
            manifold_str = f"e{self._x_dim}"

        self._x_flow_lo = None
        self._x_flow_hi = None

        if flow_layers_str is not None:
            x_flow_str = flow_layers_str
        else:
            n_submanifolds = manifold_str.count("+") + 1
            x_flow_str = "+".join(["g" * n_flow_layers + "t"] * n_submanifolds)

        self._alpha_flow_lo = None
        self._alpha_flow_hi = None

        yield_manifold_str = f"e{alpha_dim}"
        yield_str          = "g" * n_yield_layers + "t"

        self._resolved_flow_str        = x_flow_str
        self._resolved_yield_str       = yield_str
        self._resolved_yield_manifold  = yield_manifold_str
        self._resolved_flow_manifold   = manifold_str

        self.flow = jammy_flows.pdf(
            manifold_str, x_flow_str,
            conditional_input_dim=alpha_dim,
            options_overwrite=self._flow_options,
        ).double().to(self.device)

        self.yield_flow = jammy_flows.pdf(
            yield_manifold_str, yield_str,
            options_overwrite=self._yield_options,
        ).double().to(self.device)

        self.losses       = None
        self.yield_losses = None

        print("Nans in x:", np.isnan(self.x_train.cpu().float().numpy()).sum())
        print("Infs in x:", np.isinf(self.x_train.cpu().float().numpy()).sum())
        print("Nans in alphas:", np.isnan(self.alpha_train.cpu().float().numpy()).sum())
        print("Infs in alphas:", np.isinf(self.alpha_train.cpu().float().numpy()).sum())
        #self.plot_preprocessing(save_path="./")

    # ------------------------------------------------------------------
    # Cartesian column map helpers
    # ------------------------------------------------------------------

    def plot_preprocessing(self, save_path=None):
        import matplotlib.pyplot as plt
        print("PLOTTING")

        x_raw  = self.x_train.cpu().float().numpy()   # already preprocessed (logit+std)
        a_raw  = self.alpha_train.cpu().float().numpy()

        n_x = x_raw.shape[1]
        n_a = a_raw.shape[1]

        for data, n, prefix in [(x_raw, n_x, "x"), (a_raw, n_a, "alpha")]:
            n_cols = min(n, 4)
            n_rows = (n + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
            axes = np.array(axes).flatten()

            for i in range(n):
                axes[i].hist(data[:, i], bins=100, density=True)
                axes[i].set_title(f"{prefix}[{i}], mean={np.mean(data[:, i]):.2f}, std={np.std(data[:, i]):.2f}")
                axes[i].set_xlabel("preprocessed value")
            for j in range(n, len(axes)):
                axes[j].set_visible(False)

            fig.suptitle(f"{prefix} — after full preprocessing (logit + standardize)", fontsize=10)
            fig.tight_layout()

            if save_path is not None:
                p = Path(save_path) / f"debug_{prefix}_ppreprocessed.png"
                fig.savefig(p, dpi=150)
                print(f"Saved: {p}")
            else:
                plt.show()
            plt.close(fig)

    def _build_cartesian_col_map(self, x_dim_raw):
        az_set = set(self._cartesian_az_dims)
        raw_to_conv = {}
        conv_to_raw = {}
        conv_col = 0
        for raw_col in range(x_dim_raw):
            if raw_col in az_set:
                raw_to_conv[raw_col] = [conv_col, conv_col + 1]
                conv_to_raw[conv_col]     = (raw_col, 'sin')
                conv_to_raw[conv_col + 1] = (raw_col, 'cos')
                conv_col += 2
            else:
                raw_to_conv[raw_col] = [conv_col]
                conv_to_raw[conv_col] = (raw_col, 'passthrough')
                conv_col += 1
        return raw_to_conv, conv_to_raw, conv_col

    # ------------------------------------------------------------------
    # S2 coordinate conversion
    # ------------------------------------------------------------------

    def _convert_s2_coords(self, x):
        if not self._s2_dim_pairs:
            return x
        x = x.clone()
        two_pi = float(2.0 * np.pi)
        for cos_col, az_col in self._s2_dim_pairs:
            x[:, cos_col] = torch.acos(x[:, cos_col].clamp(-1.0 + 1e-6, 1.0 - 1e-6))
            x[:, az_col]  = x[:, az_col] % two_pi
        return x

    def _unconvert_s2_coords(self, x):
        if not self._s2_dim_pairs:
            return x
        x = x.clone()
        for cos_col, _ in self._s2_dim_pairs:
            x[:, cos_col] = torch.cos(x[:, cos_col])
        return x

    # ------------------------------------------------------------------
    # Cartesian azimuth conversion
    # ------------------------------------------------------------------

    def _convert_cartesian_az(self, x):
        if not self._cartesian_az_dims:
            return x
        cols = []
        for raw_col in range(self._x_dim_raw):
            if raw_col in set(self._cartesian_az_dims):
                az = x[:, raw_col]
                cols.append(torch.sin(az).unsqueeze(1))
                cols.append(torch.cos(az).unsqueeze(1))
            else:
                cols.append(x[:, raw_col : raw_col + 1])
        return torch.cat(cols, dim=1)

    def _unconvert_cartesian_az(self, x_conv):
        if not self._cartesian_az_dims:
            return x_conv
        out = torch.zeros(
            x_conv.shape[0], self._x_dim_raw,
            dtype=x_conv.dtype, device=x_conv.device,
        )
        two_pi = 2.0 * np.pi
        for raw_col, conv_cols in self._raw_to_conv.items():
            if len(conv_cols) == 1:
                out[:, raw_col] = x_conv[:, conv_cols[0]]
            else:
                az = torch.atan2(
                    x_conv[:, conv_cols[0]],
                    x_conv[:, conv_cols[1]],
                )
                out[:, raw_col] = az % two_pi
        return out

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _standardise(self, x):
        return (x.to(self.device) - self.x_mean) / self.x_std

    def _unstandardise(self, x_t):
        return x_t.to(self.device) * self.x_std + self.x_mean

    def transform_x(self, x):
        x = _to_tensor(x)
        x = self._logit_x(x)           # ← new
    
        return self._standardise(x)

    def retransform_x(self, x_t):
        x = self._unstandardise(_to_tensor(x_t))
        x = self._unlogit_x(x)         # ← new
        x = self._unconvert_cartesian_az(x)
        x = self._unconvert_s2_coords(x)
        return x
    
    def _logit_x(self, x):
        x = x.to(self.device)
        x_range = (self._x_conv_max - self._x_conv_min)
        u = ((x - self._x_conv_min) / x_range)
        u = (1 - 2 * self._eps) * u + self._eps
        return torch.log(u / (1.0 - u))

    def _unlogit_x(self, x_logit):
        x_logit = x_logit.to(self.device)
        u = torch.sigmoid(x_logit)
        return self._x_conv_min + u * (self._x_conv_max - self._x_conv_min)

    def transform_alpha(self, alpha):
        alpha = _to_tensor(alpha).to(self.device)
        u = (alpha - self.alpha_min) / self.alpha_range
        u = (1 - 2 * self._eps) * u + self._eps
        logit = torch.log(u / (1.0 - u))
        return (logit - self.alpha_logit_mean) / self.alpha_logit_std

    def retransform_alpha(self, a_t):
        a_t = _to_tensor(a_t).to(self.device)
        if not self._use_logit_alpha:
            return self.alpha_min + a_t * self.alpha_range
        logit = a_t * self.alpha_logit_std + self.alpha_logit_mean
        u = torch.sigmoid(logit)
        return self.alpha_min + u * self.alpha_range

    def _alpha_jacobian(self, alpha):
        alpha = _to_tensor(alpha).to(self.device)
        u = ((alpha - self.alpha_min) / self.alpha_range).clamp(self._eps, 1.0 - self._eps)
        if not self._use_logit_alpha:
            return 1.0 / self.alpha_range
        return 1.0 / (u * (1.0 - u) * self.alpha_range * self.alpha_logit_std)

    # ------------------------------------------------------------------
    # Internal log-prob wrappers
    # ------------------------------------------------------------------

    def _log_prob_flow(self, x_t, a_t):
        log_p, _, _ = self.flow(x_t, conditional_input=a_t)
        return log_p.view(-1)

    def _log_prob_yield(self, a_t):
        log_p, _, _ = self.yield_flow(a_t)
        return log_p.view(-1)

    # ------------------------------------------------------------------
    # Val split helper
    # ------------------------------------------------------------------

    @staticmethod
    def _split_tensors(val_split, *tensors):
        """Split tensors into (train, val) tuple-of-lists.
        val is None if val_split <= 0."""
        if val_split <= 0.0:
            return list(tensors), None
        N     = tensors[0].shape[0]
        n_val = max(1, int(N * val_split))
        perm  = torch.randperm(N, device=tensors[0].device)
        val_idx   = perm[:n_val]
        train_idx = perm[n_val:]
        train = [t[train_idx] for t in tensors]
        val   = [t[val_idx]   for t in tensors]
        return train, val

    # ------------------------------------------------------------------
    # Generic training loop
    # ------------------------------------------------------------------

    def _train_loop(
        self, model, loss_fn, tensors,
        learning_rate=3e-4, max_epochs=10, max_patience=3,
        batch_size=65536*32, label="", init_data=None,
        max_reinit_attempts=3,
        val_tensors=None,
    ):
        N         = tensors[0].shape[0]
        n_batches = max(N // batch_size, 1)

        def _init_and_push():
            if init_data is not None:
                model.init_params(data=init_data.double().cpu())
                model.to(init_data.device)

        _init_and_push()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=learning_rate / 100,
        )

        best_loss    = float("inf")
        patience     = 0
        all_losses   = []
        nan_total    = 0
        reinit_count = 0

        epoch_bar = tqdm(range(max_epochs), desc=f"{label} epochs", unit="epoch")
        for epoch in epoch_bar:
            epoch_loss    = 0.0
            valid_batches = 0
            perm     = torch.randperm(N, device=tensors[0].device)
            shuffled = [t[perm] for t in tensors]

            batch_bar = tqdm(range(n_batches), desc=f"  epoch {epoch+1}", leave=False, unit="batch")
            for b in batch_bar:
                s = b * batch_size
                batch = [t[s : s + batch_size] for t in shuffled]
                optimizer.zero_grad()

                try:
                    loss = loss_fn(batch)
                except Exception:
                    nan_total += 1
                    batch_bar.set_postfix(nll="fwd crash — skipped")
                    continue

                if not torch.isfinite(loss):
                    nan_total += 1
                    batch_bar.set_postfix(nll="NaN/Inf loss — skipped")
                    continue

                loss.backward()

                nan_in_grads = any(
                    p.grad is not None and p.grad.isnan().any()
                    for p in model.parameters()
                )
                if nan_in_grads:
                    nan_total += 1
                    optimizer.zero_grad()
                    batch_bar.set_postfix(nll="NaN grad — skipped")
                    continue

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if any(p.data.isnan().any() for p in model.parameters()):
                    nan_total += 1
                    if reinit_count < max_reinit_attempts:
                        reinit_count += 1
                        epoch_bar.write(
                            f"[{label}] NaN weights after step — reinitialising "
                            f"(attempt {reinit_count}/{max_reinit_attempts})"
                        )
                        _init_and_push()
                        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, T_max=max_epochs, eta_min=learning_rate / 100,
                        )
                        break
                    else:
                        epoch_bar.write(
                            f"[{label}] NaN weights after {max_reinit_attempts} "
                            f"reinit attempts — aborting."
                        )
                        return all_losses

                epoch_loss    += loss.item()
                valid_batches += 1
                batch_bar.set_postfix(nll=f"{loss.item():.5f}")

            if valid_batches == 0 and reinit_count >= max_reinit_attempts:
                epoch_bar.write(
                    f"[{label}] Epoch {epoch+1}: ALL batches NaN and reinit exhausted — aborting."
                )
                break

            if valid_batches == 0:
                continue

            train_avg = epoch_loss / valid_batches

            if val_tensors is not None:
                with torch.no_grad():
                    try:
                        val_loss = loss_fn(val_tensors).item()
                    except Exception:
                        val_loss = float("inf")
                metric = val_loss
                epoch_bar.set_postfix(
                    train_nll=f"{train_avg:.5f}",
                    val_nll=f"{val_loss:.5f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    patience=f"{patience}/{max_patience}",
                )
            else:
                metric = train_avg
                epoch_bar.set_postfix(
                    avg_nll=f"{train_avg:.5f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    patience=f"{patience}/{max_patience}",
                )

            all_losses.append(train_avg)
            scheduler.step()

            if metric < best_loss - 1e-8:
                best_loss = metric
                patience  = 0
            else:
                patience += 1
                if patience >= max_patience:
                    epoch_bar.write(f"[{label}] Early stopping at epoch {epoch + 1}.")
                    break

        if nan_total > 0:
            print(f"[{label}] Total NaN/Inf batches skipped: {nan_total}")

        return all_losses

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_flow(self, learning_rate=3e-4, max_epochs=10, max_patience=3,
                   batch_size=65536*32, val_split=0.1):
        base_tensors = (
            (self.x_train, self.alpha_train, self.weights)
            if self.weights is not None
            else (self.x_train, self.alpha_train)
        )
        tensors, val_tensors = self._split_tensors(val_split, *base_tensors)

        if self.weights is not None:
            def loss_fn(batch):
                x_b, a_b, w_b = batch
                log_p = self._log_prob_flow(x_b, a_b)
                w_b = w_b / (w_b.sum() + 1e-12)
                return -(w_b * log_p).sum()
        else:
            def loss_fn(batch):
                x_b, a_b = batch
                return -self._log_prob_flow(x_b, a_b).mean()

        init_data = self.x_train[:min(50_000, self.x_train.shape[0])]
        self.losses = self._train_loop(
            self.flow, loss_fn, tensors,
            learning_rate=learning_rate, max_epochs=max_epochs,
            max_patience=max_patience, batch_size=batch_size,
            label="flow p(x|alpha)", init_data=init_data,
            val_tensors=val_tensors,
        )

    def train_yield(self, learning_rate=3e-4, max_epochs=10, max_patience=3,
                    batch_size=65536*32, val_split=0.1):
        base_tensors = (
            (self.alpha_train, self.yield_weights)
            if self.yield_weights is not None
            else (self.alpha_train,)
        )
        tensors, val_tensors = self._split_tensors(val_split, *base_tensors)

        if self.yield_weights is not None:
            def loss_fn(batch):
                a_b, w_b = batch
                log_p = self._log_prob_yield(a_b)
                w_b = w_b / (w_b.sum() + 1e-12)
                return -(w_b * log_p).sum()
        else:
            def loss_fn(batch):
                (a_b,) = batch
                return -self._log_prob_yield(a_b).mean()

        init_data = self.alpha_train[:min(50_000, self.alpha_train.shape[0])]
        self.yield_losses = self._train_loop(
            self.yield_flow, loss_fn, tensors,
            learning_rate=learning_rate, max_epochs=max_epochs,
            max_patience=max_patience, batch_size=batch_size,
            label="yield p(alpha)", init_data=init_data,
            val_tensors=val_tensors,
        )

    def train_both(
        self,
        flow_learning_rate=3e-4, yield_learning_rate=3e-4,
        max_epochs=10, max_patience=3, batch_size=65536,
        val_split=0.1,
        order=("flow", "yield"),
    ):
        for which in order:
            if which == "yield":
                self.train_yield(
                    learning_rate=yield_learning_rate,
                    max_epochs=max_epochs, max_patience=max_patience,
                    batch_size=batch_size, val_split=val_split,
                )
            elif which == "flow":
                self.train_flow(
                    learning_rate=flow_learning_rate,
                    max_epochs=max_epochs, max_patience=max_patience,
                    batch_size=batch_size, val_split=val_split,
                )
            else:
                raise ValueError(f"Unknown training stage: {which!r}")

    # ------------------------------------------------------------------
    # Importance weights
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_weights(self, x_base, base_alpha, goal_alpha, batch_size=131072, include_yield=True):
        x_t      = self.transform_x(x_base).double()
        base_a_t = self.transform_alpha(torch.atleast_1d(_to_tensor(base_alpha))).double()
        goal_a_t = self.transform_alpha(torch.atleast_1d(_to_tensor(goal_alpha))).double()

        logpy_ratio = 0.0
        if include_yield:
            logpy_ratio = (self._log_prob_yield(goal_a_t.unsqueeze(0))[0]
                           - self._log_prob_yield(base_a_t.unsqueeze(0))[0])

        N, outs = x_t.shape[0], []
        for i in range(0, N, batch_size):
            xb   = x_t[i : i + batch_size]
            B    = xb.shape[0]
            base = base_a_t.unsqueeze(0).expand(B, -1)
            goal = goal_a_t.unsqueeze(0).expand(B, -1)
            outs.append(torch.exp(
                (self._log_prob_flow(xb, goal) - self._log_prob_flow(xb, base)) + logpy_ratio
            ))
        return torch.cat(outs, dim=0)

    # ------------------------------------------------------------------
    # Gradients / Hessians
    # ------------------------------------------------------------------

    def get_gradients(self, x_base, base_alpha, batch_size=8192, include_yield=True):
        x_t      = self.transform_x(x_base).double()
        base_a_t = self.transform_alpha(torch.atleast_1d(_to_tensor(base_alpha))).double()

        def log_prob_fn(a_t, x_one):
            lp = self._log_prob_flow(x_one.unsqueeze(0), a_t.unsqueeze(0))[0]
            if include_yield:
                lp = lp + self._log_prob_yield(a_t.unsqueeze(0))[0]
            return lp

        N, outs = x_t.shape[0], []
        for i in range(0, N, batch_size):
            xb = x_t[i : i + batch_size]
            grads = []
            for j in range(xb.shape[0]):
                a_t_leaf = base_a_t.detach().requires_grad_(True)
                g = torch.autograd.grad(log_prob_fn(a_t_leaf, xb[j].detach()), a_t_leaf)[0]
                grads.append(g)
            g_batch = torch.stack(grads, dim=0)
            outs.append(g_batch * self._alpha_scale.double().unsqueeze(0))
        return torch.cat(outs, dim=0)

    def get_hessians(self, x_base, base_alpha, batch_size=2048, include_yield=True):
        x_t      = self.transform_x(x_base).double()
        base_a_t = self.transform_alpha(torch.atleast_1d(_to_tensor(base_alpha))).double()

        def log_prob_fn(a_t, x_one):
            lp = self._log_prob_flow(x_one.unsqueeze(0), a_t.unsqueeze(0))[0]
            if include_yield:
                lp = lp + self._log_prob_yield(a_t.unsqueeze(0))[0]
            return lp

        N, outs, scale = x_t.shape[0], [], self._alpha_scale.double()
        for i in range(0, N, batch_size):
            xb = x_t[i : i + batch_size]
            hessians = []
            for j in range(xb.shape[0]):
                x_fixed = xb[j].detach()
                H = torch.autograd.functional.hessian(
                    lambda a: log_prob_fn(a, x_fixed), base_a_t.detach()
                )
                hessians.append(H)
            H_batch = torch.stack(hessians, dim=0)
            H_batch = H_batch * scale[None, :, None] * scale[None, None, :]
            outs.append(H_batch)
        return torch.cat(outs, dim=0)

    def get_weights_taylor(self, x_base, base_alpha, goal_alpha, order=1,
                           grads=None, hessians=None, clip_logw=50.0, include_yield=True):
        base_alpha = torch.atleast_1d(_to_tensor(base_alpha))
        goal_alpha = torch.atleast_1d(_to_tensor(goal_alpha))
        d = goal_alpha - base_alpha

        if grads is None:
            grads = self.get_gradients(x_base, base_alpha, include_yield=include_yield)
        logw = grads @ d.double()

        if order >= 2:
            if hessians is None:
                hessians = self.get_hessians(x_base, base_alpha, include_yield=include_yield)
            logw = logw + 0.5 * torch.einsum("i,nij,j->n", d.double(), hessians, d.double())

        return torch.exp(torch.clamp(logw, -clip_logw, clip_logw))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_alpha(self, n: int):
        a_t, _, _, _ = self.yield_flow.sample(samplesize=n)
        return self.retransform_alpha(a_t)

    @torch.no_grad()
    def sample_x(self, n: int, enforce_bounds: bool = True, max_attempts: int = 10):
        def _draw(n_draw):
            a_t, _, _, _ = self.yield_flow.sample(samplesize=n_draw)
            x_t, _, _, _ = self.flow.sample(conditional_input=a_t, samplesize=n_draw)
            return x_t

        if not enforce_bounds or self._x_conv_min is None:
            return self.retransform_x(_draw(n))

        lo = self._x_conv_min.double()
        hi = self._x_conv_max.double()

        collected, collected_n = [], 0
        for attempt in range(max_attempts):
            n_draw  = max(n - collected_n, 1) * 2
            x_t     = _draw(n_draw)
            x_unstd = self._unstandardise(x_t.float()).double()
            in_bounds = ((x_unstd >= lo) & (x_unstd <= hi)).all(dim=1)
            valid_t = x_t[in_bounds]
            collected.append(self.retransform_x(valid_t))
            collected_n += valid_t.shape[0]
            if collected_n >= n:
                break

        result = torch.cat(collected, dim=0)[:n]
        if result.shape[0] < n:
            print(f"[sample_x] Warning: only {result.shape[0]}/{n} in-bounds samples "
                  f"after {max_attempts} attempts.")
        return result

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta = dict(
            x_dim_raw=self._x_dim_raw,
            x_dim=self._x_dim,
            alpha_dim=self._alpha_dim,
            flow_kwargs=self._flow_kwargs,
            yield_kwargs=self._yield_kwargs,
            eps=self._eps,
            flow_manifold=self._flow_manifold,
            flow_layers_str=self._flow_layers_str,
            resolved_flow_manifold=self._resolved_flow_manifold,
            resolved_flow_str=self._resolved_flow_str,
            resolved_yield_manifold=self._resolved_yield_manifold,
            resolved_yield_str=self._resolved_yield_str,
            x_flow_lo=self._x_flow_lo,
            x_flow_hi=self._x_flow_hi,
            normalize_dims=self._normalize_dims,
            s2_dim_pairs=self._s2_dim_pairs,
            cartesian_az_dims=self._cartesian_az_dims,
            flow_options_overwrite=self._flow_options,
            yield_options_overwrite=self._yield_options,
            alpha_flow_kwargs=None,
            resolved_alpha_flow_str=None,
        )
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

        stats = dict(
            x_mean=self.x_mean.tolist(),
            x_std=self.x_std.tolist(),
            alpha_min=self.alpha_min.tolist(),
            alpha_max=self.alpha_max.tolist(),
            alpha_logit_mean=self.alpha_logit_mean.tolist(),
            alpha_logit_std=self.alpha_logit_std.tolist(),
            alpha_flow_lo=self._alpha_flow_lo,
            alpha_flow_hi=self._alpha_flow_hi,
            x_conv_min=self._x_conv_min.tolist(),
            x_conv_max=self._x_conv_max.tolist(),
        )
        (path / "stats.json").write_text(json.dumps(stats, indent=2))

        torch.save(self.flow.state_dict(),       path / "flow.pt")
        torch.save(self.yield_flow.state_dict(), path / "yield_flow.pt")

        if self.losses is not None:
            np.save(path / "losses.npy",       np.asarray(self.losses))
        if self.yield_losses is not None:
            np.save(path / "yield_losses.npy", np.asarray(self.yield_losses))

    @classmethod
    def load(cls, path, seed: int = 187):
        path  = Path(path)
        meta  = json.loads((path / "meta.json").read_text())
        stats = json.loads((path / "stats.json").read_text())

        x_dim_raw         = int(meta.get("x_dim_raw", meta["x_dim"]))
        alpha_dim         = int(meta["alpha_dim"])
        flow_kwargs       = dict(meta["flow_kwargs"])
        yield_kwargs      = dict(meta.get("yield_kwargs", {}))
        eps               = float(meta.get("eps", 1e-3))
        flow_manifold     = meta.get("flow_manifold", None)
        flow_layers_str   = meta.get("flow_layers_str", None)
        normalize_dims    = meta.get("normalize_dims", None)
        s2_dim_pairs_raw  = meta.get("s2_dim_pairs", None)
        s2_dim_pairs      = [tuple(p) for p in s2_dim_pairs_raw] if s2_dim_pairs_raw else None
        cartesian_az_dims = meta.get("cartesian_az_dims", None)

        flow_options_overwrite  = meta["flow_options_overwrite"]  if "flow_options_overwrite"  in meta else {}
        yield_options_overwrite = meta["yield_options_overwrite"] if "yield_options_overwrite" in meta else {}

        resolved_flow_str       = meta.get("resolved_flow_str",       None)
        resolved_flow_manifold  = meta.get("resolved_flow_manifold",  None)
        resolved_yield_str      = meta.get("resolved_yield_str",      None)
        resolved_yield_manifold = meta.get("resolved_yield_manifold", None)

        if resolved_flow_str is not None:
            flow_layers_str = resolved_flow_str
        if resolved_flow_manifold is not None:
            flow_manifold   = resolved_flow_manifold

        dummy_x        = torch.zeros(2, x_dim_raw)
        dummy_alpha    = torch.zeros(2, alpha_dim)
        dummy_alpha[0] = 0.2
        dummy_alpha[1] = 0.8

        self = cls(
            dummy_x, dummy_alpha,
            seed=seed,
            flow_kwargs=flow_kwargs,
            yield_kwargs=yield_kwargs,
            eps=eps,
            flow_manifold=flow_manifold,
            flow_layers_str=flow_layers_str,
            normalize_dims=normalize_dims,
            s2_dim_pairs=s2_dim_pairs,
            cartesian_az_dims=cartesian_az_dims,
            flow_options_overwrite=flow_options_overwrite,
            yield_options_overwrite=yield_options_overwrite,
        )

        if resolved_yield_str is not None and resolved_yield_manifold is not None:
            import jammy_flows as _jf
            self.yield_flow = _jf.pdf(
                resolved_yield_manifold, resolved_yield_str,
                options_overwrite=yield_options_overwrite,
            ).double().to(self.device)
            self._resolved_yield_manifold = resolved_yield_manifold
            self._resolved_yield_str      = resolved_yield_str

        dev = self.device
        self.x_mean    = torch.tensor(stats["x_mean"],    dtype=torch.float32).to(dev)
        self.x_std     = torch.tensor(stats["x_std"],     dtype=torch.float32).to(dev)
        self.alpha_min = torch.tensor(stats["alpha_min"], dtype=torch.float32).to(dev)
        self.alpha_max = torch.tensor(stats["alpha_max"], dtype=torch.float32).to(dev)
        self.alpha_range = torch.where(
            (self.alpha_max - self.alpha_min) <= 0,
            torch.ones_like(self.alpha_max),
            self.alpha_max - self.alpha_min,
        ).to(dev)

        if "alpha_logit_mean" in stats:
            self.alpha_logit_mean = torch.tensor(stats["alpha_logit_mean"], dtype=torch.float32).to(dev)
            self.alpha_logit_std  = torch.tensor(stats["alpha_logit_std"],  dtype=torch.float32).to(dev)
            self._use_logit_alpha = True
        else:
            self.alpha_logit_mean = torch.zeros_like(self.alpha_min)
            self.alpha_logit_std  = torch.ones_like(self.alpha_min)
            self._use_logit_alpha = False

        self._alpha_scale = (1.0 / (0.25 * self.alpha_range * self.alpha_logit_std)).to(dev)
        self._alpha_flow_lo = stats.get("alpha_flow_lo", None)
        self._alpha_flow_hi = stats.get("alpha_flow_hi", None)

        if "x_conv_min" in stats:
            self._x_conv_min = torch.tensor(stats["x_conv_min"], dtype=torch.float32).to(dev)
            self._x_conv_max = torch.tensor(stats["x_conv_max"], dtype=torch.float32).to(dev)
        else:
            self._x_conv_min = None
            self._x_conv_max = None

        self.flow.load_state_dict(torch.load(path / "flow.pt",            map_location=dev))
        self.yield_flow.load_state_dict(torch.load(path / "yield_flow.pt", map_location=dev))
        self.flow       = self.flow.to(dev)
        self.yield_flow = self.yield_flow.to(dev)

        self.x_train     = None
        self.alpha_train = None

        if (path / "losses.npy").exists():
            self.losses = np.load(path / "losses.npy", allow_pickle=True).tolist()
        if (path / "yield_losses.npy").exists():
            self.yield_losses = np.load(path / "yield_losses.npy", allow_pickle=True).tolist()

        return self