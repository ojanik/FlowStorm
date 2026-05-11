"""
AlphaFlow — models p(alpha | x) using jammy_flows (PyTorch).

Trains a conditional normalizing flow where:
  - target:    alpha (systematic parameters), alpha_dim-dimensional
  - condition: x (event observables), x_dim-dimensional

Preprocessing:
  - x:     cartesian azimuth expansion + standardization (same as FlowSurface)
  - alpha: min-max normalize -> logit -> standardize -> Euclidean flow (g*n + t)

Importance weights:
    w(x, alpha_goal) = p(alpha_goal | x) / p(alpha_nom | x)

This implicitly includes the yield term via Bayes' theorem:
    p(alpha|x) / p(alpha_nom|x)
    = [p(x|alpha) * p(alpha)] / [p(x|alpha_nom) * p(alpha_nom)]

so no separate yield flow is needed.

Gradients and Hessians of log p(alpha|x) w.r.t. raw alpha are also provided,
and are equivalent to those from FlowSurface with include_yield=True.

Example usage:
    flow = AlphaFlow(x, alpha, weights=w,
                     cartesian_az_dims=[2, 5], normalize_dims=[0, 4])
    flow.train()
    w = flow.get_weights(x_data, base_alpha, goal_alpha)
"""

import json
import numpy as np
from pathlib import Path

import torch
import torch.optim as optim

import jammy_flows
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Default options
# ---------------------------------------------------------------------------
_DEFAULT_OPTIONS: dict = {
    "g": {
        "fit_normalization": 0,
        "upper_bound_for_widths": 1.0,
        "lower_bound_for_widths": 0.01,
    },
    "t": {
        "cov_type": "full",
    },
}


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


class AlphaFlow:

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        x,
        alpha,
        weights=None,
        seed: int = 187,
        flow_layers: int = 8,
        eps: float = 1e-3,
        # --- x preprocessing: cartesian azimuth ---
        cartesian_az_dims: list = None,
        normalize_dims: list = None,
        # --- flow options ---
        options_overwrite: dict = None,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._flow_layers       = flow_layers
        self._eps               = float(eps)
        self._cartesian_az_dims = list(cartesian_az_dims) if cartesian_az_dims else []
        self._options           = options_overwrite if options_overwrite is not None \
                                  else _DEFAULT_OPTIONS

        x     = _to_tensor(x)
        alpha = _to_tensor(alpha)

        x_dim_raw = int(x.shape[1])
        alpha_dim = int(alpha.shape[1])
        self._x_dim_raw = x_dim_raw
        self._alpha_dim = alpha_dim

        # ---- x preprocessing ------------------------------------------------
        self._raw_to_conv, self._conv_to_raw, x_dim_conv = \
            self._build_cartesian_col_map(x_dim_raw)
        self._x_dim = x_dim_conv

        if normalize_dims is not None:
            self._normalize_dims = list(normalize_dims)
        elif self._cartesian_az_dims:
            self._normalize_dims = [
                c for c, (_, kind) in self._conv_to_raw.items()
                if kind == "passthrough"
            ]
        else:
            self._normalize_dims = list(range(x_dim_conv))

        x_conv = self._convert_cartesian_az(x)

        x_mean = torch.zeros(self._x_dim)
        x_std  = torch.ones(self._x_dim)
        for d in self._normalize_dims:
            x_mean[d] = x_conv[:, d].mean()
            s = x_conv[:, d].std()
            x_std[d]  = s if s > 0 else 1.0
        self.x_mean = x_mean.to(self.device)
        self.x_std  = x_std.to(self.device)

        # ---- alpha preprocessing: normalize -> logit -> standardize ---------
        self.alpha_min   = alpha.min(0).values.to(self.device)
        self.alpha_max   = alpha.max(0).values.to(self.device)
        self.alpha_range = torch.where(
            (self.alpha_max - self.alpha_min) <= 0,
            torch.ones_like(self.alpha_max),
            self.alpha_max - self.alpha_min,
        ).to(self.device)

        _u = ((alpha.to(self.device) - self.alpha_min) / self.alpha_range
              ).clamp(self._eps, 1.0 - self._eps)
        _logit = torch.log(_u / (1.0 - _u))
        self.alpha_logit_mean = _logit.mean(0)
        _logit_std = _logit.std(0)
        self.alpha_logit_std  = torch.where(
            _logit_std > 0, _logit_std, torch.ones_like(_logit_std)
        ).to(self.device)

        # _alpha_scale: approximate d(transform_alpha)/d(alpha) at u=0.5
        self._alpha_scale = (
            1.0 / (0.25 * self.alpha_range * self.alpha_logit_std)
        ).to(self.device)

        # ---- preprocessed training data -------------------------------------
        self.x_train     = self._standardise_x(x_conv).to(self.device).double()
        self.alpha_train = self.transform_alpha(alpha).to(self.device).double()
        self.weights     = _to_tensor(weights).to(self.device).double() \
                           if weights is not None else None

        # ---- build flow: p(alpha | x) ---------------------------------------
        manifold_str = f"e{alpha_dim}"
        flow_str     = "g" * flow_layers + "t"
        self._resolved_manifold = manifold_str
        self._resolved_flow_str = flow_str

        self.flow = jammy_flows.pdf(
            manifold_str, flow_str,
            conditional_input_dim=self._x_dim,
            options_overwrite=self._options,
        ).double().to(self.device)

        self.losses = None

    # ------------------------------------------------------------------
    # Cartesian column map
    # ------------------------------------------------------------------

    def _build_cartesian_col_map(self, x_dim_raw):
        az_set = set(self._cartesian_az_dims)
        raw_to_conv, conv_to_raw = {}, {}
        conv_col = 0
        for raw_col in range(x_dim_raw):
            if raw_col in az_set:
                raw_to_conv[raw_col] = [conv_col, conv_col + 1]
                conv_to_raw[conv_col]     = (raw_col, "sin")
                conv_to_raw[conv_col + 1] = (raw_col, "cos")
                conv_col += 2
            else:
                raw_to_conv[raw_col] = [conv_col]
                conv_to_raw[conv_col] = (raw_col, "passthrough")
                conv_col += 1
        return raw_to_conv, conv_to_raw, conv_col

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

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _standardise_x(self, x_conv):
        return (x_conv.to(self.device) - self.x_mean) / self.x_std

    def transform_x(self, x):
        """Raw x -> flow input (cartesian expansion + standardization)."""
        x = _to_tensor(x)
        x_conv = self._convert_cartesian_az(x)
        return self._standardise_x(x_conv)

    def transform_alpha(self, alpha):
        """Raw alpha -> flow space: normalize -> logit -> standardize."""
        alpha = _to_tensor(alpha).to(self.device)
        u = (alpha - self.alpha_min) / self.alpha_range
        u = u.clamp(self._eps, 1.0 - self._eps)
        logit = torch.log(u / (1.0 - u))
        return (logit - self.alpha_logit_mean) / self.alpha_logit_std

    def retransform_alpha(self, a_t):
        """Flow space -> raw alpha: unstandardize -> sigmoid -> denormalize."""
        a_t = _to_tensor(a_t).to(self.device)
        logit = a_t * self.alpha_logit_std + self.alpha_logit_mean
        u = torch.sigmoid(logit)
        return self.alpha_min + u * self.alpha_range

    def _alpha_jacobian(self, alpha):
        """Exact d(transform_alpha)/d(alpha) per dimension."""
        alpha = _to_tensor(alpha).to(self.device)
        u = ((alpha - self.alpha_min) / self.alpha_range).clamp(self._eps, 1.0 - self._eps)
        return 1.0 / (u * (1.0 - u) * self.alpha_range * self.alpha_logit_std)

    # ------------------------------------------------------------------
    # Internal log-prob
    # ------------------------------------------------------------------

    def _log_prob(self, a_t, x_t):
        log_p, _, _ = self.flow(a_t, conditional_input=x_t)
        return log_p.view(-1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, learning_rate=3e-4, max_epochs=100, max_patience=5,
              batch_size=65536, max_reinit_attempts=3):
        """Train p(alpha | x)."""

        N         = self.x_train.shape[0]
        n_batches = max(N // batch_size, 1)

        def _init_and_push():
            init_data = self.alpha_train[:min(50_000, N)]
            self.flow.init_params(data=init_data.double().cpu())
            self.flow.to(self.device)

        _init_and_push()
        optimizer = optim.Adam(self.flow.parameters(), lr=learning_rate)

        if self.weights is not None:
            def loss_fn(a_b, x_b, w_b):
                log_p = self._log_prob(a_b, x_b)
                w_b   = w_b / (w_b.sum() + 1e-12)
                return -(w_b * log_p).sum()
        else:
            def loss_fn(a_b, x_b, w_b=None):
                return -self._log_prob(a_b, x_b).mean()

        best_loss    = float("inf")
        patience     = 0
        all_losses   = []
        nan_total    = 0
        reinit_count = 0

        epoch_bar = tqdm(range(max_epochs), desc="AlphaFlow epochs", unit="epoch")
        for epoch in epoch_bar:
            epoch_loss    = 0.0
            valid_batches = 0
            perm   = torch.randperm(N, device=self.device)
            a_shuf = self.alpha_train[perm]
            x_shuf = self.x_train[perm]
            w_shuf = self.weights[perm] if self.weights is not None else None

            batch_bar = tqdm(range(n_batches), desc=f"  epoch {epoch+1}",
                             leave=False, unit="batch")
            for b in batch_bar:
                s   = b * batch_size
                a_b = a_shuf[s : s + batch_size]
                x_b = x_shuf[s : s + batch_size]
                w_b = w_shuf[s : s + batch_size] if w_shuf is not None else None

                optimizer.zero_grad()
                try:
                    loss = loss_fn(a_b, x_b, w_b)
                except Exception:
                    nan_total += 1
                    batch_bar.set_postfix(nll="fwd crash — skipped")
                    continue

                if not torch.isfinite(loss):
                    nan_total += 1
                    batch_bar.set_postfix(nll="NaN/Inf — skipped")
                    continue

                loss.backward()

                if any(p.grad is not None and p.grad.isnan().any()
                       for p in self.flow.parameters()):
                    nan_total += 1
                    optimizer.zero_grad()
                    batch_bar.set_postfix(nll="NaN grad — skipped")
                    continue

                torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=1.0)
                optimizer.step()

                if any(p.data.isnan().any() for p in self.flow.parameters()):
                    nan_total += 1
                    if reinit_count < max_reinit_attempts:
                        reinit_count += 1
                        epoch_bar.write(
                            f"NaN weights — reinit {reinit_count}/{max_reinit_attempts}"
                        )
                        _init_and_push()
                        optimizer = optim.Adam(self.flow.parameters(), lr=learning_rate)
                        break
                    else:
                        epoch_bar.write("NaN weights — max reinits reached, aborting.")
                        self.losses = all_losses
                        return

                epoch_loss    += loss.item()
                valid_batches += 1
                batch_bar.set_postfix(nll=f"{loss.item():.5f}")

            if valid_batches == 0:
                continue

            avg = epoch_loss / valid_batches
            all_losses.append(avg)
            epoch_bar.set_postfix(avg_nll=f"{avg:.5f}", patience=f"{patience}/{max_patience}")

            if avg < best_loss - 1e-8:
                best_loss = avg
                patience  = 0
            else:
                patience += 1
                if patience >= max_patience:
                    epoch_bar.write(f"Early stopping at epoch {epoch + 1}.")
                    break

        if nan_total > 0:
            print(f"[AlphaFlow] Total NaN/Inf batches skipped: {nan_total}")

        self.losses = all_losses

    # ------------------------------------------------------------------
    # Importance weights
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_weights(self, x_base, base_alpha, goal_alpha, batch_size=131072):
        """
        Importance weights: p(alpha_goal | x) / p(alpha_base | x).

        Equivalent to FlowSurface.get_weights(include_yield=True) since by
        Bayes the yield term is implicitly included.
        """
        x_t      = self.transform_x(x_base).double()
        base_a_t = self.transform_alpha(torch.atleast_1d(_to_tensor(base_alpha))).double()
        goal_a_t = self.transform_alpha(torch.atleast_1d(_to_tensor(goal_alpha))).double()

        N, outs = x_t.shape[0], []
        for i in range(0, N, batch_size):
            xb   = x_t[i : i + batch_size]
            B    = xb.shape[0]
            base = base_a_t.unsqueeze(0).expand(B, -1)
            goal = goal_a_t.unsqueeze(0).expand(B, -1)
            log_w = self._log_prob(goal, xb) - self._log_prob(base, xb)
            outs.append(torch.exp(log_w))
        return torch.cat(outs, dim=0)

    # ------------------------------------------------------------------
    # Gradients / Hessians
    # ------------------------------------------------------------------

    def get_gradients(self, x_base, base_alpha, batch_size=8192):
        """d log p(alpha|x) / d alpha at base_alpha, per event in x_base."""
        x_t      = self.transform_x(x_base).double()
        base_a_t = self.transform_alpha(torch.atleast_1d(_to_tensor(base_alpha))).double()
        jac      = self._alpha_jacobian(torch.atleast_1d(_to_tensor(base_alpha))).double()

        def log_prob_fn(a_t, x_one):
            return self._log_prob(a_t.unsqueeze(0), x_one.unsqueeze(0))[0]

        N, outs = x_t.shape[0], []
        for i in range(0, N, batch_size):
            xb = x_t[i : i + batch_size]
            grads = []
            for j in range(xb.shape[0]):
                a_t_leaf = base_a_t.detach().requires_grad_(True)
                g = torch.autograd.grad(
                    log_prob_fn(a_t_leaf, xb[j].detach()), a_t_leaf
                )[0]
                grads.append(g)
            g_batch = torch.stack(grads, dim=0)
            outs.append(g_batch * jac.unsqueeze(0))
        return torch.cat(outs, dim=0)

    def get_hessians(self, x_base, base_alpha, batch_size=2048):
        """d² log p(alpha|x) / d alpha² at base_alpha, per event in x_base."""
        x_t      = self.transform_x(x_base).double()
        base_a_t = self.transform_alpha(torch.atleast_1d(_to_tensor(base_alpha))).double()
        jac      = self._alpha_jacobian(torch.atleast_1d(_to_tensor(base_alpha))).double()

        def log_prob_fn(a_t, x_one):
            return self._log_prob(a_t.unsqueeze(0), x_one.unsqueeze(0))[0]

        N, outs = x_t.shape[0], []
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
            H_batch = H_batch * jac[None, :, None] * jac[None, None, :]
            outs.append(H_batch)
        return torch.cat(outs, dim=0)

    def get_weights_taylor(self, x_base, base_alpha, goal_alpha, order=1,
                           grads=None, hessians=None, clip_logw=50.0):
        """Taylor-expanded importance weights using gradients/hessians."""
        base_alpha = torch.atleast_1d(_to_tensor(base_alpha))
        goal_alpha = torch.atleast_1d(_to_tensor(goal_alpha))
        d = goal_alpha - base_alpha

        if grads is None:
            grads = self.get_gradients(x_base, base_alpha)
        logw = grads @ d.double()

        if order >= 2:
            if hessians is None:
                hessians = self.get_hessians(x_base, base_alpha)
            logw = logw + 0.5 * torch.einsum(
                "i,nij,j->n", d.double(), hessians, d.double()
            )
        return torch.exp(torch.clamp(logw, -clip_logw, clip_logw))

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
            flow_layers=self._flow_layers,
            eps=self._eps,
            cartesian_az_dims=self._cartesian_az_dims,
            normalize_dims=self._normalize_dims,
            options_overwrite=self._options,
            resolved_manifold=self._resolved_manifold,
            resolved_flow_str=self._resolved_flow_str,
        )
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

        stats = dict(
            x_mean=self.x_mean.tolist(),
            x_std=self.x_std.tolist(),
            alpha_min=self.alpha_min.tolist(),
            alpha_max=self.alpha_max.tolist(),
            alpha_logit_mean=self.alpha_logit_mean.tolist(),
            alpha_logit_std=self.alpha_logit_std.tolist(),
        )
        (path / "stats.json").write_text(json.dumps(stats, indent=2))

        torch.save(self.flow.state_dict(), path / "flow.pt")

        if self.losses is not None:
            np.save(path / "losses.npy", np.asarray(self.losses))

    @classmethod
    def load(cls, path, seed: int = 187):
        path  = Path(path)
        meta  = json.loads((path / "meta.json").read_text())
        stats = json.loads((path / "stats.json").read_text())

        x_dim_raw         = int(meta["x_dim_raw"])
        alpha_dim         = int(meta["alpha_dim"])
        flow_layers       = int(meta["flow_layers"])
        eps               = float(meta.get("eps", 1e-3))
        cartesian_az_dims = meta.get("cartesian_az_dims", None)
        normalize_dims    = meta.get("normalize_dims", None)
        options_overwrite = meta.get("options_overwrite", None)

        dummy_x        = torch.zeros(2, x_dim_raw)
        dummy_alpha    = torch.zeros(2, alpha_dim)
        dummy_alpha[0] = 0.2
        dummy_alpha[1] = 0.8

        self = cls(
            dummy_x, dummy_alpha,
            seed=seed,
            flow_layers=flow_layers,
            eps=eps,
            cartesian_az_dims=cartesian_az_dims,
            normalize_dims=normalize_dims,
            options_overwrite=options_overwrite,
        )

        dev = self.device
        self.x_mean           = torch.tensor(stats["x_mean"],           dtype=torch.float32).to(dev)
        self.x_std            = torch.tensor(stats["x_std"],            dtype=torch.float32).to(dev)
        self.alpha_min        = torch.tensor(stats["alpha_min"],        dtype=torch.float32).to(dev)
        self.alpha_max        = torch.tensor(stats["alpha_max"],        dtype=torch.float32).to(dev)
        self.alpha_range      = (self.alpha_max - self.alpha_min).clamp(min=1e-8).to(dev)
        self.alpha_logit_mean = torch.tensor(stats["alpha_logit_mean"], dtype=torch.float32).to(dev)
        self.alpha_logit_std  = torch.tensor(stats["alpha_logit_std"],  dtype=torch.float32).to(dev)
        self._alpha_scale     = (1.0 / (0.25 * self.alpha_range * self.alpha_logit_std)).to(dev)

        self.flow.load_state_dict(torch.load(path / "flow.pt", map_location=dev))
        self.flow = self.flow.to(dev)

        self.x_train     = None
        self.alpha_train = None

        if (path / "losses.npy").exists():
            self.losses = np.load(path / "losses.npy", allow_pickle=True).tolist()

        return self