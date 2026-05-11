import jax
import jax.numpy as jnp
import jax.random as jr

from flowjax.distributions import Normal, Transformed
from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_data
from flowjax.train.losses import MaximumLikelihoodLoss, WeightedMaximumLikelihoodLoss
from flowjax.bijections import RationalQuadraticSpline

from paramax import non_trainable
import flowjax.bijections as bij

import json
import numpy as np
from pathlib import Path
import equinox as eqx


class FlowSurface:
    """
    Models:
      - conditional flow: p(x | alpha)   (self.flow)
      - yield flow:       p(alpha)       (self.yield_flow)

    Preprocessing:
      - x:     standardization (mean/std) -> N(0,1)-like, no bijector in flow
      - alpha: min/max normalize to [eps, 1-eps] + Sigmoid bijector in flow

    RQS intervals:
      - x flow:     interval=5  (covers +-5 sigma of standardized x)
      - yield flow: interval computed from logit(eps) to cover full alpha range
    """

    def __init__(
        self,
        x,
        alpha,
        weights=None,
        seed=187,
        flow_kwargs=None,
        yield_kwargs=None,
        eps=1e-3,
    ):
        rng = jr.key(seed)

        flow_defaults  = dict(nn_depth=4, flow_layers=16)
        yield_defaults = dict(nn_depth=2, flow_layers=8)

        flow_kwargs  = {**flow_defaults,  **(flow_kwargs  or {})}
        yield_kwargs = {**yield_defaults, **(yield_kwargs or {})}

        x     = jnp.asarray(x)
        alpha = jnp.asarray(alpha)

        x_dim     = int(x.shape[1])
        alpha_dim = int(alpha.shape[1])

        self._x_dim        = x_dim
        self._alpha_dim    = alpha_dim
        self._flow_kwargs  = dict(flow_kwargs)
        self._yield_kwargs = dict(yield_kwargs)
        self._eps          = float(eps)

        # -------------------------
        # x: standardization (mean / std)
        # -------------------------
        self.x_mean = jnp.mean(x, axis=0)
        self.x_std  = jnp.std(x,  axis=0)
        self.x_std  = jnp.where(self.x_std <= 0, 1.0, self.x_std)

        # -------------------------
        # alpha: min/max normalization -> [eps, 1-eps]
        # -------------------------
        self.alpha_min   = jnp.min(alpha, axis=0)
        self.alpha_max   = jnp.max(alpha, axis=0)
        self.alpha_range = jnp.where(
            (self.alpha_max - self.alpha_min) <= 0, 1.0,
            (self.alpha_max - self.alpha_min)
        )
        # constant scale factor for chain rule (alpha_norm wrt alpha_raw)
        self._alpha_scale = (1.0 - 2.0 * self._eps) / self.alpha_range  # (alpha_dim,)

        # -------------------------
        # Preprocessed training data
        # -------------------------
        self.x_train     = self.transform_x(x)
        self.alpha_train = self.transform_alpha(alpha)

        # -------------------------
        # RQS intervals
        # x is ~N(0,1) after standardization -> interval=5 covers +-5 sigma
        # alpha_train is in [eps, 1-eps]; the Sigmoid bijector applies logit internally,
        # so the RQS sees logit([eps, 1-eps]) -> interval must cover logit(1-eps)
        # -------------------------
        x_rqs_interval     = 5
        alpha_rqs_interval = int(np.ceil(abs(np.log(eps / (1.0 - eps))))) + 1
        self._x_rqs_interval     = x_rqs_interval
        self._alpha_rqs_interval = alpha_rqs_interval

        # -------------------------
        # Alpha support constraint: Sigmoid maps R -> (0,1)
        # No bijector on x (standardized x is already unbounded)
        # -------------------------
        self.alpha_constrained = bij.Stack(
            [bij.Sigmoid() for _ in range(self._alpha_dim)]
        )

        # -------------------------
        # Loss functions
        # -------------------------
        if weights is not None:
            self.flow_loss  = WeightedMaximumLikelihoodLoss()
            self.yield_loss = WeightedMaximumLikelihoodLoss()
        else:
            self.flow_loss  = MaximumLikelihoodLoss()
            self.yield_loss = MaximumLikelihoodLoss()

        self.weights = weights

        # -------------------------
        # Build flows
        # -------------------------
        rng, sub = jr.split(rng)
        self.flow = masked_autoregressive_flow(
            key=sub,
            base_dist=Normal(jnp.zeros(x_dim)),
            cond_dim=alpha_dim,
            transformer=RationalQuadraticSpline(knots=8, interval=x_rqs_interval),
            **flow_kwargs,
        )
        # No bijector wrapping: standardized x is unbounded, flow models it directly

        rng, sub = jr.split(rng)
        base_yield = masked_autoregressive_flow(
            key=sub,
            base_dist=Normal(jnp.zeros(alpha_dim)),
            transformer=RationalQuadraticSpline(knots=8, interval=alpha_rqs_interval),
            **yield_kwargs,
        )
        # Sigmoid bijector: flow models logit(alpha_train), Sigmoid maps back to (0,1)
        self.yield_flow = Transformed(base_yield, non_trainable(self.alpha_constrained))

        self.rng          = rng
        self.losses       = None
        self.yield_losses = None

    # -------------------------
    # x: standardization helpers
    # -------------------------
    def transform_x(self, x):
        """Raw x -> standardized  (mean=0, std=1 per dimension)"""
        x = jnp.asarray(x)
        return (x - self.x_mean) / self.x_std

    def retransform_x(self, x_t):
        """Standardized x -> raw x"""
        x_t = jnp.asarray(x_t)
        return x_t * self.x_std + self.x_mean

    # -------------------------
    # alpha: min/max normalization helpers
    # -------------------------
    def transform_alpha(self, alpha):
        """Raw alpha -> normalized to [eps, 1-eps]"""
        alpha = jnp.asarray(alpha)
        u = (alpha - self.alpha_min) / self.alpha_range
        return self._eps + (1.0 - 2.0 * self._eps) * u

    def retransform_alpha(self, a_t):
        """Normalized alpha -> raw alpha"""
        a_t = jnp.asarray(a_t)
        u = (a_t - self._eps) / (1.0 - 2.0 * self._eps)
        return self.alpha_min + u * self.alpha_range

    # -------------------------
    # Training
    # -------------------------
    def train_flow(self, learning_rate=3e-4, max_epochs=200, max_patience=10, batch_size=65536):
        rng, sub = jr.split(self.rng)

        if self.weights is not None:
            train_tuple = (self.x_train, self.weights, self.alpha_train)
        else:
            train_tuple = (self.x_train, self.alpha_train)

        flow, losses = fit_to_data(
            sub,
            self.flow,
            data=train_tuple,
            learning_rate=learning_rate,
            max_patience=max_patience,
            max_epochs=max_epochs,
            batch_size=batch_size,
            loss_fn=self.flow_loss,
        )
        self.flow   = flow
        self.losses = losses
        self.rng    = rng

    def train_yield(self, learning_rate=3e-4, max_epochs=200, max_patience=10, batch_size=65536):
        rng, sub = jr.split(self.rng)

        if self.weights is not None:
            train_tuple = (self.alpha_train, self.weights)
        else:
            train_tuple = self.alpha_train

        yield_flow, losses = fit_to_data(
            sub,
            self.yield_flow,
            data=train_tuple,
            learning_rate=learning_rate,
            max_patience=max_patience,
            max_epochs=max_epochs,
            batch_size=batch_size,
            loss_fn=self.yield_loss,
        )
        self.yield_flow   = yield_flow
        self.yield_losses = losses
        self.rng          = rng

    def train_both(
        self,
        flow_learning_rate=3e-4,
        yield_learning_rate=3e-4,
        max_epochs=200,
        max_patience=10,
        batch_size=65536,
        order=("yield", "flow"),
    ):
        for which in order:
            if which == "yield":
                self.train_yield(
                    learning_rate=yield_learning_rate,
                    max_epochs=max_epochs,
                    max_patience=max_patience,
                    batch_size=batch_size,
                )
            elif which == "flow":
                self.train_flow(
                    learning_rate=flow_learning_rate,
                    max_epochs=max_epochs,
                    max_patience=max_patience,
                    batch_size=batch_size,
                )
            else:
                raise ValueError(f"Unknown training stage: {which}")
            jax.clear_caches()

    # -------------------------
    # Log-prob helpers (preprocessed space)
    # -------------------------
    def _logp_x_single(self, x_t, a_t):
        return self.flow.log_prob(x_t[None, :], a_t[None, :])[0]

    def _logp_alpha_single(self, a_t):
        return self.yield_flow.log_prob(a_t[None, :])[0]

    def _logp_joint_single(self, x_t, a_t):
        return self._logp_x_single(x_t, a_t) + self._logp_alpha_single(a_t)

    # -------------------------
    # Weights
    # -------------------------
    def get_weights(
        self,
        x_base,
        base_alpha,
        goal_alpha,
        batch_size=131072,
        jit=True,
        include_yield=True,
    ):
        x_t = self.transform_x(x_base)

        base_alpha = jnp.atleast_1d(base_alpha)
        goal_alpha = jnp.atleast_1d(goal_alpha)
        base_a_t   = self.transform_alpha(base_alpha)
        goal_a_t   = self.transform_alpha(goal_alpha)
        D          = base_a_t.shape[0]

        if include_yield:
            logpy_goal  = self.yield_flow.log_prob(goal_a_t[None, :])[0]
            logpy_base  = self.yield_flow.log_prob(base_a_t[None, :])[0]
            logpy_ratio = logpy_goal - logpy_base
        else:
            logpy_ratio = 0.0

        def weights_batch(xb):
            B    = xb.shape[0]
            base = jnp.broadcast_to(base_a_t, (B, D))
            goal = jnp.broadcast_to(goal_a_t, (B, D))
            logp_goal = self.flow.log_prob(xb, goal)
            logp_base = self.flow.log_prob(xb, base)
            return jnp.exp((logp_goal - logp_base) + logpy_ratio)

        if jit:
            weights_batch = jax.jit(weights_batch)

        outs = []
        N    = x_t.shape[0]
        for i in range(0, N, batch_size):
            outs.append(weights_batch(x_t[i : i + batch_size]))
        return jnp.concatenate(outs, axis=0)

    # -------------------------
    # Gradients / Hessians wrt RAW alpha (chain rule via constant _alpha_scale)
    # -------------------------
    def get_gradients(self, x_base, base_alpha, batch_size=8192, jit=True, include_yield=True):
        x_t        = self.transform_x(x_base)
        base_alpha = jnp.atleast_1d(base_alpha)
        base_a_t   = self.transform_alpha(base_alpha)
        scale      = self._alpha_scale  # (alpha_dim,)

        if include_yield:
            grad_fn = jax.grad(lambda a_t, x_one: self._logp_joint_single(x_one, a_t))
        else:
            grad_fn = jax.grad(lambda a_t, x_one: self._logp_x_single(x_one, a_t))

        def grads_batch(xb):
            g = jax.vmap(lambda x_one: grad_fn(base_a_t, x_one))(xb)  # (B, alpha_dim)
            return g * scale[None, :]  # chain rule -> wrt raw alpha

        if jit:
            grads_batch = jax.jit(grads_batch)

        outs = []
        N    = x_t.shape[0]
        for i in range(0, N, batch_size):
            outs.append(grads_batch(x_t[i : i + batch_size]))
        return jnp.concatenate(outs, axis=0)

    def get_hessians(self, x_base, base_alpha, batch_size=2048, jit=True, include_yield=True):
        x_t        = self.transform_x(x_base)
        base_alpha = jnp.atleast_1d(base_alpha)
        base_a_t   = self.transform_alpha(base_alpha)
        scale      = self._alpha_scale  # (alpha_dim,)

        if include_yield:
            hess_fn = jax.hessian(lambda a_t, x_one: self._logp_joint_single(x_one, a_t))
        else:
            hess_fn = jax.hessian(lambda a_t, x_one: self._logp_x_single(x_one, a_t))

        def hess_batch(xb):
            H = jax.vmap(lambda x_one: hess_fn(base_a_t, x_one))(xb)  # (B, D, D)
            return H * scale[None, :, None] * scale[None, None, :]

        if jit:
            hess_batch = jax.jit(hess_batch)

        outs = []
        N    = x_t.shape[0]
        for i in range(0, N, batch_size):
            outs.append(hess_batch(x_t[i : i + batch_size]))
        return jnp.concatenate(outs, axis=0)

    def get_weights_taylor(
        self,
        x_base,
        base_alpha,
        goal_alpha,
        order=1,
        grads=None,
        hessians=None,
        clip_logw=50.0,
        include_yield=True,
    ):
        base_alpha = jnp.atleast_1d(base_alpha)
        goal_alpha = jnp.atleast_1d(goal_alpha)
        d          = goal_alpha - base_alpha

        if grads is None:
            grads = self.get_gradients(x_base, base_alpha, include_yield=include_yield)

        logw = grads @ d

        if order >= 2:
            if hessians is None:
                hessians = self.get_hessians(x_base, base_alpha, include_yield=include_yield)
            logw = logw + 0.5 * jnp.einsum("i,nij,j->n", d, hessians, d)

        return jnp.exp(jnp.clip(logw, -clip_logw, clip_logw))

    # -------------------------
    # Sampling helpers
    # -------------------------
    def sample(self, key, n, alpha_raw):
        """
        Sample x ~ p(x | alpha_raw), returned in original x space.

        Parameters
        ----------
        key       : JAX PRNGKey
        n         : int
        alpha_raw : array (alpha_dim,)

        Returns
        -------
        x_raw : jnp.array (n, x_dim)
        """
        alpha_raw  = jnp.atleast_1d(alpha_raw)
        a_t        = self.transform_alpha(alpha_raw)
        a_t_batch  = jnp.broadcast_to(a_t, (n, self._alpha_dim))
        x_t        = self.flow.sample(key, (), a_t_batch)   # (n, x_dim)
        return self.retransform_x(x_t)

    def sample_with_alpha(self, key, n):
        """
        Draw (x, alpha) jointly from the full model p(x, alpha) = p(x|alpha) p(alpha).
        Both returned in original (raw) space.

        Returns
        -------
        x_raw     : jnp.array (n, x_dim)
        alpha_raw : jnp.array (n, alpha_dim)
        """
        key_a, key_x = jr.split(key)
        a_t = self.yield_flow.sample(key_a, (n,))       # (n, alpha_dim) normalized
        x_t = self.flow.sample(key_x, (), a_t)          # (n, x_dim) standardized
        return self.retransform_x(x_t), self.retransform_alpha(a_t)

    # -------------------------
    # Save / Load
    # -------------------------
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta = dict(
            x_dim=self._x_dim,
            alpha_dim=self._alpha_dim,
            flow_kwargs=self._flow_kwargs,
            yield_kwargs=self._yield_kwargs,
            eps=self._eps,
            x_rqs_interval=self._x_rqs_interval,
            alpha_rqs_interval=self._alpha_rqs_interval,
        )
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

        stats = dict(
            x_mean=jnp.asarray(self.x_mean).tolist(),
            x_std=jnp.asarray(self.x_std).tolist(),
            alpha_min=jnp.asarray(self.alpha_min).tolist(),
            alpha_max=jnp.asarray(self.alpha_max).tolist(),
        )
        (path / "stats.json").write_text(json.dumps(stats, indent=2))

        eqx.tree_serialise_leaves(path / "flow.eqx",       self.flow)
        eqx.tree_serialise_leaves(path / "yield_flow.eqx", self.yield_flow)

        if self.losses is not None:
            np.save(path / "losses.npy",       np.asarray(self.losses))
        if self.yield_losses is not None:
            np.save(path / "yield_losses.npy", np.asarray(self.yield_losses))

    @classmethod
    def load(cls, path, seed=187):
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())

        x_dim              = int(meta["x_dim"])
        alpha_dim          = int(meta["alpha_dim"])
        flow_kwargs        = dict(meta["flow_kwargs"])
        yield_kwargs       = dict(meta.get("yield_kwargs", {}))
        eps                = float(meta.get("eps", 1e-3))
        x_rqs_interval     = int(meta.get("x_rqs_interval", 5))
        alpha_rqs_interval = int(meta.get("alpha_rqs_interval",
                                          int(np.ceil(abs(np.log(eps / (1.0 - eps))))) + 1))

        stats     = json.loads((path / "stats.json").read_text())
        x_mean    = jnp.array(stats["x_mean"])
        x_std     = jnp.array(stats["x_std"])
        alpha_min = jnp.array(stats["alpha_min"])
        alpha_max = jnp.array(stats["alpha_max"])

        rng = jr.key(seed)

        # --- conditional flow (no bijector wrapper) ---
        rng, sub      = jr.split(rng)
        flow_template = masked_autoregressive_flow(
            key=sub,
            base_dist=Normal(jnp.zeros(x_dim)),
            cond_dim=alpha_dim,
            transformer=RationalQuadraticSpline(knots=8, interval=x_rqs_interval),
            **flow_kwargs,
        )
        flow = eqx.tree_deserialise_leaves(path / "flow.eqx", flow_template)

        # --- yield flow (Sigmoid bijector wrapper) ---
        rng, sub           = jr.split(rng)
        alpha_constrained  = bij.Stack([bij.Sigmoid() for _ in range(alpha_dim)])
        yield_template_base = masked_autoregressive_flow(
            key=sub,
            base_dist=Normal(jnp.zeros(alpha_dim)),
            transformer=RationalQuadraticSpline(knots=8, interval=alpha_rqs_interval),
            **yield_kwargs,
        )
        yield_template = Transformed(yield_template_base, non_trainable(alpha_constrained))
        yield_flow     = eqx.tree_deserialise_leaves(path / "yield_flow.eqx", yield_template)

        self = cls.__new__(cls)
        self.flow              = flow
        self.yield_flow        = yield_flow
        self.alpha_constrained = alpha_constrained
        self.rng               = rng

        self.losses       = None
        self.yield_losses = None
        self.x_train      = None
        self.alpha_train  = None

        self._x_dim              = x_dim
        self._alpha_dim          = alpha_dim
        self._flow_kwargs        = flow_kwargs
        self._yield_kwargs       = yield_kwargs
        self._eps                = eps
        self._x_rqs_interval     = x_rqs_interval
        self._alpha_rqs_interval = alpha_rqs_interval

        self.x_mean  = x_mean
        self.x_std   = x_std

        self.alpha_min   = alpha_min
        self.alpha_max   = alpha_max
        self.alpha_range = jnp.where(
            (alpha_max - alpha_min) <= 0, 1.0, (alpha_max - alpha_min)
        )
        self._alpha_scale = (1.0 - 2.0 * eps) / self.alpha_range

        losses_path = path / "losses.npy"
        if losses_path.exists():
            self.losses = np.load(losses_path, allow_pickle=True)

        ylosses_path = path / "yield_losses.npy"
        if ylosses_path.exists():
            self.yield_losses = np.load(ylosses_path, allow_pickle=True)

        return self