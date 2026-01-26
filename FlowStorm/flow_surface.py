import jax
from jax import jit
import jax.numpy as jnp
import jax.random as jr

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.train import fit_to_data

import os
import json
from pathlib import Path

import equinox as eqx


class FlowSurface:
    def __init__(self, x, alpha, seed=187, **flow_kwargs):
        rng = jr.key(seed)

        default_flow_kwargs = dict(
            nn_depth=1,
            nn_block_dim=4,
            flow_layers=1,
        )
        flow_kwargs = {**default_flow_kwargs, **flow_kwargs}

        rng, sub = jr.split(rng)
        flow = block_neural_autoregressive_flow(
            key=sub,
            base_dist=Normal(jnp.zeros(x.shape[1])),
            cond_dim=alpha.shape[1],
            **flow_kwargs,
        )

        self.flow = flow
        self.rng = rng
        self.x_train = x
        self.alpha_train = alpha
        self.losses = None

        # store for save/load
        self._x_dim = int(x.shape[1])
        self._alpha_dim = int(alpha.shape[1])
        self._flow_kwargs = dict(flow_kwargs)

    def train_flow(self, learning_rate=5e-2, max_epochs=100,max_patience=10,batch_size=65536):
        rng, sub = jr.split(self.rng)
        flow, losses = fit_to_data(
            sub,
            self.flow,
            data=(self.x_train, self.alpha_train),   # (x, cond)
            learning_rate=learning_rate,
            max_patience=max_patience,
            max_epochs=max_epochs,
            batch_size=batch_size,
        )
        self.flow = flow
        self.losses = losses
        self.rng = rng

    def get_weights(self, x_base, base_alpha, goal_alpha):
        base_alpha = jnp.atleast_1d(base_alpha)
        goal_alpha = jnp.atleast_1d(goal_alpha)
        assert base_alpha.shape == goal_alpha.shape

        N = x_base.shape[0]
        base = jnp.broadcast_to(base_alpha, (N, base_alpha.shape[0]))
        goal = jnp.broadcast_to(goal_alpha, (N, goal_alpha.shape[0]))

        logp_goal = self.flow.log_prob(x_base, goal)
        logp_base = self.flow.log_prob(x_base, base)

        return jnp.exp(logp_goal - logp_base)
    
    # --- internal: scalar logp for one event ---
    def _logp_single(self, x, alpha):
        # x: (x_dim,), alpha: (alpha_dim,)
        return self.flow.log_prob(x[None, :], alpha[None, :])[0]

    def get_gradients(self, x_base, base_alpha, batch_size=8192, jit=True):
        """
        Per-event gradient wrt alpha of log p(x|alpha) at alpha=base_alpha.

        Returns
        -------
        grads : (N, alpha_dim)
        """
        base_alpha = jnp.atleast_1d(base_alpha)

        grad_fn = jax.grad(lambda a, x: self._logp_single(x, a))

        def grads_batch(xb):
            return jax.vmap(lambda x: grad_fn(base_alpha, x))(xb)  # (B, alpha_dim)

        if jit:
            grads_batch = jax.jit(grads_batch)

        # run in batches to avoid huge compile / memory
        N = x_base.shape[0]
        outs = []
        for i in range(0, N, batch_size):
            outs.append(grads_batch(x_base[i:i+batch_size]))
        return jnp.concatenate(outs, axis=0)

    def get_hessians(self, x_base, base_alpha, batch_size=2048, jit=True):
        """
        Per-event Hessian wrt alpha of log p(x|alpha) at alpha=base_alpha.

        Returns
        -------
        H : (N, alpha_dim, alpha_dim)
        """
        base_alpha = jnp.atleast_1d(base_alpha)

        hess_fn = jax.hessian(lambda a, x: self._logp_single(x, a))

        def hess_batch(xb):
            return jax.vmap(lambda x: hess_fn(base_alpha, x))(xb)  # (B, D, D)

        if jit:
            hess_batch = jax.jit(hess_batch)

        N = x_base.shape[0]
        outs = []
        for i in range(0, N, batch_size):
            outs.append(hess_batch(x_base[i:i+batch_size]))
        return jnp.concatenate(outs, axis=0)

    def get_weights_taylor(self, x_base, base_alpha, goal_alpha, order=1,
                           grads=None, hessians=None, clip_logw=50.0):
        """
        Fast approximate weights using Taylor expansion in alpha around base_alpha.

        Parameters
        ----------
        order : 1 or 2
        grads : optionally precomputed grads from get_gradients
        hessians : optionally precomputed hessians from get_hessians
        """
        base_alpha = jnp.atleast_1d(base_alpha)
        goal_alpha = jnp.atleast_1d(goal_alpha)
        d = goal_alpha - base_alpha  # (D,)

        if grads is None:
            grads = self.get_gradients(x_base, base_alpha)

        logw = grads @ d

        if order >= 2:
            if hessians is None:
                hessians = self.get_hessians(x_base, base_alpha)
            logw = logw + 0.5 * jnp.einsum("i, nij, j->n", d, hessians, d)

        logw = jnp.clip(logw, -clip_logw, clip_logw)
        return jnp.exp(logw)
    

    # -------------------------
    # Save / Load
    # -------------------------
    def save(self, path):
        """Save flow + metadata to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta = dict(
            x_dim=self._x_dim,
            alpha_dim=self._alpha_dim,
            flow_kwargs=self._flow_kwargs,
        )
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

        # saves all array leaves of the pytree
        eqx.tree_serialise_leaves(path / "flow.eqx", self.flow)

        # optional: save losses if present
        if self.losses is not None:
            import numpy as np
            np.save(path / "losses.npy", np.asarray(self.losses))

    @classmethod
    def load(cls, path, seed=187):
        """Load flow + metadata from a directory. Training data is not restored."""
        path = Path(path)

        meta = json.loads((path / "meta.json").read_text())
        x_dim = int(meta["x_dim"])
        alpha_dim = int(meta["alpha_dim"])
        flow_kwargs = dict(meta["flow_kwargs"])

        # reconstruct the exact same flow structure
        rng = jr.key(seed)
        rng, sub = jr.split(rng)
        flow_template = block_neural_autoregressive_flow(
            key=sub,
            base_dist=Normal(jnp.zeros(x_dim)),
            cond_dim=alpha_dim,
            **flow_kwargs,
        )

        flow = eqx.tree_deserialise_leaves(path / "flow.eqx", flow_template)

        # build instance without calling __init__
        self = cls.__new__(cls)
        self.flow = flow
        self.rng = rng
        self.x_train = None
        self.alpha_train = None
        self.losses = None

        self._x_dim = x_dim
        self._alpha_dim = alpha_dim
        self._flow_kwargs = flow_kwargs

        # optional: restore losses if present
        losses_path = path / "losses.npy"
        if losses_path.exists():
            import numpy as np
            self.losses = np.load(losses_path,allow_pickle=True)

        return self