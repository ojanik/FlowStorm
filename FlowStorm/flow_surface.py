import jax
import jax.numpy as jnp
import jax.random as jr

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.train import fit_to_data


class FlowSurface:
    def __init__(self, x, alpha, seed=187):
        """
        x:      (N, x_dim)      event observables
        alpha:  (N, alpha_dim)  detector parameters
        """
        rng = jr.key(seed)

        # build conditional flow
        rng, sub = jr.split(rng)
        flow = block_neural_autoregressive_flow(
            key=sub,
            base_dist=Normal(jnp.zeros(x.shape[1])),   # x_dim
            cond_dim=alpha.shape[1],                   # alpha_dim
        )

        self.flow = flow
        self.rng = rng
        self.x_train = x
        self.alpha_train = alpha
        self.losses = None

    def train_flow(self, learning_rate=5e-2, max_patience=10):
        rng, sub = jr.split(self.rng)
        flow, losses = fit_to_data(
            sub,
            self.flow,
            data=(self.x_train, self.alpha_train),   # (x, cond)
            learning_rate=learning_rate,
            max_patience=max_patience,
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