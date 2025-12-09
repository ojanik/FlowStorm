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