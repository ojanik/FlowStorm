import jax.numpy as jnp
import jax.random as jr

import numpy as onp
import matplotlib.pyplot as plt

from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow, block_neural_autoregressive_flow
from flowjax.train import fit_to_data

class FlowSurface():
    def __init__(self, x, alpha, seed = 187):
        self.rng = jr.key(seed)

        self.x_train = x
        self.alpha_train = alpha

        self._set_flow()

    def _set_flow(self):
        self.rng, subkey = jr.split(self.rng)
        flow = block_neural_autoregressive_flow(
            key=subkey,
            base_dist=Normal(jnp.zeros(self.x_train.shape[1])),
            cond_dim=self.alpha_train.shape[1],
        )
        self.flow = flow

    def train_flow(self):
        self.rng, subkey = jr.split(self.rng)
        self.flow, self.losses = fit_to_data(
            subkey,
            self.flow,
            data=(self.x_train, self.alpha_train),
            learning_rate=5e-2,
            max_patience=10,
        )

    def get_weights(self,x_base,base_alpha,goal_alpha):
        assert len(base_alpha) == len(goal_alpha), "Base and goal alphas must have the same length"
        N = len(x_base)
        base_alphas = jnp.broadcast_to(base_alpha, (N, base_alpha.shape[0]))
        goal_alphas = jnp.broadcast_to(goal_alpha, (N, goal_alpha.shape[0]))

        prob_goal = self.flow.log_prob(x_base,goal_alphas)
        prob_base = self.flow.log_prob(x_base,base_alphas)

        reweight = (jnp.exp(prob_goal)) / (jnp.exp(prob_base))

        return reweight

    