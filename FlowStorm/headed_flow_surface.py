import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.train import fit_to_data


# ----------------- Yield network (Equinox) ----------------- #

class YieldHead(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, in_dim: int, width: int = 32, depth: int = 2, *, key):
        self.mlp = eqx.nn.MLP(
            in_size=in_dim,
            out_size=1,
            width_size=width,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, alpha):
        # alpha: (N, in_dim) or (in_dim,)
        alpha = jnp.atleast_2d(alpha)
        out = self.mlp(alpha)       # (N, 1)
        return out.squeeze(-1)      # (N,)


# ----------------- FlowSurface: shape + yield ----------------- #

class FlowSurface(eqx.Module):
    flow: object          # FlowJAX flow
    yield_head: YieldHead
    rng: jax.Array

    # training data (optional to store)
    x_train: jax.Array
    alpha_train: jax.Array

    def __init__(self, x, alpha, seed: int = 187):
        """
        x:      (N, x_dim)       observables
        alpha:  (N, alpha_dim)   detector parameters
        """
        rng = jr.key(seed)

        # flow
        rng, sub = jr.split(rng)
        flow = block_neural_autoregressive_flow(
            key=sub,
            base_dist=Normal(jnp.zeros(x.shape[1])),
            cond_dim=alpha.shape[1],
        )

        # yield head
        rng, sub = jr.split(rng)
        yh = YieldHead(in_dim=alpha.shape[1], key=sub)

        self.flow = flow
        self.yield_head = yh
        self.rng = rng
        self.x_train = x
        self.alpha_train = alpha

    # --------- FLOW TRAINING (shape only) --------- #

    def train_flow(self, learning_rate: float = 5e-2, max_patience: int = 10):
        rng, sub = jr.split(self.rng)
        flow, losses = fit_to_data(
            sub,
            self.flow,
            data=(self.x_train, self.alpha_train),  # (x, cond)
            learning_rate=learning_rate,
            max_patience=max_patience,
        )
        self.flow = flow
        self.rng = rng
        self.losses = losses

    # --------- SHAPE WEIGHTS: p(x|goal)/p(x|base) --------- #

    def get_shape_weights(self, x_base, base_alpha, goal_alpha):
        base_alpha = jnp.atleast_1d(base_alpha)
        goal_alpha = jnp.atleast_1d(goal_alpha)
        assert base_alpha.shape == goal_alpha.shape

        N = x_base.shape[0]
        base = jnp.broadcast_to(base_alpha, (N, base_alpha.shape[0]))
        goal = jnp.broadcast_to(goal_alpha, (N, base_alpha.shape[0]))

        logp_goal = self.flow.log_prob(x_base, goal)
        logp_base = self.flow.log_prob(x_base, base)

        return jnp.exp(logp_goal - logp_base)

    # --------- YIELD WEIGHTS: N(goal)/N(base) --------- #

    def get_yield_weights(self, base_alpha, goal_alpha):
        logN_base = self.yield_head(base_alpha)  # scalar or (batch,)
        logN_goal = self.yield_head(goal_alpha)
        return jnp.exp(logN_goal - logN_base)

    # --------- TOTAL WEIGHTS: shape * yield --------- #

    def get_total_weights(self, x_base, base_alpha, goal_alpha):
        w_shape = self.get_shape_weights(x_base, base_alpha, goal_alpha)
        w_yield = self.get_yield_weights(base_alpha, goal_alpha)
        return w_shape * w_yield