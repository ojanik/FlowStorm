import jax.random as jr
import jax.numpy as jnp

def calc_acceptance(mu,sigma):
    p = abs(mu**2 - jnp.sqrt(sigma)) / 100
    assert jnp.all(p < 1.) and jnp.all(p > 0.)
    return p

def generate_discrete_dataset(key, mu, sigma, n=1):
    key, k1 = jr.split(key)
    x = mu + sigma * jr.normal(k1, (n,))
    key, k1 = jr.split(key)
    p = calc_acceptance(mu,sigma)
    mask = jr.bernoulli(k1, p,shape=x.shape)

    return x[:, None][mask]


def generate_snowstorm_dataset(key, n):
    # draw mus

    key, k1 = jr.split(key)
    mus = 10*jr.uniform(k1, (n,)) - 5.
    key, k1 = jr.split(key)
    sigmas = 5 * jr.uniform(k1, (n,)) + 0.01
    key, k1 = jr.split(key)
    x = mus + sigmas * jr.normal(k1, (n,))
    p = calc_acceptance(mus, sigmas)
    key, k1 = jr.split(key)
    mask = jr.bernoulli(k1, p,shape=x.shape)
    return x[:, None][mask], jnp.stack([mus,sigmas],axis=-1)[mask]
