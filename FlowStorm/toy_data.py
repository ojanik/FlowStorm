import jax.random as jr
import jax.numpy as jnp


def generate_discrete_dataset(key, mu, sigma, n):
    key, k1 = jr.split(key)
    x = mu + sigma * jr.normal(k1, (n,))

    return x[:, None]

def generate_snowstorm_dataset(key, mu, sigma_mu, sigma, n):
    # draw mus
    key, k1 = jr.split(key)
    #mus = mu + sigma_mu * jr.normal(k1, (n,))
    mus = jr.uniform(k1, (n,)) - .5
    key, k1 = jr.split(key)
    sigmas = jr.uniform(k1, (n,)) + 0.1
    key, k1 = jr.split(key)
    x = mus + sigmas * jr.normal(k1, (n,))
    return x[:, None], jnp.stack([mus,sigmas],axis=-1) 
