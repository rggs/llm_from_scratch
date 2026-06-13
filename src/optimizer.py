import jax
import jax.numpy as jnp



def AdamWOptim(params, grads, m, v, t, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, _lambda=0.01):
    m = jax.tree.map(lambda g, m: beta_1*m + (1-beta_1)*g, grads, m)
    v = jax.tree.map(lambda v, g: beta_2*v + (1-beta_2)*(g**2), v, grads)
    params = jax.tree.map(lambda p, m, v: p - ((lr*(m/(1-beta_1**t))/(jnp.sqrt(v/(1-beta_2**t))+epsilon)) + _lambda*lr*p), params, m, v)

    return params, m, v




def createAdamW(params):
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    
    return m, v
