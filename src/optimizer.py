import jax
import jax.numpy as jnp



def AdamWOptim(params, grads, m, v, t, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, _lambda=0.01):
    # m_hat = m.copy()
    # v_hat = v.copy()
    # for params in param_dicts:
        # grads[pg][i] += _lambda * lr * _param
    # grads = jax.tree.map(lambda g, p: g + (_lambda * lr * p), grads, params)
    m = jax.tree.map(lambda g, m: beta_1*m + (1-beta_1)*g, grads, m)
    v = jax.tree.map(lambda v, g: beta_2*v + (1-beta_2)*(g**2), v, grads)
        # for pg in params:
        #    for i, _param in enumerate(params[pg]):
        #         m[pg][i] = beta_1*m[pg][i] + (1-beta_1)*grads[pg][i]
        #         v[pg][i] = beta_2*v[pg][i] + (1-beta_2)*grads[pg][i]
            # m_hat[_param] = m[_param]/(1-beta_1**t)
            # v_hat[_param] = v[_param]/(1-beta_2**t)
            # params[_param] = params[_param] - eta*((lr*(m[_param]/(1-beta_1**t))/(jnp.sqrt(v[_param]/(1-beta_2**t))+epsilon)) + _lambda*params[_param])
    params = jax.tree.map(lambda p, m, v: p - ((lr*(m/(1-beta_1**t))/(jnp.sqrt(v/(1-beta_2**t))+epsilon)) + _lambda*lr*p), params, m, v)

    # del m_hat
    # del v_hat
    return params, m, v




def createAdamW(params):
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    
    return m, v
