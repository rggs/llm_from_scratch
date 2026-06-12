import jax
import jax.numpy as jnp



def AdamW(params, grads, m, v, t, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, _lambda=0.01, eta=1):
    # m_hat = m.copy()
    # v_hat = v.copy()
    for _param in params:
        grads[_param] += _lambda * lr * params[_param]
        m[_param] = beta_1*m[_param] + (1-beta_1)*grads[_param]
        v[_param] = beta_2*v[_param] + (1-beta_2)*grads[_param]
        # m_hat[_param] = m[_param]/(1-beta_1**t)
        # v_hat[_param] = v[_param]/(1-beta_2**t)
        params[_param] = params[_param] - eta*((lr*(m[_param]/(1-beta_1**t))/(jnp.sqrt(v[_param]/(1-beta_2**t))+epsilon)) + _lambda*params[_param])

    del m_hat
    del v_hat

    return params, m, v 





