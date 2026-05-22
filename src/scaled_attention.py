import jax
import jax.numpy as jnp



def ScaledDotProductAttention(K,Q,V):
    kq = Q @ K.T
    kq = kq / jnp.sqrt(K.size)
    kq = softmax(kq)
    return kq @ V


def softmax(z):
    return jnp.exp(z) / jnp.sum(jnp.exp(z), axis=-1, keepdims=True)


if __name__ == "__main__":
    dim = 512
    K = jnp.ones((dim, 1)) 
    Q = jnp.ones((dim, 1))
    V = jnp.ones((dim, 1))


    print(ScaledDotProductAttention(K,Q,V))
