import jax
import jax.numpy as jnp



def ScaledDotProductAttention(Q,K,V, mask=None):
    kq = Q @ jnp.swapaxes(K, -1, -2)
    if mask is not None:
        kq = kq * mask
    kq = kq / jnp.sqrt(K.size)
    kq = softmax(kq)
    return kq @ V


def softmax(z):
    return jnp.exp(z) / jnp.sum(jnp.exp(z), axis=-1, keepdims=True)

def log_softmax(z):
    z = z - jnp.max(z, axis=-1, keepdims=True)
    _z = jnp.log(jnp.sum(jnp.exp(z), axis=-1, keepdims=True))
    return z - _z


if __name__ == "__main__":
    dim = 512
    K = jnp.ones((100, dim)) 
    Q = jnp.ones((100, dim))
    V = jnp.ones((100,dim))


    print(ScaledDotProductAttention(Q,K,V).shape)
