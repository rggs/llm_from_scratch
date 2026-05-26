import jax
import jax.numpy as jnp



def ScaledDotProductAttention(Q,K,V):
    kq = Q @ K.T
    kq = kq / jnp.sqrt(K.size)
    kq = softmax(kq)
    return kq @ V


def softmax(z):
    return jnp.exp(z) / jnp.sum(jnp.exp(z), axis=-1, keepdims=True)


if __name__ == "__main__":
    dim = 512
    K = jnp.ones((1, dim)) 
    Q = jnp.ones((1, dim))
    V = jnp.ones((1,dim))


    print(ScaledDotProductAttention(Q,K,V))
