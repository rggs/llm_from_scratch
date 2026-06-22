import jax
import jax.numpy as jnp
import sys



def dropout(arr, key, P=0.1, train=True):
    if not train or P==0.0:
        return x
    
    prob = 1-P
    # First generate a uniform random matrix with the same shape as arr
    mask = jax.random.bernoulli(key, prob, arr.shape)

    # Now do element-wise mul between arr and mask
    return arr * mask / prob



if __name__ == "__main__":

    keyint = sys.argv[1] if len(sys.argv)>1 else 42
    KEY = jax.random.key(int(keyint))
    arr = jax.random.uniform(KEY, (10,10))
    arr = dropout(arr, key=KEY)
    print(arr)
