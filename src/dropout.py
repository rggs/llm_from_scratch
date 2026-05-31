import jax
import jax.numpy as jnp
import sys



def dropout(arr, P=0.1, key=jax.random.key(42)):
    # First generate a uniform random matrix with the same shape as arr
    mask = jax.random.uniform(key, arr.shape)
    # Now, use P as a dropout threshold
    mask = mask.at[mask>P].set(1)
    mask = mask.at[mask<=P].set(0)

    # Now do element-wise mul between arr and mask
    return arr * mask



if __name__ == "__main__":

    keyint = sys.argv[1] if len(sys.argv)>1 else 42
    KEY = jax.random.key(int(keyint))
    arr = jax.random.uniform(KEY, (10,10))
    arr = dropout(arr, key=KEY)
    print(arr)
