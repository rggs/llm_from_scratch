import jax
import jax.numpy as jnp
from src.scaled_attention import softmax
from jax.tree_util import register_pytree_node_class
from src.optimizer import AdamWOptim, createAdamW





# JAX is new to me, so building a feed-forward net is a good place to start
# It'll also obviously get used in the transformer later on. 
# The goal will be to construct a feed forward net with dense linear layers and
# activations. We already wrote softmax for attention, so we can just import and use that


@jax.jit
def linear_forward(W, b, x):
    x = x @ W 
    x = x + b 
    return x

def create_lin_layer(insize, outsize, key=None):
    if key is None:
        key = jax.random.PRNGKey(42)

    W = jax.random.normal(key, shape=(insize, outsize)) * jnp.sqrt(2 / insize)
    b = jax.random.normal(key, shape=(outsize))
    return W, b


def create_ff(sizes, activations, key=None):
    # Either no activations, or have something in the activations list for each layer 
    assert len(activations)==len(sizes) - 1 or len(activations)==0
    layers = {}
    lin_layers = {}
    _activations = {}
    for _ in range(len(sizes)-1):
        lin_layers[f"layer_{_}"] = create_lin_layer(sizes[_], sizes[_+1], key=key)
        
    for _ in range(len(activations)):
        _activations[f"act_{_}"] = activations[_]

    return lin_layers #, _activations



def ff_forward(layers, x):
    for i in range(len(layers)-1):
        l = layers[f"layer_{i}"]
        x = linear_forward(l[0], l[1], x)
        x = relu(x)
        # if len(activations) > 0:
        #     a = activations[f"act_{i}"]
        #     if a is not None:
        #         if a == "relu":
        #             x = relu(x)
    l = layers[f"layer_{len(layers)-1}"]
    return linear_forward(l[0], l[1], x)

@jax.jit
def relu(x):
    return jnp.maximum(x, 0)


#JAX one-hot encoding function
def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


# To make sure this actually trains, we'll use MNIST following the steps in the JAX MNIST tutorial
if __name__ == "__main__":
    import tensorflow_datasets as tfds
    data_dir = '/tmp/tfds'
    batch_size = 32
    ds = tfds.load(name='mnist', split='train', as_supervised=True, data_dir=data_dir)
    ds = ds.batch(batch_size).prefetch(1)
    train_set = tfds.as_numpy(ds)

    layers, activations = create_ff([28*28,256,256,10], ["relu", "relu", "softmax"])
    m, v, = createAdamW(layers)

    lr = 1e-5
    epochs = 20
    def mse_loss(layers, activations, x, y):
        preds = ff_forward(layers, activations, x)
        loss = jnp.sum((preds-y)**2)/x.shape[0]
        return loss

    timesteps = 1

    for e in range(epochs):
        print(f"Epoch {e}")
        epoch_acc = []
        for x,y in train_set:
            x = x.reshape(-1, 28*28)
            y = one_hot(y, 10)

            preds = ff_forward(layers, activations, x)

            output = jnp.argmax(preds, axis=-1)

            accuracy = 100 * jnp.mean(output == jnp.argmax(y, axis=-1))
            epoch_acc.append(accuracy)
            
            grads = jax.grad(mse_loss, argnums=0)(layers, activations, x, y)

            # Update our params
            # layers = jax.tree.map(lambda p, g: p - lr * g, layers, grads) 
            layers, m, v = AdamWOptim(layers, grads, m, v, timesteps, lr=lr)
            timesteps += 1
        acc = jnp.mean(jnp.array(epoch_acc)).item()
        print(f"Accuracy: {acc}")





