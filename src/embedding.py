import jax
import jax.numpy as jnp
from src.feed_forward import create_lin_layer, linear_forward


# Yeah, this is basically just re-doing the linear layer code from eariler,
# But I want to keep naming consistent
def create_embedding_layer(vocab_size=37_000, model_size=512):
    W, b = create_lin_layer(vocab_size, model_size)
    del b
    return W



# Now we can tackle the positional encodings
@jax.jit
def _generate_pos_encoding(pos, dim, size=10_000, dmodel=512):
    i = dim//2
    return (((dim%2)+1)*jnp.sin(pos / (size**(2*i/dmodel)))) + (((dim%2)+0) * jnp.cos(pos / (size**(2*i/dmodel))))

def gen_pos_encodings(seq_length, dmodel=512):
    position_indices = jnp.arange(seq_length)[:,None]
    dim_indices = jnp.arange(dmodel)[None, :]
    pos_encoding_matrix = _generate_pos_encoding(position_indices, dim_indices, dmodel=dmodel)
    return pos_encoding_matrix



if __name__=="__main__":
    key = jax.random.key(42)
    # Random input with 1 row, 100 tokens
    random_input = jax.random.randint(key, (1, 100), 0, 37000)
    embed = create_embedding_layer()
    embedded_input = embed[random_input]
    assert embedded_input.shape == (1, 100, 512), embedded_input.shape
    pos_encod = gen_pos_encodings(100)

    output = embedded_input + pos_encod
    print(output)

