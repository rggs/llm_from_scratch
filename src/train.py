import jax
import jax.numpy as jnp


from src.feed_forward import linear_forward, create_ff, ff_forward, relu, softmax
from src.mha import _gen_multi_head_attention, _forward_mha
from src.dropout import dropout
from src.embedding import create_embedding_layer, gen_pos_encodings
from src.optimizer import createAdamW, AdamWOptim
from src.model import create_llm, model_forward


def calculate_batch_loss(model, x, y):
    # Create shifted input and target sequences
    x_s = jnp.zeros((y.shape[0], y.shape[1]+1, dtype=int))
    x_s = x_s.at[:,1:].add(y)
    # 2 = <bos>
    x_s = x_s.at[:,0].add(2)

    _y = jnp.zeros((y.shape[0], y.shape[1]+1, dtype=int))
    y = _y.at[:, :-1].add(y)
    # 3 = <eos>
    y = y.at[:, -1].add(3)

    logits = model_forward(model_forward, x, x_s)
 

    log_probs = softmax(logits)
    log_probs = jnp.take_along_axis(logits, y[..., None], axis=-1).squeeze(-1)
    # find where the padding tokens are
    mask = (y != 0)
    loss = -jnp.sum(log_probs * mask) / jnp.sum(mask)
    return loss



def process_batch(model, optimizer, x, y):







if __name__ == "__main__":
    key = jax.random.key(42)
    model = create_llm(2)
    test_input = jax.random.randint(key, (8, 800), 0, 37000)
    ouput = model_forward(model, test_input)
