import jax
import jax.numpy as jnp
import pickle

from src.feed_forward import linear_forward, create_ff, ff_forward, relu, softmax
from src.mha import _gen_multi_head_attention, _forward_mha
from src.dropout import dropout
from src.embedding import create_embedding_layer, gen_pos_encodings
from src.optimizer import createAdamW, AdamWOptim
from src.model import create_llm, model_forward
from src.dataloader import WMTDataLoader
from torch.utils.tensorboard import SummaryWriter
from src.scaled_attention import log_softmax
from tqdm import tqdm


writer = SummaryWriter()

def clip_grads(g, maxnorm=1.):
    norm = jnp.sqrt(sum(jnp.sum(_g**2) for _g in jax.tree.leaves(g)))
    scale_factor = jnp.minimum(1.0, maxnorm / (norm + 1e-6))
    return jax.tree.map(lambda g: g * scale_factor, g)

def calculate_batch_loss(model, x, y, key, train=True):
    # Create shifted input and target sequences
    x_s = jnp.zeros((y.shape[0], y.shape[1]+1), dtype=int)
    x_s = x_s.at[:,1:].add(y)
    # 2 = <bos>
    x_s = x_s.at[:,0].add(2)

    _y = jnp.zeros((y.shape[0], y.shape[1]+1), dtype=int)
    y = _y.at[:, :-1].add(y)
    # 3 = <eos>
    y = y.at[:, -1].add(3)

    logits = model_forward(model, x, x_s, key, training=train)
 

    log_probs = log_softmax(logits)
    log_probs = jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    # find where the padding tokens are
    mask = (y != 0)
    loss = -jnp.sum(log_probs * mask) / jnp.sum(mask)
    
    # Also get preds so we can check accuracy
    if not train:
        preds = jnp.argmax(logits, axis=-1)
        correct = (preds == y)
        acc = jnp.sum(correct * mask) / jnp.sum(mask)

    if train:
        return loss
    else:
        return loss, acc


def test(model, dataloader, key, epoch=0, include_print=True):
    dataloader.reset()
    steps = 0
    total_loss = 0
    total_accuracy = 0
    for en, de in dataloader.get_batch():
        loss, acc = calculate_batch_loss(model, en, de, key, train=False)
        total_loss += loss.item()
        total_accuracy += acc.item()
        steps += 1

    writer.add_scalar("Loss/test",total_loss / steps, epoch)
    writer.add_scalar("Accruacy/test",total_accuracy / steps, epoch)
    if include_print:
        print(f"Epoch {epoch} test loss: {total_loss / steps}")
        print(f"Epoch {epoch} test accuracy: {total_accuracy / steps}")


@jax.jit
def train_batch(model, en, de, m, v, timesteps, key, lr=1e-5):
    loss, grads = jax.value_and_grad(calculate_batch_loss, argnums=0)(model, en, de, key)
    grads = clip_grads(grads, maxnorm=1.0)
    model, m, v = AdamWOptim(model, grads, m, v, timesteps, lr=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    return model, m, v, loss


def train(model, train_dataloader, test_dataloader, key, epochs=10, include_print=False, dmodel=512, warmup=4000, batch_size=32):
    m, v = createAdamW(model)
    timesteps = 1
    _lr = lambda t: dmodel ** (-0.5) * min(t**(-0.5), t * warmup**(-1.5))
    for e in range(epochs):
        print(f"Epoch {e+1}")
        train_dataloader.reset()
        # For testing
        for en, de in tqdm(train_dataloader.get_batch(), desc=f"Epoch {e+1}", total = (len(train_dataloader)//batch_size)):
            # en, de = next(train_dataloader.get_batch())
            # while timesteps < 1e5:
            lr = _lr(timesteps)
            key, subkey = jax.random.split(key)
            model, m, v, loss = train_batch(model, en, de, m, v, timesteps, subkey)
            timesteps += 1
            writer.add_scalar("Loss/train",loss.item(), timesteps)
            if include_print:
                print(f"Loss: {loss}")

        # test(model, test_dataloader, key, epoch=epoch)
    
    model = jax.device_get(model)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)









if __name__ == "__main__":
    STACKS=2
    model = create_llm(STACKS)
    # Create a mask to delineate which params are learnable, and which are not (like activations)
    key = jax.random.key(42)
    batch_size=8
    train_dataloader = WMTDataLoader("/home/rgswope/workspace/llm_from_scratch/data/wmt_2017_en_de/wmt14_translate_de-en_train_tok.csv", batch_size=batch_size)
    test_dataloader = WMTDataLoader("/home/rgswope/workspace/llm_from_scratch/data/wmt_2017_en_de/wmt14_translate_de-en_test_tok.csv", batch_size=batch_size)
    train(model, train_dataloader, test_dataloader, key, include_print=False, batch_size=batch_size)

