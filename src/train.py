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
from functools import partial
import os
# import mlflow



def clip_grads(g, maxnorm=1.):
    norm = jnp.sqrt(sum(jnp.sum(_g**2) for _g in jax.tree.leaves(g)))
    scale_factor = jnp.minimum(1.0, maxnorm / (norm + 1e-6))
    return jax.tree.map(lambda g: g * scale_factor, g)

def calculate_train_batch_loss(model, x, y, key, stacks=2):
    # Create shifted input and target sequences
    x_s = jnp.zeros((y.shape[0], y.shape[1]+1), dtype=int)
    x_s = x_s.at[:,1:].add(y)
    # 2 = <bos>
    x_s = x_s.at[:,0].add(2)

    _y = jnp.zeros((y.shape[0], y.shape[1]+1), dtype=int)
    y = _y.at[:, :-1].add(y)
    # 3 = <eos>
    y = y.at[:, -1].add(3)

    logits = model_forward(model, x, x_s, key, stacks=stacks, training=True)
 

    log_probs = log_softmax(logits)
    log_probs = jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    # find where the padding tokens are
    mask = (y != 0)
    loss = -jnp.sum(log_probs * mask) / jnp.maximum(jnp.sum(mask), 1)
    return loss
    
def calculate_test_batch_loss(model, x, y, key, stacks=2):
    # Create shifted input and target sequences
    x_s = jnp.zeros((y.shape[0], y.shape[1]+1), dtype=int)
    x_s = x_s.at[:,1:].add(y)
    # 2 = <bos>
    x_s = x_s.at[:,0].add(2)

    _y = jnp.zeros((y.shape[0], y.shape[1]+1), dtype=int)
    y = _y.at[:, :-1].add(y)
    # 3 = <eos>
    y = y.at[:, -1].add(3)

    logits = model_forward(model, x, x_s, key, stacks=stacks, training=False)
 

    log_probs = log_softmax(logits)
    log_probs = jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    # find where the padding tokens are
    mask = (y != 0)
    loss = -jnp.sum(log_probs * mask) / jnp.maximum(jnp.sum(mask), 1)
    
    # Also get preds so we can check accuracy
    preds = jnp.argmax(logits, axis=-1)
    correct = (preds == y)
    acc = jnp.sum(correct * mask) / jnp.sum(mask)

    return loss, acc


def test(model, dataloader, key, timesteps=1, stacks=2, include_print=True):
    dataloader.reset()
    steps = 0
    total_loss = 0
    total_accuracy = 0
    for en, de in dataloader.get_batch():
        loss, acc = calculate_test_batch_loss(model, en, de, key, stacks=stacks)
        total_loss += loss.item()
        total_accuracy += acc.item()
        steps += 1

    writer.add_scalar("Loss/test",total_loss / steps, timesteps)
    writer.add_scalar("Accuracy/test",total_accuracy / steps, timesteps)
    _model = jax.device_get(model)
    with open(os.path.join("runs", EXP_NAME, "model.pkl"), "wb") as f:
        pickle.dump({"model": _model, "timesteps": timesteps}, f)
    del _model
    # mlflow.log_metric("Test Loss", total_loss / steps, step=timesteps)
    # mlflow.log_metric("Accuracy", total_accuracy / steps, step=timesteps)


@partial(jax.jit,static_argnames=("stacks",))
def train_batch(model, en, de, m, v, timesteps, key, stacks=2, lr=1e-5):
    loss, grads = jax.value_and_grad(calculate_train_batch_loss, argnums=0)(model, en, de, key, stacks=stacks)
    grads = clip_grads(grads, maxnorm=1.0)
    model, m, v = AdamWOptim(model, grads, m, v, timesteps, lr=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    return model, m, v, loss


def train(model, train_dataloader, test_dataloader, key, epochs=10, include_print=False, stacks=2, dmodel=512, 
          warmup=4000, batch_size=32, timesteps=1, total_timesteps=2_500_000):
    m, v = createAdamW(model)
    _lr = lambda t: dmodel ** (-0.5) * min(t**(-0.5), t * warmup**(-1.5))
    for e in range(epochs):
        print(f"Epoch {e+1}")
        train_dataloader.reset()
        # For testing
        for en, de in tqdm(train_dataloader.get_batch(), desc=f"Epoch {e+1}", total = (len(train_dataloader)//batch_size)):
            # en, de = next(train_dataloader.get_batch())
            # while timesteps < 1e5:
            lr = min(_lr(timesteps), 1e-5)
            key, subkey = jax.random.split(key)
            model, m, v, loss = train_batch(model, en, de, m, v, timesteps, subkey, stacks=stacks, lr=lr)
            timesteps += 1
            writer.add_scalar("Loss/train",loss.item(), timesteps)
            # mlflow.log_metric("Train Loss", loss.item(), step=timesteps)
            if include_print:
                print(f"Loss: {loss}")
            
            # test ~ once every 10k steps
            if timesteps % 10_000 == 0:
                test(model, test_dataloader, key, stacks=stacks, timesteps=timesteps)

            if timesteps == total_timesteps:
                return model

    return model

    
    








if __name__ == "__main__":
    # Get vocab size
    with open("/home/rgswope/workspace/llm_from_scratch/data/vocab.bpe.32000", "r") as f:
        v = sum(1 for _ in f)
    STACKS=4
    DMODEL=512
    HEADS=8
    VOCAB_SIZE = 4 + v
    MAX_LENGTH=256
    EPOCHS=5
    assert DMODEL % HEADS == 0
    hparams = {"Stacks": STACKS,
               "d_model": DMODEL,
               "Heads": HEADS,
               "Vocab Size": VOCAB_SIZE,
               "Max Length": MAX_LENGTH,
               "Total Epochs": EPOCHS
    }

    EXP_NAME = f"stacks_{STACKS}_dmodel_{DMODEL}_heads_{HEADS}_maxlen_{MAX_LENGTH}"
    writer = SummaryWriter(log_dir=os.path.join("runs", EXP_NAME))
    # mlflow.set_experiment("llm_from_scratch")
    print(f"Vocab size: {VOCAB_SIZE}")
    model = create_llm(STACKS, dmodel=DMODEL, heads=HEADS, vocab_size=VOCAB_SIZE)
    # Create a mask to delineate which params are learnable, and which are not (like activations)
    key = jax.random.key(42)
    batch_size=8
    train_dataloader = WMTDataLoader("/home/rgswope/workspace/llm_from_scratch/data/wmt_2017_en_de/wmt14_translate_de-en_train_tok.csv", 
                                     batch_size=batch_size, max_length=MAX_LENGTH)
    test_dataloader = WMTDataLoader("/home/rgswope/workspace/llm_from_scratch/data/wmt_2017_en_de/wmt14_translate_de-en_test_tok.csv", 
                                    batch_size=batch_size, max_length=MAX_LENGTH)
    # with mlflow.start_run():
    #     mlflow.log_params(hparams)
    timesteps = 1
    if os.path.exists(os.path.join("runs", EXP_NAME, "model.pkl")):
        with open(os.path.join("runs", EXP_NAME, "model.pkl"), "rb") as f:
            md = pickle.load(f)
        timesteps = md["timesteps"]
        model = md["model"]
        print(f"Successfully loaded model dict with {timesteps} timesteps.")
    model = train(model, train_dataloader, test_dataloader, key, stacks=STACKS, include_print=False, batch_size=batch_size, dmodel=DMODEL, epochs=EPOCHS, timesteps=timesteps)
    model = jax.device_get(model)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)


