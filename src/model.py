import jax
import jax.numpy as jnp


from src.feed_forward import linear_forward, create_ff, ff_forward, relu, softmax
from src.mha import _gen_multi_head_attention, _forward_mha
from src.dropout import dropout
from src.embedding import create_embedding_layer, gen_pos_encodings
from src.optimizer import createAdamW, AdamWOptim
from src.dropout import dropout


h = 8
dk = dv = 64
dm = 512
PAD_IDX = 0

def layer_norm(x, eps=1e-6):
    # Shape is ~ B, dmodel, dmodel
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x-mean)**2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


def create_encoder(stacks=6, dmodel=512, heads=8, key=None):
    if key is None:
        key = jax.random.PRNGKey(42)
    encoder_dict = {}
    for s in range(stacks):
        # encoder_dict[f"norm_{s}_0"] = "layer_norm"
        key, mhsa_key, ff_key = jax.random.split(key, 3)
        encoder_dict[f"mhsa_{s}"] = _gen_multi_head_attention(dk=int(dmodel/heads), dv=int(dmodel/heads), dmodel=dmodel, h=heads, key=mhsa_key)
        # encoder_dict[f"norm_{s}_1"] = "layer_norm"
        encoder_dict[f"ff_{s}"] = create_ff([dmodel, 2048, dmodel], activations=["relu", None], key=ff_key)
    s+=1
    # encoder_dict[f"norm_{s}_0"] = "layer_norm"
    return encoder_dict


def create_decoder(stacks=6, dmodel=512, heads=8, key=None):
    if key is None:
        key = jax.random.PRNGKey(42)

    decoder_dict = {}
    for s in range(stacks):
        key, mhma_key, mhca_key, ff_key = jax.random.split(key, 4)
        # decoder_dict[f"norm_{s}_0"] = "layer_norm"
        decoder_dict[f"mhma_{s}"] = _gen_multi_head_attention(dk=int(dmodel/heads), dv=int(dmodel/heads), dmodel=dmodel, h=heads, key=mhma_key)
        # decoder_dict[f"norm_{s}_1"] = "layer_norm"
        decoder_dict[f"mhca_{s}"] = _gen_multi_head_attention(dk=int(dmodel/heads), dv=int(dmodel/heads), dmodel=dmodel, h=heads, key=mhca_key)
        # decoder_dict[f"norm_{s}_2"] = "layer_norm"
        decoder_dict[f"ff_{s}"] = create_ff([dmodel, 2048, dmodel], activations=["relu", None], key=ff_key)
    s+=1
    # decoder_dict[f"norm_{s}_0"] = "layer_norm"
    return decoder_dict

def create_llm(stacks=6, dmodel=512, heads=8, vocab_size=37_000, key=None):
    if key is None:
        key = jax.random.key(42)
    model_params = {}
    model_params["embeddings"] = create_embedding_layer(vocab_size=vocab_size, model_size=dmodel)
    key, _key = jax.random.split(key)
    model_params["encoder"] = create_encoder(stacks=stacks, dmodel=dmodel, heads=heads, key=key)
    key, _key = jax.random.split(key)
    model_params["decoder"] = create_decoder(stacks=stacks, dmodel=dmodel, heads=heads, key=key)
    
    return model_params

def model_forward(model_params, x, x_s, key, stacks=2, training=True, p_dropout=0.1):
    # Create shifted targets
    # x_s = jnp.zeros((x.shape[0], x.shape[1]+1), dtype=int)
    # x_s = x_s.at[:,1:].add(x)
    # Create the encoder & decoder masks here
    encoder_not_pad = x != PAD_IDX
    decoder_not_pad = x_s != PAD_IDX

    encoder_mask = encoder_not_pad[:, None, :]
    cross_mask = decoder_not_pad[:, None, :]
    tri = jnp.tril(jnp.ones((x_s.shape[1], x_s.shape[1])))
    decoder_mask = (tri[None, :, :] & decoder_not_pad[:, None, :])

    # perform embedding
    x = model_params["embeddings"][x]
    x_s = model_params["embeddings"][x_s]

    # Create positional embeddings
    x = x + gen_pos_encodings(x.shape[1], x.shape[2])
    x_s = x_s + gen_pos_encodings(x_s.shape[1], x_s.shape[2])

    # Encoder forward
    for s in range(stacks):
        # if model_params["encoder"][f"norm_{s}_0"] == "layer_norm":
        _x = _forward_mha(x, x, x, model_params["encoder"][f"mhsa_{s}"], mask=encoder_mask)
        key, subkey = jax.random.split(key)
        _x = dropout(_x, subkey, P=p_dropout, train=training)
        x = x + _x
        x = layer_norm(x)
        # if model_params["encoder"][f"norm_{s}_1"] == "layer_norm":
        _x = ff_forward(model_params["encoder"][f"ff_{s}"], x)
        key, subkey = jax.random.split(key)
        _x = dropout(_x, subkey, P=p_dropout, train=training)
        x = x + _x
        x = layer_norm(x)

    s += 1
    # if model_params["encoder"][f"norm_{s}_0"] == "layer_norm":
    # x = layer_norm(x)

    for s in range(stacks):
        # if model_params["decoder"][f"norm_{s}_0"] == "layer_norm":

        _x = _forward_mha(x_s, x_s, x_s, model_params["decoder"][f"mhma_{s}"], mask=decoder_mask)
        key, subkey = jax.random.split(key)
        _x = dropout(_x, subkey, P=p_dropout, train=training)
        x_s = x_s + _x
        x_s = layer_norm(x_s)

        # if model_params["decoder"][f"norm_{s}_1"] == "layer_norm":
        _x = _forward_mha(x_s, x, x, model_params["decoder"][f"mhca_{s}"], mask=cross_mask)
        key, subkey = jax.random.split(key)
        _x = dropout(_x, subkey, P=p_dropout, train=training)
        x_s = x_s + _x
        x_s = layer_norm(x_s)

        # if model_params["decoder"][f"norm_{s}_2"] == "layer_norm":
        _x = ff_forward(model_params["decoder"][f"ff_{s}"], x_s)
        key, subkey = jax.random.split(key)
        _x = dropout(_x, subkey, P=p_dropout, train=training)
        x_s = x_s + _x
        x_s = layer_norm(x_s)

    s += 1
    # if model_params["decoder"][f"norm_{s}_0"] == "layer_norm":
    # x_s = layer_norm(x_s)

    # We need to swap axis in the embeddings to go back from embedding space to token space
    logits = x_s @ jnp.swapaxes(model_params["embeddings"], -1, -2)

    return logits



if __name__ == "__main__":
    key = jax.random.key(42)
    model = create_llm(2)
    test_input = jax.random.randint(key, (8, 800), 0, 37000)
    test_input_shifted = jnp.zeros((test_input.shape[0], test_input.shape[1]+1), dtype=int)
    test_input_shifted = test_input_shifted.at[:,1:].add(test_input)
    output = model_forward(model, test_input, test_input_shifted)
    print(output.shape)









