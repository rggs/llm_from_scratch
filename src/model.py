import jax
import jax.numpy as jnp


from src.feed_forward import linear_forward, create_ff, ff_forward, relu, softmax
from src.mha import _gen_multi_head_attention, _forward_mha
from src.dropout import dropout
from src.embedding import create_embedding_layer, gen_pos_encodings
from src.optimizer import createAdamW, AdamWOptim


h = 8
dk = dv = 64
dm = 512

def create_encoder(stacks=6, dmodel=512, heads=8):
    encoder_dict = {}
    for s in range(stacks):
        encoder_dict[f"norm_{s}_0"] = relu
        encoder_dict[f"mhsa_{s}"] = _gen_multi_head_attention()
        encoder_dict[f"norm_{s}_1"] = relu
        encoder_dict[f"ff_{s}"] = create_ff([dmodel, 2048, dmodel], activations=[relu, None])
    s+=1
    encoder_dict[f"norm_{s}_0"] = relu
    return encoder_dict


def create_decoder(stacks=6, dmodel=512, heads=8):
    decoder_dict = {}
    for s in range(stacks):
        decoder_dict[f"norm_{s}_0"] = relu
        decoder_dict[f"mhma_{s}"] = _gen_multi_head_attention()
        decoder_dict[f"norm_{s}_1"] = relu
        decoder_dict[f"mhca_{s}"] = _gen_multi_head_attention()
        decoder_dict[f"norm_{s}_2"] = relu
        decoder_dict[f"ff_{s}"] = create_ff([dmodel, 2048, dmodel], activations=[relu, None])
    s+=1
    decoder_dict[f"norm_{s}_0"] = relu
    return decoder_dict

def create_llm(stacks=6, dmodel=512, heads=8, vocab_size=37_000):
    model_params = {"stacks":stacks}
    model_params["embeddings"] = create_embedding_layer(vocab_size=vocab_size, model_size=dmodel)
    model_params["encoder"] = create_encoder(stacks=stacks, dmodel=dmodel, heads=heads)
    model_params["decoder"] = create_decoder(stacks=stacks, dmodel=dmodel, heads=heads)
    
    return model_params


def model_forward(model_params, x, x_s):
    # Create shifted targets
    # x_s = jnp.zeros((x.shape[0], x.shape[1]+1), dtype=int)
    # x_s = x_s.at[:,1:].add(x)

    # perform embedding
    x = model_params["embeddings"][x]
    x_s = model_params["embeddings"][x_s]

    # Create positional embeddings
    x = x + gen_pos_encodings(x.shape[1], x.shape[2])
    x_s = x_s + gen_pos_encodings(x_s.shape[1], x_s.shape[2])

    # Encoder forward
    for s in range(model_params["stacks"]):
        _x = model_params["encoder"][f"norm_{s}_0"](x)
        _x = _forward_mha(_x, _x, _x, model_params["encoder"][f"mhsa_{s}"])
        x = x + _x
        _x = model_params["encoder"][f"norm_{s}_1"](x)
        _x = ff_forward(model_params["encoder"][f"ff_{s}"][0], model_params["encoder"][f"ff_{s}"][1], _x)
        x = x + _x

    s += 1
    x = model_params["encoder"][f"norm_{s}_0"](x)

    for s in range(model_params["stacks"]):
        _x = model_params["decoder"][f"norm_{s}_0"](x_s)

        tri = jnp.tril(jnp.ones((_x.shape[1], _x.shape[1])))
        attention_mask = jnp.tile(jnp.expand_dims(tri,0), [x.shape[0],1,1])
        _x = _forward_mha(_x, _x, _x, model_params["decoder"][f"mhma_{s}"], mask=attention_mask)
        x_s = x_s + _x

        _x = model_params["decoder"][f"norm_{s}_1"](x_s)
        _x = _forward_mha(_x, x, x, model_params["decoder"][f"mhca_{s}"])
        x_s = x_s + _x

        _x = model_params["decoder"][f"norm_{s}_2"](x_s)
        _x = ff_forward(model_params["decoder"][f"ff_{s}"][0], model_params["decoder"][f"ff_{s}"][1], _x)
        x_s = x_s + _x

    s += 1
    x_s = model_params["decoder"][f"norm_{s}_0"](x)

    # We need to swap axis in the embeddings to go back from embedding space to token space
    logits = x_s @ jnp.swapaxes(model_params["embeddings"], -1, -2)

    return logits



if __name__ == "__main__":
    key = jax.random.key(42)
    model = create_llm(2)
    test_input = jax.random.randint(key, (8, 800), 0, 37000)
    output = model_forward(model, test_input)
    print(output.shape)









