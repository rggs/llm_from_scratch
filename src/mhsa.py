import jax
import jax.numpy as jnp
from src.scaled_attention import softmax, ScaledDotProductAttention


def _gen_self_attention(dk=64, dv=64, dmodel=512, key=jax.random.key(42)):
    Wq = jax.random.uniform(key, (dmodel, dk), maxval=0.001)
    Wk = jax.random.uniform(key, (dmodel, dk), maxval=0.001)
    Wv = jax.random.uniform(key, (dmodel, dv), maxval=0.001)
    return Wq, Wk, Wv


def _gen_multi_head_self_attention(dmodel=512, h=8, dk=64, dv=64, key=jax.random.key(42)):
    K = jax.random.uniform(key, (1, dmodel), maxval=0.001)
    V = jax.random.uniform(key, (1, dmodel), maxval=0.001)
    Wo = jax.random.uniform(key, (h*dv, dmodel), maxval=0.001)

    heads = {}
    wqs = []
    wks = []
    wvs = []
    for i in range(h):
        wq, wk, wv = _gen_self_attention(key=jax.random.key(i))
        wqs.append(wq)
        wks.append(wk)
        wvs.append(wv)

    wqs = jnp.stack(wqs, axis=0)
    wks = jnp.stack(wks, axis=0)
    wvs = jnp.stack(wvs, axis=0)

    params = {}
    params["K"] = K
    params["V"] = V
    params["Wo"] = Wo
    params["Wqs"] = wqs
    params["Wks"] = wks
    params["Wvs"] = wvs

    return params

@jax.jit
def _forward_mhsa(Q, mhsa):
    K = mhsa["K"]
    V = mhsa["V"]
    Wo = mhsa["Wo"]
    Wqs = mhsa["Wqs"]
    Wks = mhsa["Wks"]
    Wvs = mhsa["Wvs"]

    # Q has shape (b, seq, d_model)
    # Project to the 8 heads
    # We'll also put the stacked dim at the end
    stacked_attention = jax.vmap(_single_head_forward, in_axes=[None, None, None, 0,0,0], out_axes=-1)

    head_results = stacked_attention(Q, K, V, Wqs, Wks, Wvs)
    
    head_results = head_results.reshape(Q.shape[0], Q.shape[1], -1)

    results = head_results @ Wo

    return results


@jax.jit
def _single_head_forward(Q, K, V, Wq, Wk, Wv):
    q = Q @ Wq
    k = K @ Wk
    v = V @ Wv
    return ScaledDotProductAttention(q,k,v)



if __name__=="__main__":
    key = jax.random.key(42)
    mhsa = _gen_multi_head_self_attention()
    
    Q = jax.random.uniform(key, (1, 100, 512))

    attn_output = _forward_mhsa(Q, mhsa)

    print(attn_output)
    print(attn_output.shape)
