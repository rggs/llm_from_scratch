import jax
import jax.numpy as jnp
import json
import random

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD_IDX=0

class WMTDataLoader:
    def __init__(self, filepath, batch_size=32, shuffle=True, max_length=256):
        # We should be able to just read in the files and hold them in memory, they aren't that large
        self.data = []
        with open(filepath, "r") as f:
            for row in f:
                self.data.append(json.loads(row))
        self.batch_idxs = []
        self.batch_size=batch_size
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def get_batch(self):
        while len(self.batch_idxs) > 0:
            idxs = self.batch_idxs.pop()
            de = []
            en = []
            # Make the default max_length 256. This way, most batches will have padded length 256 and jit won't have to recalculate
            max_en_len = 128
            max_de_len = 128
            for i in idxs:
                de.append(self.data[i]["de"])
                en.append(self.data[i]["en"])

                if len(de[-1]) > max_de_len:
                    max_de_len = len(de[-1])
                if len(en[-1]) > max_en_len:
                    max_en_len = len(en[-1])
            
            # This would be more efficient in the previous loop, but this is fine for now
            # en_length = min(max_en_len, self.max_length)
            # de_length = min(max_de_len, self.max_length)
            en_length = self.max_length
            de_length = self.max_length
            for i in range(len(en)):
                en[i] = en[i][:en_length]
                en[i] = en[i] + [PAD_IDX]*(en_length - len(en[i]))

                de[i] = de[i][:de_length]
                de[i].append(3) # append eos token
                de[i] = de[i] + [PAD_IDX]*(de_length - len(de[i]) + 1) # add one for the eos padded token

            yield (jnp.array(en), jnp.array(de))

    def reset(self):
        # shuffle all indices and split into batch_size sub-arrays
        data_indxs = list(range(len(self.data)))
        random.shuffle(data_indxs)
        self.batch_idxs = [data_indxs[i:i+self.batch_size] for i in range(0, len(self.data), self.batch_size)]
        # Drop the last batch since it has less samples
        self.batch_idxs = self.batch_idxs[:-1]
