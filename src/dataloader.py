import jax
import jax.numpy as jnp
import json
import random

class WMTDataLoader:
    def __init__(filepath, shuffle=True):
        # We should be able to just read in the files and hold them in memory, they aren't that large
        with open(filepath, "r") as f:
            self.data = json.load(filepath)
        self.batch_idxs = []

    def __len__(self):
        return len(self.data)

    def get_batch():
        if len(self.batch_idxs) == 0:
            return None
        else:
            idxs = self.batch_idxs.pop()
            de = []
            en = []
            max_en_len = 0
            max_de_len = 0
            for i in idxs:
                de.append(self.data[i]["de"])
                en.append(self.data[i]["en"])

                if len(de[-1]) > max_de_len:
                    max_de_len = len(de[-1])
                if len(en[-1]) > max_en_len:
                    max_en_len = len(en[-1])


        
