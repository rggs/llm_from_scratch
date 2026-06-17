from src.tokenizer import _generate_token_hashmap, _construct_merge_trie, encode, decode
import pandas as pd
import json
import sys


if __name__ == "__main__":
    filepath = "/home/rgswope/workspace/llm_from_scratch/data/vocab.bpe.32000"
    vocab_dict, token_dict = _generate_token_hashmap(filepath)
    root_node = _construct_merge_trie(filepath)


    data_paths = ["/home/rgswope/workspace/llm_from_scratch/data/wmt_2017_en_de/wmt14_translate_de-en_test.csv",
                  "/home/rgswope/workspace/llm_from_scratch/data/wmt_2017_en_de/wmt14_translate_de-en_train.csv",
                  "/home/rgswope/workspace/llm_from_scratch/data/wmt_2017_en_de/wmt14_translate_de-en_validation.csv"]
    for data_file in data_paths:
        newname = data_file.replace(".csv", "_tok.csv")
        with open(newname, "w") as output:
            # Since the training file is huge we need to do this in chunks
            alldata = pd.read_csv(data_file, chunksize=10000, engine="python", on_bad_lines="skip")
            for data in alldata:
                for i, row in data.iterrows():
                    #opening the output file buffer lets us directly write out each row instead of aggregating
                    #rows in a dict, which also saves us memory
                    output_dict = {c: encode(vocab_dict, root_node, [str(row[c])])[0] for c in row.index}
                    output.write(json.dumps(output_dict) + "\n")

