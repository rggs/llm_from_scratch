import jax
import jax.numpy as jnp
from collections import defaultdict
import re

PUNCT = r"""([.,!?:(){}])"""
# There's no semi-colon here because of the special characters later on. In reality we'd want to do reject to preserve
# the punctuation where it's legit

NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", ")", "]", "}", "%"}
NO_SPACE_AFTER = {"(", "[", "{", "$", "£", "€", "“", "‘"}

def handle_special_chars(text):
    return ( text.replace("&", "&amp;").replace("|", "&#124;").replace("<", "&lt;")
        .replace(">", "&gt;").replace("'", "&apos;").replace('"', "&quot;")
        .replace("[", "&#91;").replace("]", "&#93;")
    )

def restore_special_chars(text):
    return ( text.replace("&#93;", "]").replace("&#91;", "[").replace("&quot;", '"')
        .replace("&apos;", "'").replace("&gt;", ">").replace("&lt;", "<")
        .replace("&#124;", "|").replace("&amp;", "&")
    )

class Node:

    def __init__(self, parent, value, full_value=""):
        self.value = value
        self.parent = parent
        self.children = {}
        self.full_value = full_value + self.value
        self.is_full_token = False
    
    def add_child(self, child):
        self.children.append(child)



def _construct_merge_trie(vocab_filepath):
    root_node = Node(None, "")

    with open(vocab_filepath, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line[0] == "#" and i == 0:
                continue
            if line.endswith("@@"):
                chars = line[:-2]
            else:
                chars = line + "</w>"
            current_node = root_node
            for c in chars:
                in_tree = False
                if c in current_node.children:
                    current_node = current_node.children[c]
                else:
                    new_node = Node(current_node, c, current_node.full_value)
                    current_node.children[c] = (new_node)
                    current_node = new_node
            current_node.is_full_token = True

    return root_node


def _search_trie(root_node, word):
    substrings = defaultdict(list)
    for i, base_char in enumerate(word):
        current_node = root_node
        for char in word[i:]:
            if char in current_node.children:
                current_node = current_node.children[char]
                if current_node.is_full_token:
                    substrings[base_char].append(current_node.full_value)
            else:
                break
    return substrings


def _find_optimal_tokenization(word, trie):
    dp_array = [float("inf")] * (len(word) + 1) # We'll basically be shifted by one
    parent = [-1] * (len(word) + 1)
    dp_array[0] = 0


    for start_pos in range(len(word)):
        # We haven't been able to reach this spot yet
        if dp_array[start_pos] == float("inf"):
            continue

        # Offset so indexing is correct inside the dp array
        current_node = trie
        for end_pos in range(start_pos + 1, len(word)+1):
            char = word[end_pos-1]
            if char not in current_node.children:
                break
            current_node = current_node.children[char]
            if current_node.is_full_token:
                if dp_array[start_pos] + 1 < dp_array[end_pos]:
                    dp_array[end_pos] = dp_array[start_pos] + 1
                    parent[end_pos] = start_pos

    if dp_array[-1] == float('inf'):
        return None
    
    # If there's a valid tokenization, return the reconstructed word
    chunks = []
    i = len(word)
    while i > 0:
        chunks.insert(0, word[parent[i]:i])
        i = parent[i]

    return chunks
    

def _generate_token_hashmap(vocab_filename):
    vocab_dict = {}
    token_dict = {}
    with open(vocab_filename, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            vocab_dict[line] = i
            token_dict[i] = line

    return vocab_dict, token_dict

def _tokenize_substrings(vocab_dict, substrings):
    tokens = []
    for i, s in enumerate(substrings):
        s = s.replace("</w>", "")
        if i != len(substrings) - 1:
            s = s+"@@"

        if s not in vocab_dict:
            tokens.append(-1)
        else:
            token = vocab_dict[s]
            tokens.append(token)
    return tokens


def construct_vocab_dict(bpe_filepath):
    vocab_dict, token_dict = _generate_token_hashmap(bpe_filepath)
    return vocab_dict, token_dict

def encode(vocab_dict, trie, inputs: list[str]):
    # Assume inputs is a list of strings, where each string is one or more words
    tokenized_inputs = []
    for text in inputs:
        tokens = []
        text = handle_special_chars(text)
        text = re.sub(PUNCT, r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.split()
        text = [t + "</w>" for t in text]
        for t in text:
            chunks = _find_optimal_tokenization(t, trie)
            if chunks is None:
                print("Could not tokenize:", repr(t))
                tokens.append(-1)
                continue
            tokens += _tokenize_substrings(vocab_dict, chunks)
        tokenized_inputs.append(tokens)
    return tokenized_inputs


def decode(token_dict, inputs: list[int]):
    outputs = []
    for tokens in inputs:
        strings = [token_dict[t] for t in tokens]
        text = ""
        for s in strings:
            if not text:
                text = s
            elif s in NO_SPACE_BEFORE:
                text += s
            elif text[-1] in NO_SPACE_AFTER:
                text += s
            else:
                text += " " + s
        text = restore_special_chars(text)
        outputs.append(text)
    return outputs
        

    



if __name__ == "__main__":
    filepath = "/home/rgswope/workspace/llm_from_scratch/data/vocab.bpe.32000"
    vocab_dict, token_dict = _generate_token_hashmap(filepath)
    root_node = _construct_merge_trie(filepath)
    # print([c.value for c in root_node.children.values()])
    # node = root_node.children['t']
    # while node is not None:
    #     children = list(node.children.values())
    #     if len(children) > 0:
    #         child = children[len(children)//2]
    #         node = child
    #     else:
    #         break
    # print(node.full_value)

    # # Look for breakthrough</w>
    # substrings = _search_trie(root_node, "breakthrough</w>")
    # print(substrings)


    # bt = _find_optimal_tokenization("breakthrough</w>", root_node)
    # pe = _find_optimal_tokenization("perfidious</w>", root_node)
    # print(bt)
    # print(pe)

    # bt_tokens = _tokenize_substrings(vocab_dict, bt)
    # pe_tokens = _tokenize_substrings(vocab_dict, pe)

    # print(bt_tokens)
    # print(pe_tokens)
    inputs = ["The rain in Spain falls mainly on the plain."]
    tokenized_inputs = encode(vocab_dict, root_node, inputs)
    print(tokenized_inputs)
    decoded_strings = decode(token_dict, tokenized_inputs)
    print(decoded_strings)




