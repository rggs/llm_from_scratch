import jax
import jax.numpy as jnp
from collections import defaultdict




class Node:

    def __init__(self, parent, value, full_value=""):
        self.value = value
        self.parent = parent
        self.children = {}
        self.full_value = full_value + self.value
        self.is_full_token = False
    
    def add_child(self, child):
        self.children.append(child)



def _construct_merge_trie(merge_filepath):
    root_node = Node(None, "")

    with open(merge_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            chars = line.replace(" ", "")
            if chars[0] == "#":
                continue
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
    with open(vocab_filename, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            vocab_dict[line] = i

    return vocab_dict

def _tokenize_substrings(vocab_dict, substrings):
    tokens = []
    for i, s in enumerate(substrings):
        s = s.replace("</w>", "")
        if i == len(substrings) - 1:
            token = vocab_dict[s]
        else:
            token = vocab_dict[s+"@@"]
        tokens.append(token)
    return tokens



if __name__ == "__main__":
    filepath = "/home/rgswope/workspace/llm_from_scratch/data/bpe.32000"
    root_node = _construct_merge_trie(filepath)
    print([c.value for c in root_node.children.values()])
    node = root_node.children['t']
    while node is not None:
        children = list(node.children.values())
        if len(children) > 0:
            child = children[len(children)//2]
            node = child
        else:
            break
    print(node.full_value)

    # Look for breakthrough</w>
    substrings = _search_trie(root_node, "breakthrough</w>")
    print(substrings)


    bt = _find_optimal_tokenization("breakthrough</w>", root_node)
    pe = _find_optimal_tokenization("perfidious</w>", root_node)
    print(bt)
    print(pe)

    vocab_dict = _generate_token_hashmap("/home/rgswope/workspace/llm_from_scratch/data/vocab.bpe.32000")
    bt_tokens = _tokenize_substrings(vocab_dict, bt)
    pe_tokens = _tokenize_substrings(vocab_dict, pe)

    print(bt_tokens)
    print(pe_tokens)




