"""
GloVe embedding loader and aligner.
Initializes embeddings with pre-trained GloVe vectors for faster convergence.
"""
import numpy as np
import torch
import os


def load_glove_embeddings(glove_path, vocab, embedding_dim=100):
    embeddings = np.random.randn(len(vocab), embedding_dim) * 0.01
    found = 0
    if os.path.exists(glove_path):
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) < embedding_dim + 1:
                    continue
                word = parts[0]
                vec = np.array(parts[1:], dtype=np.float32)
                if word in vocab and len(vec) == embedding_dim:
                    embeddings[vocab[word]] = vec
                    found += 1
    print(f"GloVe: found {found}/{len(vocab)} tokens ({100*found/len(vocab):.1f}%)")
    return torch.tensor(embeddings, dtype=torch.float)


def init_embeddings_with_glove(embedding_layer, glove_path, vocab, embedding_dim=100):
    glove_weights = load_glove_embeddings(glove_path, vocab, embedding_dim)
    with torch.no_grad():
        embedding_layer.weight.copy_(glove_weights)
    return embedding_layer