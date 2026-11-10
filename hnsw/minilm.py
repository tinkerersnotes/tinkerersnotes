"""Sentence embeddings with MiniLM: texts to unit vectors whose dot products measure similarity of meaning.

all-MiniLM-L6-v2 is a small transformer (6 layers, 22M parameters) that sentence-transformers trained on about
a billion pairs of related sentences, so related texts end up pointing in similar directions. This runs its
ONNX export directly, doing the steps sentence-transformers would:

1. Tokenize: split each text into word pieces from a 30,522 entry vocabulary, at most 256 of them, and pad the
   batch to its longest text. The attention mask marks real tokens with 1 and padding with 0.
2. Transformer: each token comes out as a 384 dimensional vector that depends on the whole text.
3. Mean pooling: average the real tokens' vectors (the mask leaves out the padding) into one vector per text.
4. Normalise to length 1, so the dot product of two embeddings is their cosine similarity.

The model (about 90 MB) and tokenizer are downloaded from Hugging Face on first use and cached.
"""
import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from tokenizers import Tokenizer

REPO = 'sentence-transformers/all-MiniLM-L6-v2'


def download(repo, filename):
    """Path of a file of a Hugging Face repository, from the cache without asking the server when it's there."""
    try:
        return hf_hub_download(repo, filename, local_files_only=True)
    except LocalEntryNotFoundError:
        return hf_hub_download(repo, filename)


class MiniLM:
    def __init__(self, repo=REPO, max_tokens=256):
        self.tokenizer = Tokenizer.from_file(download(repo, 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_tokens)
        self.tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
        self.session = onnxruntime.InferenceSession(download(repo, 'onnx/model.onnx'),
                                                    providers=['CPUExecutionProvider'])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.dim = self.session.get_outputs()[0].shape[-1]

    def tokens(self, text):
        return self.tokenizer.encode(text).tokens

    def encode(self, texts, batch_size=64):
        """Embeddings of texts as rows of an array of shape (len(texts), 384), each of length 1."""
        if isinstance(texts, str):
            texts = [texts]
        result = np.empty((len(texts), self.dim), dtype=np.float32)
        # Texts of similar length batched together pad less
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            encodings = self.tokenizer.encode_batch([texts[i] for i in batch])
            feeds = {
                'input_ids': np.array([e.ids for e in encodings], dtype=np.int64),
                'attention_mask': np.array([e.attention_mask for e in encodings], dtype=np.int64),
                'token_type_ids': np.array([e.type_ids for e in encodings], dtype=np.int64),
            }
            hidden = self.session.run(None, {k: v for k, v in feeds.items() if k in self.inputs})[0]
            mask = feeds['attention_mask'][:, :, None].astype(np.float32)
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
            result[batch] = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return result
