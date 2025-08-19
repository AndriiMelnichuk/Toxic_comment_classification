"""
Module for working with FastText embeddings.

Provides:
- Global FastText model, loaded once at import;
- FastText class with helper methods:
  - word2vec: convert a single word to its vector representation;
  - text2matrix: convert a list of words into a matrix of embeddings.

Example:
    from embedding import FastText

    # Create wrapper instance (model already loaded globally)
    ft = FastText()

    # Convert a single word to an embedding vector
    vec = ft.word2vec("hello")

    # Convert a list of words to an embedding matrix
    matrix = ft.text2matrix(["hello", "world"])
"""

import fasttext
import numpy as np
import logging
import os


logging.basicConfig(level=logging.INFO)

SCRIPT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'cc.en.300.bin')

logging.info('Loading FastText model from %s ...', MODEL_PATH)
FAST_TEXT_MODEL = fasttext.load_model(MODEL_PATH)
logging.info('FastText model loaded')


class FastText:
  """Wrapper for FastText embeddings using a globally loaded model."""
  def word2vec(self, word: str) -> np.ndarray:
    """
    Convert a word into its FastText embedding vector.

    Args:
      word: Input word.

    Returns:
      np.ndarray: Embedding vector of shape (300,).
    """
    vec = FAST_TEXT_MODEL.get_word_vector(word)
    return np.array(vec, dtype=np.float32)


  def text2matrix(self, words: list[str]) -> np.ndarray:
    """
    Convert list of words into a matrix of embeddings.

    Args:
        words: List of input words.

    Returns:
      np.ndarray: Matrix of shape (len(words), 300).
    """
    if not words:
      return np.empty((0, 300), dtype=np.float32)
    return np.stack([self.word2vec(word) for word in words])