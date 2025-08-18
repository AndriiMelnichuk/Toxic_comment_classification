"""
Dataset class and collation function for using FastText embeddings
with PyTorch DataLoader in a multi-label classification task.
"""

import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from app.embedding import FastText

class FastTextAdaptedDataset(Dataset):
  """
  Custom PyTorch Dataset for text classification using FastText embeddings.

  Args:
      text (pd.Series): Series of text strings (already preprocessed and tokenized).
      target (pd.DataFrame): DataFrame of target labels (multi-label format).
  """
  def __init__(self, text: pd.Series,  target: pd.DataFrame):
    super().__init__()
    self.text = text
    self.target = target
    

  def __len__(self):
    """Returns the number of samples in the dataset."""
    return len(self.text)


  def __getitem__(self, index):
    """
      Retrieves one sample of text embeddings and its labels.

      Args:
        index (int): Index of the sample.

      Returns:
        X (torch.Tensor): Shape (sequence_length, embedding_dim)
        y (torch.Tensor): Shape (num_classes,)
        """
    text = str(self.text.iloc[index]).split(' ')
    labels = self.target.iloc[index].to_numpy()
    emb = FastText()

    X = torch.from_numpy(emb.text2matrix(text))
    y = torch.from_numpy(labels).to(torch.float32)
    return X, y


def fasttext_colleate_fn(batch):
  """
  Collate function for DataLoader to handle variable-length sequences.

  Args:
    batch (list of tuples): Each element is (X, y) where:
      X is a tensor of shape (seq_len, embedding_dim),
      y is a tensor of shape (num_classes,).

  Returns:
    padded_inputs (torch.Tensor): Shape (batch_size, max_seq_len, embedding_dim)
    targets (torch.Tensor): Shape (batch_size, num_classes)
  """
  inputs, targets = zip(*batch)
  padded_inputs = pad_sequence(inputs, batch_first=True)
  targets = torch.stack(targets)
  return padded_inputs, targets