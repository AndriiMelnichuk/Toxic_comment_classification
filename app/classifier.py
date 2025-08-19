"""
Wrapper for a trained Bidirectional LSTM model to classify text toxicity.

Provides the `ToxicityClassifier` class with:
- Preprocessing (configurable);
- Batch inference on GPU/CPU;
- Convenient methods for single and multiple texts.

Example:
    from app.toxicity_classifier import ToxicityClassifier

    clf = ToxicityClassifier("model_weights")
    print(clf.predict_text("You are amazing!"))
    print(clf.predict_texts(["I hate you", "Love this!"]))
"""

import os
import polars as pl
from app.modeling import BidirectionalLSTMModel
import torch
from app.preprocessing import create_preprocessed_df
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import sigmoid
from app.embedding import FastText
import math

class ToxicityClassifier:
  """Wrapper for toxicity classification using a trained Bidirectional LSTM model."""
  def __init__(
    self,
    weights_name='toxic_model_weights',
    new_line_del: bool = True, 
    caps_lower: bool = True,
    punctuation_del: bool = True, 
    stop_words_del: bool = False,
    remove_empty_string: bool = False,
    lemmatize: bool = False,
    text_column_name: str = 'comment_text',
    batch_size: int = 64,
    device: str = None,
    threshold: float = 0.5
  ):
    """
    Initialize toxicity classifier.

    Args:
      weights_name: Name of model weights file (.pth).
      new_line_del: Remove newlines during preprocessing.
      caps_lower: Lowercase text during preprocessing.
      punctuation_del: Remove punctuation during preprocessing.
      stop_words_del: Remove stop words during preprocessing.
      remove_empty_string: Drop empty strings after preprocessing.
      lemmatize: Apply lemmatization.
      text_column_name: Column name for text in dataframe.
      batch_size: Batch size for inference.
      device: "cuda" or "cpu" (auto-detect if None).
      threshold: Decision threshold for classification (default 0.5).
    """
    # Define model file path
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    FILE_PATH = os.path.join(SCRIPT_DIR, '..', 'models', weights_name + '.pth')
    if not os.path.exists(FILE_PATH):
      raise FileNotFoundError(f'Model file {FILE_PATH} not found')
    
    # Initialize LSTM model 
    if device is None:
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
      self.device = device
    model = BidirectionalLSTMModel()
    model.load_state_dict(torch.load(FILE_PATH, map_location=self.device))
    self.model = model.to(self.device)
    self.model.eval()
    
    # processing info
    self.new_line_del = new_line_del
    self.caps_lower = caps_lower
    self.punctuation_del = punctuation_del
    self.stop_words_del = stop_words_del
    self.text_column_name = text_column_name
    self.remove_empty_string = remove_empty_string
    self.lemmatize =lemmatize
    
    # Inference parameters
    self.batch_size = batch_size
    self.threshold = threshold
  
  
  def predict_text(self, text: str) -> list[int]:
    """Predict toxicity for a single text string."""
    df = pl.DataFrame({
      self.text_column_name: [text]
    })
    return self._predict(df)
  

  def predict_texts(self, text: list[str]) -> list[list[int]]:
    """Predict toxicity for a list of text strings."""
    df = pl.DataFrame({
      self.text_column_name: text
    })
    return self._predict(df)
  

  def _predict(self, df: pl.DataFrame) -> list[int] | list[list[int]]:
    """
    Internal prediction method.

    Args:
      df: Polars DataFrame with column `self.text_column_name`.

    Returns:
      List of predictions (binary vectors).
    """
    # Preprocessing
    df = create_preprocessed_df(
      df, new_line_del = self.new_line_del, caps_lower = self.caps_lower,
      punctuation_del = self.punctuation_del, stop_words_del = self.stop_words_del,
      lemmatize= self.lemmatize, text_column_name = self.text_column_name,
      remove_empty_string= self.remove_empty_string,
    )
    texts = df[self.text_column_name].str.split(' ').to_list()

    ft = FastText()
    batch_count = math.ceil(len(df) / self.batch_size)
    result = []

    # Inference loop
    with torch.no_grad():
      for batch in range(batch_count):
        start = batch * self.batch_size
        end = start + self.batch_size
        batch_texts = texts[start : end]

        X = pad_sequence(
          [torch.from_numpy(ft.text2matrix(t)) for t in batch_texts],
          batch_first=True
        ).to(self.device)

        res = self.model(X)
        result.append(res)
    
    logits = torch.vstack(result)
    prediction = (sigmoid(logits) > self.threshold).to(torch.int)
    return prediction.tolist()
