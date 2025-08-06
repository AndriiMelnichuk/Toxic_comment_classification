"""
This module provides an interface for preprocessing textual data prior to model training,
specifically tailored for toxic comment classification tasks. It includes functions for
cleaning and normalizing text data, such as removing URLs, digits, punctuation, stopwords,
and handling capitalization and newlines. The module also supports lemmatization and
vocabulary creation from preprocessed text. The preprocessing pipeline is configurable
through various parameters, allowing for flexible experimentation with different text
cleaning strategies.
Main functionalities:
- Clean and preprocess comment text with customizable options.
- Remove or retain specific punctuation, stopwords, and handle capitalization.
- Lemmatize tokens for normalization.
- Filter out short comments based on a minimum length.
- Construct a vocabulary dictionary from the processed text.
- Designed for integration with PyTorch datasets and NLP pipelines.
interface for data processing before model learning
"""

import pandas as pd

import torch
from torch.utils.data import Dataset

import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from typing import Literal

STOP_WORDS = set(stopwords.words('english'))

def create_preprocessed_df(
  df: pd.DataFrame,
  new_line_del: bool = False,
  caps_lower: bool = False,
  punctuation_del: bool = False,
  stop_words_del: bool = False,
  short_word_len: int = 5,
  text_column_name: str = 'comment_text'
):
  df = df.copy(deep=bool)
  df[text_column_name] = df[text_column_name].apply(lambda x: clean_text(x, new_line_del, caps_lower, punctuation_del, stop_words_del))

  # too short lines cleaning
  too_short_line = df[text_column_name].apply(len) < short_word_len
  df  = df[~too_short_line].reset_index(drop=True)
    
  return df


def create_vocabulary(clean_text: pd.Series) -> dict[str, int]:
  vocabulary = {
    '<PAD>': 0,
    '<UNK>': 1,
    '<EOS>': 2,
    '<URL>': 3,
  }
  counter = len(vocabulary) + 1
  for text in clean_text:
    for word in text.split(' '):
        if word not in vocabulary.keys():
           vocabulary[word] = counter
  return vocabulary


def clean_text(text: str, new_line_del: bool, caps_lower: bool, punctuation_del: bool, stop_words_del: bool) -> str:
  text = re.sub(r'http\S+', '<URL>', text) # remove urls
  text = re.sub(r'\d+', '', text) # remove digits

  if new_line_del:
    text = text.replace('\n', ' ')
  else:
    text = text.replace('\n', ' NewLine ')
  
  # TODO: new strategy: 'HELLO' -> '<CAPS> hello'
  if caps_lower: 
    text = text.lower()

  if punctuation_del:
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
  else:
    keep = {'!', '.', '-', ',', '?'}
    # Remove all punctuation except those in keep
    punct_to_remove = ''.join(c for c in string.punctuation if c not in keep)
    translator = str.maketrans({c: ' ' for c in punct_to_remove})
    text = text.translate(translator)
    # Collapse consecutive keep characters to a single occurrence
    for char in keep:
      pattern = re.escape(char) + r'{2,}'
      text = re.sub(pattern, char, text)
  
  text = re.sub(r'\s+', ' ', text).strip() # remove spaces

  if stop_words_del:
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    text = ' '.join(words)
  
  return lemmatize_text(text)


def lemmatize_text(text: str, lemmatizer = WordNetLemmatizer()) -> str:
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(lemmatized_words)
  

if __name__ == "__main__":
  df = pd.read_csv('data/train.csv')

  new_line_del=True,
  caps_lower= True,
  punctuation_del=True,
  stop_words_del=True,
  short_word_len = 5,
  
  preprocessed_df = create_preprocessed_df(
    df,
    new_line_del=new_line_del,
    caps_lower=caps_lower,
    punctuation_del=punctuation_del,
    stop_words_del=stop_words_del,
    short_word_len=short_word_len
  )

  preprocessed_df.to_csv('data_preprocessed/csv/new_line_del={new_line_del},caps_lower={caps_lower},punctuation_del={punctuation_del},stop_words_del={stop_words_del},short_word_len={short_word_len}')

