"""
Text preprocessing and back-translation utilities for NLP toxic comment classification.
Uses Polars for fast DataFrame operations, spaCy for lemmatization,
and Google Translate API for data augmentation.
"""

import re
import string
import spacy

import polars as pl
from nltk.corpus import stopwords

from googletrans import Translator
import asyncio
import httpx

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS_PATTERN = r"\b(" + "|".join(map(re.escape, STOP_WORDS)) + r")\b"

nlp = spacy.load("en_core_web_sm")

# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------
def create_preprocessed_df(
  df: pl.DataFrame,
  new_line_del: bool = False,
  caps_lower: bool = False,
  punctuation_del: bool = False,
  stop_words_del: bool = False,
  short_word_len: int = 5,
  text_column_name: str = 'comment_text'
) -> pl.DataFrame:
  """
    Preprocesses the given Polars DataFrame text column.

    Args:
      df (pl.DataFrame): Input DataFrame.
      new_line_del (bool): Replace `\n` with space instead of token "NewLine".
      caps_lower (bool): Convert text to lowercase.
      punctuation_del (bool): Remove all punctuation.
      stop_words_del (bool): Remove English stop words.
      short_word_len (int): Filter out rows with <= N words.
      text_column_name (str): Column name containing text.

    Returns:
      pl.DataFrame: Preprocessed DataFrame.
  """

  # Step 1: Basic regex cleaning
  preprocessed_column =pl.col(text_column_name)\
    .str.replace_all(r'https?://\S+', 'URL')\
    .str.replace_all(r'\d+', '')\
    .str.replace_all('\n', ' ' if new_line_del else ' NewLine')
  
  # Step 2: Case handling
  if caps_lower:
    preprocessed_column = preprocessed_column.str.to_lowercase()

  # Step 3: Punctuation handling
  if punctuation_del:
    punctuation_pattern = r'[^\w\s<>]'
    preprocessed_column = preprocessed_column.str.replace_all(punctuation_pattern, '')
  else:
    keep = {'!', '.', '-', ',', '?'}
    all_punct = set(string.punctuation)
    remove_punct = ''.join(all_punct - keep)
    pattern_remove = f"[{re.escape(remove_punct)}]"
    preprocessed_column = preprocessed_column.str.replace_all(pattern_remove, ' ')

  # Step 4: Lemmatization
  preLemmatize = df.with_columns(preprocessed_column)
  texts = preLemmatize.select(pl.col(text_column_name)).to_series().to_list()
  lemmas = [[token.lemma_ for token in doc] for doc in nlp.pipe(texts, batch_size=256, n_process=32)]
  
  result = (
    pl.DataFrame({text_column_name: lemmas})
    .with_columns(pl.col(text_column_name).list.join(' '))
    .with_columns(preLemmatize.drop(text_column_name))
  )
  
  # Step 5: Stop words removal
  if stop_words_del:
    result = result.with_columns(
      pl.col(text_column_name).str.replace_all(STOP_WORDS_PATTERN, '')
    )
    
  # Step 6: Final cleanup
  result = (
    result.with_columns(pl.col(text_column_name).str.replace_all(r'\s{2,}', ' '))
    .filter(pl.col(text_column_name).str.split(' ').list.len() > short_word_len)
    .unique()
  )

  return result


# ----------------------------------------------------------------------
# Back Translation
# ----------------------------------------------------------------------
def back_translation(
  df: pl.DataFrame, 
  text_column_name: str = 'comment_text', 
  languages: list[str] = ['de','fr','es'], 
  max_concurrent = 128,
) -> pl.DataFrame:
  """
  Performs back-translation for data augmentation.

  Args:
    df (pl.DataFrame): Input DataFrame.
    text_column_name (str): Column name containing text.
    languages (list[str]): List of intermediate languages for translation.
    max_concurrent (int): Max concurrent translation requests.

  Returns:
    pl.DataFrame: Original + augmented data.
  """

  texts = df.filter(df.drop(text_column_name).sum_horizontal() != 0)
  results = asyncio.run(
    translate_all(texts[text_column_name].to_list(), languages, max_concurrent)
  )

  new_df = pl.DataFrame({text_column_name: results}).with_columns(
    texts.drop(text_column_name)
  ).explode(text_column_name)

  res = pl.concat([df, new_df]).unique()
  return res


async def translate_all(texts: str, langs: list[str], max_concurrent=32):
  semaphore = asyncio.Semaphore(max_concurrent)
  counter = 0
  total = len(texts)
  lock = asyncio.Lock()

  async def translate_one(text):
    nonlocal counter
    async with semaphore:
      result =  await translate_sync(text, langs)
      async with lock:
        counter +=1
        print(f'Translated: {counter}/{total} texts')
      return result

  tasks = [translate_one(t) for t in texts]
  return await asyncio.gather(*tasks)


async def translate_sync(
  text: str, mid_list: list[str] = ['de', 'fr', 'es'], retries=15
) -> list[str]:
  """
  Translates text to multiple languages and back to English.

  Args:
    text (str): Input text.
    mid_list (list[str]): Intermediate languages.
    retries (int): Retry attempts on timeout.

  Returns:
    list[str]: Back-translated variants.
  """
  tr = Translator(timeout=60.0)
  result = []
  for mid in mid_list:
    for _ in range(retries):
      try: 
        translation = (await tr.translate(text,   src='en', dest=mid)).text
        translation = (await tr.translate(translation, src=mid,  dest='en')).text
        result.append(translation)
        break
      except (httpx.ReadTimeout, httpx.ConnectTimeout):
        await asyncio.sleep(2)
  return result


# ----------------------------------------------------------------------
# CLI Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
  data_path           = '../data/train.csv'
  new_line_del        = True
  caps_lower          = True
  punctuation_del     = True
  stop_words_del      = True
  short_word_len      = 5
  is_train            = True
  is_back_translation = True
  
  df = pl.read_csv(data_path)

  if is_train:
    df = df.drop('id')

  if is_train and is_back_translation:
    df = back_translation(df)

  df = create_preprocessed_df(
    df,
    new_line_del=new_line_del,
    caps_lower=caps_lower,
    punctuation_del=punctuation_del,
    stop_words_del=stop_words_del,
    short_word_len=short_word_len
  )

  output_path = (
    f'../data/processed/'
    f'{'train' if is_train else 'test'}/'
    f'new_line_del={new_line_del},caps_lower={caps_lower},'
    f'punctuation_del={punctuation_del},stop_words_del={stop_words_del},'
    f'short_word_len={short_word_len},back_translation={is_back_translation}',
  )

  df.write_csv(output_path)