"""
This module defines the BidirectionalLSTMModel class and provides a training and validation pipeline for multi-label toxic comment classification using PyTorch.
Classes:
  BidirectionalLSTMModel(nn.Module)
Main Execution:
  - Loads preprocessed toxic comment data.
  - Splits data into training and validation sets.
  - Initializes datasets and dataloaders for FastText embeddings.
  - Configures the BidirectionalLSTMModel, optimizer, loss function (with positive class weights), learning rate scheduler, and mixed precision scaler.
  - Trains the model for a specified number of epochs, logging training loss and validation F1 scores (micro and macro) to TensorBoard.
  - Saves model weights after each epoch.
Dependencies:
  - torch, torch.nn, torch.optim, torch.utils.data, torch.utils.tensorboard
  - pandas, numpy, sklearn.metrics, sklearn.model_selection
  - app.dataset (FastTextAdaptedDataset, fasttext_colleate_fn)
"""

import torch
from torch import nn

# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
class BidirectionalLSTMModel(nn.Module):
  """
  BidirectionalLSTMModel is a PyTorch neural network module for multi-label classification tasks using a bidirectional LSTM architecture.
  Args:
    hidden_size (int): Number of features in the hidden state of the LSTM. Default is 128.
    num_layers (int): Number of recurrent layers in the LSTM. Default is 1.
    label_count (int): Number of output labels/classes for classification. Default is 6.
    linear_nr_counts (int): Number of units in the intermediate linear layers. Default is 256.
    dropout (float): Dropout probability applied after LSTM and between linear layers. Default is 0.5.
  Architecture:
    - Bidirectional LSTM layer with input size 300.
    - Max pooling over the sequence dimension.
    - Three fully connected (Linear) layers with ReLU activations and dropout regularization.
    - Final output layer produces logits for each label.
  Forward Input:
    X (torch.Tensor): Input tensor of shape (batch_size, sequence_length, 300).
  Forward Output:
    logits (torch.Tensor): Output tensor of shape (batch_size, label_count) containing classification scores for each label.
  """

  def __init__(
    self, 
    hidden_size: int = 128, 
    num_layers: int = 1, 
    label_count: int = 6,
    linear_nr_counts: int = 256,
    dropout: float = 0.5,
  ):
    super().__init__()
    self.lstm = nn.LSTM(
      input_size = 300,
      hidden_size = hidden_size,
      num_layers = num_layers,
      batch_first = True,
      bidirectional = True,
    )
    self.dropout1  = nn.Dropout(dropout)
    self.linear1   = nn.Linear(hidden_size * 2, linear_nr_counts)
    self.relu1     = nn.ReLU()
    self.dropout2  = nn.Dropout(dropout)
    self.linear2   = nn.Linear(linear_nr_counts, linear_nr_counts)
    self.relu2     = nn.ReLU()
    self.dropout3  = nn.Dropout(dropout)
    self.linear3   = nn.Linear(linear_nr_counts, label_count)
  
  
  def forward(self, X):
    output, _ = self.lstm(X)
    output, _ = torch.max(output, dim=1)
    output    = self.dropout1(output)

    logits = self.linear1(output)
    logits = self.relu1(logits)
    logits = self.dropout2(logits)

    logits = self.linear2(logits)
    logits = self.relu2(logits)
    logits = self.dropout3(logits)

    logits = self.linear3(logits)
    return logits


# ----------------------------------------------------------------------
# CLI Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
  import torch
  from torch import nn
  from torch.optim import Adam
  from torch.utils.data import DataLoader
  from torch.optim.lr_scheduler import StepLR
  import pandas as pd
  import numpy as np
  from sklearn.metrics import f1_score
  from app.dataset import FastTextAdaptedDataset, fasttext_colleate_fn
  from torch.utils.tensorboard import SummaryWriter
  from sklearn.model_selection import train_test_split


  # Configuration
  device            = 'cuda' if torch.cuda.is_available() else 'cpu'
  BATCH_SIZE        = 400
  EPOCH_COUNT       = 40
  RANDOM_STATE      = 42
  POS_WEIGHTS_POWER = 0.1
  STEP_SIZE         = 10
  GAMMA             = 0.1
  LR                = 0.01
  TEXT_COLUMN_NAME  = 'comment_text'
  LABEL_NAMES = [
    'toxic', 'severe_toxic', 'obscene',
    'threat', 'insult', 'identity_hate'
  ]
  DATA_PATH = (
    '..data/processed/train/'
    'new_line_del=True,caps_lower=True,punctuation_del=True,'
    'stop_words_del=True,short_word_len=5,back_translation=True'
  )

  # TensorBoard
  writer = SummaryWriter()

  # Load and split
  df = pd.read_csv(DATA_PATH)
  train_df, val_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
  train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

  # Dataset and DataLoader
  train_ds = FastTextAdaptedDataset(train_df[TEXT_COLUMN_NAME], train_df[LABEL_NAMES])
  val_ds   = FastTextAdaptedDataset(val_df[TEXT_COLUMN_NAME], val_df[LABEL_NAMES])
  train_dl = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=fasttext_colleate_fn, num_workers=16, pin_memory=True
  )
  val_dl = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=fasttext_colleate_fn, num_workers=16, pin_memory=True
  )

  # Model, loss, optimizer, scheduler
  model = BidirectionalLSTMModel().to(device)
  counts = df[LABEL_NAMES].sum().values
  total = len(df)
  pos_weights = torch.tensor(
    (total - counts) / counts, dtype=torch.float32
  ).to(device) * POS_WEIGHTS_POWER

  loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
  optimizer = Adam(model.parameters(), lr = LR)
  scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
  scaler    = torch.amp.GradScaler('cuda')

  for epoch in range(EPOCH_COUNT):
    # Train cycle 
    model.train()
    for i, (X, y) in enumerate(train_dl):
      X = X.to(device)
      y = y.to(device)
      optimizer.zero_grad()
      with torch.amp.autocast('cuda'):
        logits = model(X)
        loss = loss_fn(logits, y)
      scaler.scale(loss).backward()
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      scaler.step(optimizer)
      scaler.update()

      global_step = i + epoch * len(train_dl)
      writer.add_scalar("Loss/train", loss.item(), global_step =global_step)
      print(f"[Epoch {epoch+1:02}/{EPOCH_COUNT}] "
            f"[Batch {i+1:03}/{len(train_dl)}] "
            f"➤ Loss: {loss:.4f}")

    # Validation cycle
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
      for X, y in val_dl:
        with torch.amp.autocast('cuda'):
          X, y = X.to(device), y.to(device)
          logits = model(X)
          probs = torch.sigmoid(logits)
        preds = (probs > 0.5).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y.cpu().numpy())   

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    micro_f1 = f1_score(all_targets, all_preds, average='micro')
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    writer.add_scalar("f1/val/micro", micro_f1, global_step = epoch)
    writer.add_scalar("f1/val/macro", macro_f1, global_step = epoch)
    print(f"[Validation] Epoch {epoch+1}: "
          f"Micro F1 = {micro_f1:.4f}, Macro F1 = {macro_f1:.4f}")
    print('-' * 50)

    scheduler.step()
    torch.save(model.state_dict(), '../models/model_weights.pth')