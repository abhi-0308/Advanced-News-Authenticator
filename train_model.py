import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import time
import logging
from math import ceil

# Configure for maximum performance
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimized Constants
MAX_VOCAB = 12000
MAX_LEN = 256
BATCH_SIZE = 128
EPOCHS = 15
DROPOUT = 0.3
NUM_WORKERS = 4

class OptimizedNewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def fast_clean_text(text):
    """Streamlined text cleaning"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:500]

class LiteFakeNewsDetector(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 96)
        self.lstm = nn.LSTM(96, 64, 
                           num_layers=1,
                           bidirectional=True,
                           batch_first=True)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return self.sigmoid(self.fc(x[:, -1, :]))

def load_and_balance_data():
    """Load data with automatic labeling"""
    try:
        df_fake = pd.read_csv('aiproject/data/Fake.csv').assign(label=0)
        df_real = pd.read_csv('aiproject/data/True.csv').assign(label=1)
        
        if 'text' not in df_fake.columns or 'text' not in df_real.columns:
            raise ValueError("CSV files must contain 'text' column")
            
        min_len = min(len(df_fake), len(df_real))
        return pd.concat([
            df_fake.sample(min_len),
            df_real.sample(min_len)
        ]).sample(frac=1).reset_index(drop=True)
    
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Load and preprocess data
    logger.info("Loading and preprocessing data...")
    try:
        df = load_and_balance_data()
        df['text'] = df['text'].apply(fast_clean_text)
        
        # 2. Tokenization
        logger.info("Tokenizing text...")
        tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<UNK>')
        tokenizer.fit_on_texts(df['text'])
        sequences = tokenizer.texts_to_sequences(df['text'])
        X = pad_sequences(sequences, maxlen=MAX_LEN, truncating='post')
        y = df['label'].values
        
        # 3. Prepare datasets
        dataset = OptimizedNewsDataset(
            torch.tensor(X, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        )
        
        train_size = int(0.8 * len(dataset))
        train_data, val_data = random_split(dataset, [train_size, len(dataset) - train_size])
        
        # 4. Create optimized data loaders
        train_loader = DataLoader(
            train_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_data,
            batch_size=BATCH_SIZE * 2,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        
        # 5. Initialize model
        model = LiteFakeNewsDetector(MAX_VOCAB).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 6. Training loop
        logger.info(f"Starting training for {EPOCHS} epochs...")
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0
            start_time = time.time()
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(device, non_blocking=True)
                    y_val = y_val.to(device, non_blocking=True)
                    
                    outputs = model(X_val)
                    val_loss += criterion(outputs, y_val).item()
                    correct += ((outputs >= 0.5).float() == y_val).sum().item()
            
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Time: {epoch_time:.1f}s | "
                f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
                f"Val Acc: {correct/len(val_data):.2%}"
            )
        
        # 7. Save model
        torch.save(model.state_dict(), 'aiproject/model/optimized_model.pt')
        with open('aiproject/model/tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)
        
        logger.info("Training complete! Model saved to aiproject/model/")
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    train()