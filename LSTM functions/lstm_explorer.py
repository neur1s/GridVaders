import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
import torch.nn.functional as F
from n_back_spatial_task import create_n_back_dataset, NBackDataset as ItaloNBackDataset

# Configuration constants
SEED = 0
GRID_SIZE = np.array([5, 5], dtype=int)
HIDDEN_SIZE = 256
INPUT_SIZE = 4 
OUTPUT_SIZE = GRID_SIZE[0] * GRID_SIZE[1]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEQ_LENGTH = 50
BATCH_SIZE = 100
LEARNING_RATE = 0.001
EPOCHS = 50
BOUNDARY = 'strict'

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class NBackDataset(ItaloNBackDataset):
    def __init__(self, num_samples, n_back):
        dataset = create_n_back_dataset(
            num_samples=num_samples,
            n=n_back,
            max_length=SEQ_LENGTH + n_back,
            grid_size=GRID_SIZE,
            boundary=BOUNDARY
        )
        super().__init__(dataset.x, dataset.y)

class NBackLSTM(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.h0 = nn.Parameter(torch.randn(1, 1, HIDDEN_SIZE) * 0.05)
        self.c0 = nn.Parameter(torch.randn(1, 1, HIDDEN_SIZE) * 0.05)
        
    def forward(self, x, return_hidden=False):
        batch_size = x.size(0)
        x = F.one_hot(x, num_classes=4).float().to(DEVICE)
        h0 = self.h0.expand(1, batch_size, -1).contiguous()
        c0 = self.c0.expand(1, batch_size, -1).contiguous()
        lstm_out, hidden = self.lstm(x, (h0, c0))
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        
        if return_hidden:
            return logits, lstm_out
        return logits

def train(model, n_back, epochs=EPOCHS):
    train_loader = DataLoader(
        NBackDataset(10_000, n_back),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    train_losses = []
    
    for epoch in trange(epochs, desc=f"Training N={n_back}"):
        total_loss = 0
        for actions, targets in train_loader:
            actions, targets = actions.to(DEVICE), targets.to(DEVICE)
            batch_size = actions.size(0)
            
            optimizer.zero_grad()
            logits = model(actions)
            
            loss = criterion(
                logits[:, n_back:].reshape(-1, OUTPUT_SIZE),
                targets[:, 1:].reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        train_losses.append(total_loss / len(train_loader))
    
    return train_losses

def test(model, n_back, test_samples=1_000):
    test_loader = DataLoader(
        NBackDataset(test_samples, n_back),
        batch_size=BATCH_SIZE
    )
    
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = total = 0
    test_loss = 0
    
    with torch.no_grad():
        for actions, targets in test_loader:
            actions, targets = actions.to(DEVICE), targets.to(DEVICE)
            logits = model(actions)
            
            loss = criterion(
                logits[:, n_back:].reshape(-1, OUTPUT_SIZE),
                targets[:, 1:].reshape(-1))
            test_loss += loss.item()
            
            preds = logits[:, n_back:].argmax(-1)
            tgts = targets[:, 1:]
            correct += (preds == tgts).sum().item()
            total += tgts.numel()
    
    accuracy = correct / total if total > 0 else 0
    test_loss = test_loss / len(test_loader)
    return accuracy, test_loss

def sample_and_train(n_back, epochs=EPOCHS):
    model = NBackLSTM().to(DEVICE)
    train_loss = train(model, n_back, epochs)
    accuracy, test_loss = test(model, n_back)
    return model, accuracy, test_loss, train_loss

def get_hidden_states(model, dataset, time_step, n_back, max_samples=1000):
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    all_hidden_states = []
    current_labels = []
    nback_labels = []
    
    with torch.no_grad():
        for actions, targets in loader:
            if len(all_hidden_states) >= max_samples:
                break
            actions, targets = actions.to(DEVICE), targets.to(DEVICE)
            _, hidden_states = model(actions, return_hidden=True)
            
            if time_step < hidden_states.shape[1] and (time_step + 1) < targets.shape[1]:
                if (time_step - n_back) >= 0:
                    for i in range(actions.size(0)):
                        hidden = hidden_states[i, time_step, :].cpu().numpy()
                        current_label = targets[i, time_step + 1].item()
                        nback_label = targets[i, time_step - n_back + 1].item()
                        
                        all_hidden_states.append(hidden)
                        current_labels.append(current_label)
                        nback_labels.append(nback_label)
    
    return np.array(all_hidden_states), np.array(current_labels), np.array(nback_labels)