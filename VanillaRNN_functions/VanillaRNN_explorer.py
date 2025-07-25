##### !!!! Ensure you have Italo's n_back_spatial_task.py in the same directory as this file !!!! ##### 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from n_back_spatial_task import (  
    create_n_back_dataset,
    NBackDataset as ItaloNBackDataset
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)  

#--Configuration--
grid_size = np.array([5, 5], dtype=int)  
hidden_state_size = 128
input_size = 4 
output_size = grid_size[0] * grid_size[1] 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  
seq_length = 50  
batch_size = 100
learning_rate = 0.0005
epochs = 50
boundary = 'strict'  

class NBackVanillaRNN(nn.Module):
    def __init__(self, hidden_state_size, num_layers=1, grid_size=np.array([5, 5], dtype=int), dropout=0.2):
        super().__init__()
        self.hidden_state_size = hidden_state_size
        self.output_size = grid_size[0] * grid_size[1]
        self.num_layers = num_layers

        self.core = nn.RNN(
            input_size=4,
            hidden_size=hidden_state_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh',
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_state_size, self.output_size)

        # Initial hidden state only (no cell state in RNN)
        self.h0 = nn.Parameter(torch.randn(num_layers, 1, hidden_state_size) * 0.05)

    def forward(self, X, return_hidden=False):
        X = F.one_hot(X, num_classes=4).to(torch.float32)
        batch_size = X.size(0)

        # Expand initial hidden state to batch size
        h0 = self.h0.expand(self.num_layers, batch_size, -1).contiguous()

        states, _ = self.core(X, h0)  # Only h0 needed
        states = torch.cat((h0[-1:].transpose(0,1), states), 1)  # Add initial state to sequence

        logits = self.head(self.dropout(states))

        if return_hidden:
            return logits.transpose(1, 2), states
        else:
            return logits.transpose(1, 2)

class NBackLoss(nn.Module):
    def __init__(self, n_back):
        super().__init__()
        self.n_back = n_back
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, logits, target):
        return self.loss(logits[:, :, self.n_back:], target)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    loss_list = []
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_list.append(loss.item())

    return np.mean(loss_list)

def test(dataloader, model, loss_fn, return_hidden=False, time_step=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    
    test_loss, correct = 0, 0
    all_hidden_states = []
    current_labels = []
    nback_labels = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            if return_hidden and time_step is not None:
                pred, hidden = model(X, return_hidden=True)
                
                # Ensure we have enough sequence length
                if time_step < hidden.shape[1]:
                    for i in range(X.size(0)):
                        # Only collect if we have valid positions
                        if time_step + 1 < y.shape[1]:  # Check current position exists
                            all_hidden_states.append(hidden[i, time_step].cpu().numpy())
                            current_labels.append(y[i, time_step + 1].item())
                            
                            # Check if n-back position exists
                            if time_step - loss_fn.n_back >= 0 and time_step - loss_fn.n_back + 1 < y.shape[1]:
                                nback_labels.append(y[i, time_step - loss_fn.n_back + 1].item())
                            else:
                                nback_labels.append(-1)  # Mark as invalid
            else:
                pred = model(X)
                
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)[:, loss_fn.n_back:] == y).type(torch.float).mean().item()
    
    test_loss /= num_batches
    correct /= num_batches
    
    if return_hidden:
        # Convert to numpy arrays and filter out invalid entries
        current_labels = np.array(current_labels)
        valid_mask = current_labels != -1
        return (
            correct,
            test_loss,
            np.array(all_hidden_states)[valid_mask],
            current_labels[valid_mask],
            np.array(nback_labels)[valid_mask]
        )
    return correct, test_loss

def sample_and_train(n_back, hidden_units, num_layers=1, 
                    train_sample_size=100_000, test_sample_size=10_000, 
                    batch_size=1000, epochs=epochs, lr=learning_rate, 
                    dropout=0.2, l2_reg=0., boundary=boundary, device=device):
    
    train_dataloader = DataLoader(create_n_back_dataset(train_sample_size, n_back, boundary=boundary), 
                                batch_size=batch_size)
    test_dataloader = DataLoader(create_n_back_dataset(test_sample_size, n_back, boundary=boundary), 
                               batch_size=batch_size)
    
    model = NBackVanillaRNN(hidden_units, num_layers=num_layers).to(device)
    loss_fn = NBackLoss(n_back)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    
    acc, test_loss, train_loss = [], [], []
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        tr = train(train_dataloader, model, loss_fn, optimizer)
        a, t = test(test_dataloader, model, loss_fn)
        acc.append(a)
        test_loss.append(t)
        train_loss.append(tr)
        
    print("Done!")
    return model, acc, test_loss, train_loss, test_dataloader