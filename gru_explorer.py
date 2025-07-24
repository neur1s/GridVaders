import torch.nn.functional as F
from torch import nn
from torch import optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from n_back_spatial_task import *

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class GRUExplorer(nn.Module):
    
    def __init__(self, hidden_state_size, num_layers=1, grid_size=np.array([5, 5], dtype=int), dropout=0.2):

        super(GRUExplorer, self).__init__()

        self.hidden_state_size = hidden_state_size
        self.output_size = grid_size[0]*grid_size[1]
        self.num_layers = num_layers

        self.core = nn.GRU(4, self.hidden_state_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.hidden_state_size, self.output_size)

    def forward(self, X, return_hidden=False):

        X = F.one_hot(X, num_classes=4).to(torch.float32)
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_state_size).to(X.device)

        states, _= self.core(X, h0)
        states = torch.cat((h0[-1:].transpose(0,1), states), 1)
        
        logits = self.head(self.dropout(states))
        
        if return_hidden:
            return logits.transpose(1,2), states
        else:
            return logits.transpose(1,2)
            

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
#            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)[:, loss_fn.n_back:] == y).type(torch.float).mean().item()
    test_loss /= num_batches
    correct /= num_batches
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss
    

class NBackLoss(nn.Module):
    
    def __init__(self, n_back):
        super(NBackLoss, self).__init__()
        
        self.n_back = n_back
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, logits, target):
        
        return self.loss(logits[:, :, self.n_back:], target)


def sample_and_train(n_back, hidden_units, num_layers=1, train_sample_size=100_000, test_sample_size=10_000, batch_size=1000, epochs=10, lr=0.001, dropout=0.2, l2_reg=0., boundary='strict'):

    train_dataloader = DataLoader(create_n_back_dataset(train_sample_size, n_back, boundary=boundary), batch_size=batch_size)
    test_dataloader = DataLoader(create_n_back_dataset(test_sample_size, n_back, boundary=boundary), batch_size=batch_size)
        
    
    model = GRUExplorer(hidden_units, num_layers=num_layers).to(device)
    loss_fn = NBackLoss(n_back)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    acc, test_loss, train_loss = [], [], []
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        tr = train(train_dataloader, model, loss_fn, optimizer)
        a, t = test(test_dataloader, model, loss_fn)
        acc.append(a); test_loss.append(t); train_loss.append(tr)
        
    print("Done!")

    return model, acc, test_loss, train_loss
