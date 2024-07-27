import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Ensure all_characters and char_to_index are available
import string
all_characters = string.ascii_letters + string.digits + string.punctuation + ' '
char_to_index = {ch: idx for idx, ch in enumerate(all_characters)}

def name_to_tensor(name, max_len=50):
    indices = [char_to_index.get(ch, 0) for ch in name[:max_len]]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return torch.tensor(indices, dtype=torch.long)

class LSTMNameClassifier(nn.Module):
    def __init__(self, n_characters, embedding_dim=64, hidden_dim=128, n_layers=2, max_len=50):
        super(LSTMNameClassifier, self).__init__()
        self.embedding = nn.Embedding(n_characters, embedding_dim)
        
        self.lstm_first_name = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.lstm_last_name = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, first_name, last_name):
        first_embedded = self.embedding(first_name)
        last_embedded = self.embedding(last_name)
        
        _, (first_hidden, _) = self.lstm_first_name(first_embedded)
        _, (last_hidden, _) = self.lstm_last_name(last_embedded)
        
        combined = torch.cat((first_hidden[-1], last_hidden[-1]), dim=1)
        
        x = F.relu(self.fc1(combined))
        x = torch.sigmoid(self.fc2(x))
        
        return x

class BalancedNameDataset(Dataset):
    def __init__(self, data, max_len=50):
        self.data = data
        self.max_len = max_len
        self.char_to_index = {ch: idx for idx, ch in enumerate(all_characters)}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        first_name = self.data.iloc[idx, 0]
        last_name = self.data.iloc[idx, 1]
        label = self.data.iloc[idx, 2]
        
        first_name = first_name.strip() if isinstance(first_name, str) else ""
        last_name = last_name.strip() if isinstance(last_name, str) else ""
        
        first_name_tensor = name_to_tensor(first_name, self.max_len)
        last_name_tensor = name_to_tensor(last_name, self.max_len)
        
        return first_name_tensor, last_name_tensor, torch.tensor(label, dtype=torch.float)

def balance_dataset(data, label_col='label'):
    class_0 = data[data[label_col] == 0]
    class_1 = data[data[label_col] == 1]
    
    min_class_size = min(len(class_0), len(class_1))
    
    balanced_class_0 = class_0.sample(min_class_size)
    balanced_class_1 = class_1.sample(min_class_size)
    
    balanced_data = pd.concat([balanced_class_0, balanced_class_1]).sample(frac=1).reset_index(drop=True)
    return balanced_data

def train_model(model, dataset, num_epochs=10, batch_size=32):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        balanced_data = balance_dataset(dataset)
        balanced_dataset = BalancedNameDataset(balanced_data)
        train_loader = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for first_name, last_name, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(first_name, last_name)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            predicted = (outputs.squeeze() > 0.5).float()
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_predictions * 100
        
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    
    print("Training Complete")

