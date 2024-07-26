import torch
import torch.nn as nn
import torch.nn.functional as F

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

import torch.optim as optim

def train_model(model, train_loader, num_epochs=10):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
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

            predicted = (outputs.squeeze() > 0.5).float()
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_predictions * 100

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
 
        
    print("Training Complete")

