import string
import torch
from torch.utils.data import DataLoader
from dataset import NameDataset
from model import LSTMNameClassifier, train_model

# Constants
CSV_FILE = 'persian_names_part0.csv'
BATCH_SIZE = 32
NUM_EPOCHS = 10
MAX_LEN = 50
N_CHARACTERS = len(string.ascii_letters + string.digits + string.punctuation + ' ')

def main():
    # Load dataset
    dataset = NameDataset(CSV_FILE, max_len=MAX_LEN)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = LSTMNameClassifier(N_CHARACTERS)
    
    # Train the model
    train_model(model, train_loader, num_epochs=NUM_EPOCHS)

if __name__ == "__main__":
    main()

