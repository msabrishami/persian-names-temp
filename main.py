import string
import torch
from torch.utils.data import DataLoader
import pandas as pd
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
    dataset = pd.read_csv(CSV_FILE)
    
    # Initialize model
    model = LSTMNameClassifier(N_CHARACTERS)
    
    # Train the model
    train_model(model, dataset, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

if __name__ == "__main__":
    main()

