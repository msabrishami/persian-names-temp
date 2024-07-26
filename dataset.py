import pandas as pd
import torch
from torch.utils.data import Dataset

# Create a character to index mapping
import string
all_characters = string.ascii_letters + string.digits + string.punctuation + ' '
char_to_index = {ch: idx for idx, ch in enumerate(all_characters)}

def name_to_tensor(name, max_len=50):
    indices = [char_to_index.get(ch, 0) for ch in name[:max_len]]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return torch.tensor(indices, dtype=torch.long)

class NameDataset(Dataset):
    def __init__(self, csv_file, max_len=50):
        self.data = pd.read_csv(csv_file)
        self.max_len = max_len
        
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

