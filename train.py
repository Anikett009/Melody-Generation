import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader

# Constants
SEQUENCE_LENGTH = 64
OUTPUT_UNITS = 38
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_UNITS = [256, 256]
SINGLE_FILE_DATASET = "file_dataset"
SAVE_MODEL = "model.pth"
MAPPING_PATH = "mapping.json"
DEVICE = torch.device("cuda:0") 

def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song

def convert_songs_to_int(songs):
    # load mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    # transform songs string to list
    songs = songs.split()

    # map songs to int
    return [mappings[symbol] for symbol in songs]

class MusicDataset(Dataset):
    def __init__(self, sequence_length):
        # load songs and map them to int
        songs = load(SINGLE_FILE_DATASET)
        self.int_songs = convert_songs_to_int(songs)
        self.sequence_length = sequence_length
        self.vocabulary_size = len(set(self.int_songs))
        
    def __len__(self):
        return len(self.int_songs) - self.sequence_length

    def __getitem__(self, idx):
        # Get sequence and target
        sequence = self.int_songs[idx:idx + self.sequence_length]
        target = self.int_songs[idx + self.sequence_length]
        
        # Convert to one-hot encoding
        sequence_tensor = torch.zeros(self.sequence_length, self.vocabulary_size)
        for i, value in enumerate(sequence):
            sequence_tensor[i][value] = 1
            
        return sequence_tensor, torch.tensor(target, dtype=torch.long)

class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MusicLSTM, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_sizes[1], output_size)
        
    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        # Second LSTM layer
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        # We only want the last output
        x = x[:, -1, :]
        
        # Dense layer
        x = self.fc(x)
        return x

def train():
    # Create dataset and dataloader
    dataset = MusicDataset(SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Vocabulary size: {dataset.vocabulary_size}")
    print(f"Number of sequences: {len(dataset)}")
    
    # Initialize model
    model = MusicLSTM(
        input_size=dataset.vocabulary_size,
        hidden_sizes=NUM_UNITS,
        output_size=dataset.vocabulary_size
    ).to(DEVICE)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        total_batches = 0
        
        for batch_idx, (sequences, targets) in enumerate(dataloader):
            sequences = sequences.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward pass
            outputs = model(sequences.float())
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()

            
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx}/{len(dataloader)}], '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / total_batches
        print(f'Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_loss:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), SAVE_MODEL)
    print("Training completed and model saved!")
    return model

if __name__ == "__main__":
    # Make sure PyTorch is available
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {DEVICE}")
    
    # Verify the existence of required files
    try:
        with open(MAPPING_PATH, "r") as fp:
            print("Mapping file found")
        with open(SINGLE_FILE_DATASET, "r") as fp:
            print("Dataset file found")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure both mapping.json and file_dataset exist in the correct locations")
    else:
        # If files exist, proceed with training
        print("Starting training process...")
        model = train()
        print("Training completed!")