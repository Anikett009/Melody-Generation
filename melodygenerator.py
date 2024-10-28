import torch
import torch.nn as nn
import json
import music21 as m21
import numpy as np
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MusicLSTM, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_sizes[1], output_size)
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class MelodyGenerator:
    def __init__(self, model_path="model.pth"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)
        
        input_size = len(self._mappings)
        hidden_sizes = [256, 256]
        output_size = len(self._mappings)
        
        self.model = MusicLSTM(input_size, hidden_sizes, output_size)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature=1.0):
        """
        Generate a melody sequence.
        
        Args:
            seed (str): Initial melody sequence
            num_steps (int): Number of steps to generate
            max_sequence_length (int): Maximum length of input sequence
            temperature (float): Controls randomness in generation
                               - temperature < 1.0: More focused/conservative
                               - temperature = 1.0: Normal sampling
                               - temperature > 1.0: More random/creative
                               Recommended range: 0.5 to 1.5
        """
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")

        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            seed = seed[-max_sequence_length:]
            
            onehot_seed = torch.zeros(len(seed), len(self._mappings))
            for i, val in enumerate(seed):
                onehot_seed[i][val] = 1

            onehot_seed = onehot_seed.unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(onehot_seed.float())
                
                # Apply temperature scaling to logits before softmax
                scaled_logits = logits / temperature
                
                # Apply softmax to get probabilities
                probabilities = torch.softmax(scaled_logits, dim=1)
                probabilities = probabilities[0].cpu().numpy()

            # Sample from the distribution
            output_int = self._sample_with_distribution(probabilities)

            seed.append(output_int)
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            if output_symbol == "/":
                break

            melody.append(output_symbol)

        return melody

    def _sample_with_distribution(self, probabilities):
        """
        Sample from a probability distribution.
        
        Args:
            probabilities (numpy.array): Array of probabilities summing to 1
            
        Returns:
            int: Sampled index from the distribution
        """
        try:
            # Ensure probabilities sum to 1 (handle numerical instability)
            probabilities = probabilities / np.sum(probabilities)
            
            # Handle any remaining numerical instability
            probabilities = np.nan_to_num(probabilities)
            probabilities = np.clip(probabilities, 0, 1)
            probabilities = probabilities / np.sum(probabilities)
            
            return np.random.choice(len(probabilities), p=probabilities)
        except ValueError as e:
            print(f"Warning: Sampling failed with error: {e}")
            print(f"Falling back to argmax selection")
            return np.argmax(probabilities)

    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.midi"):
        stream = m21.stream.Stream()
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):
            if symbol != "_" or i + 1 == len(melody):
                if start_symbol is not None:   
                    quarter_length_duration = step_duration * step_counter
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                    stream.append(m21_event)  
                    step_counter = 1
                start_symbol = symbol
            else:
                step_counter += 1

        stream.write(format, file_name)

if __name__ == "__main__":
    mg = MelodyGenerator()
    
    # Example usage with different temperatures
    seed = "60 _ 60 _ 59 _ 60 _ 62"
    
    # Generate conservative melody
    conservative_melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, temperature=0.5)
    mg.save_melody(conservative_melody, file_name="conservative_melody.midi")
    
    # Generate balanced melody
    balanced_melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, temperature=1.0)
    mg.save_melody(balanced_melody, file_name="balanced_melody.midi")
    
    # Generate creative melody
    creative_melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, temperature=1.5)
    mg.save_melody(creative_melody, file_name="creative_melody.midi")