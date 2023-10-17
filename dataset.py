import torch
from torch.utils.data import Dataset, DataLoader
import csv
import math

class TrafficDataset(Dataset):
    def __init__(self, csv_file, src_len=100, tgt_len=100, input_cols=5, output_col=5):
        """
        Initialize the dataset.
        
        Parameters:
        - csv_file: path to the CSV file.
        - src_len: length of source sequences.
        - tgt_len: length of target sequences.
        - input_cols: number of columns to consider for source/target.
        - output_col: column index for the output.
        """
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.input_cols = input_cols
        self.output_col = output_col
        self.csv_file = csv_file
        
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip the header
            self.data = [row[1:] for row in reader] # skip the first column (time)

    def __getitem__(self, index):
        src_start = index * self.src_len
        src_end = (index + 1) * self.src_len
        tgt_start = src_end
        tgt_end = min(tgt_start + self.tgt_len, len(self.data))  # Prevent IndexError
        
        src = [[float(col) for col in row[:self.input_cols]] for row in self.data[src_start:src_end]]
        tgt = [[float(col) for col in row[:self.input_cols]] for row in self.data[tgt_start:tgt_end]]
        output = [float(self.data[src_end][self.output_col])]  # Use last row of src for output
        
        return torch.tensor(src), torch.tensor(tgt), torch.tensor(output)

    def __len__(self):
        return math.floor((len(self.data) - self.tgt_len) / self.src_len)
    
if __name__ == '__main__':
    dataset = TrafficDataset('data/100M50ms_bbr_Dldataset_202310.csv')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, (src, tgt, output) in enumerate(dataloader):
        print(src)
        print(tgt)
        print(output)
        if i == 0:
            break

    print(len(dataset))