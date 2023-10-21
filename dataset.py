import torch
from torch.utils.data import Dataset, DataLoader
import csv
import math
import os
import numpy as np


class TrafficDataset(Dataset):
    def __init__(
        self,
        csv_file_dir,
        src_len=100,
        tgt_len=100,
        input_cols=8,
        output_col=-1,
        normalize=True,
    ):
        """
        Initialize the dataset.

        Parameters:
        - csv_file_dir: path to the dir of CSV file.
        - src_len: length of source sequences.
        - tgt_len: length of target sequences.
        - input_cols: number of columns to consider for source/target.
        - output_col: column index for the output.
        """
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.input_cols = input_cols
        self.output_col = output_col

        self.data = []
        # Read all csv file data into memory
        filenames = os.listdir(csv_file_dir)
        for filename in filenames:
            if filename.endswith(".csv"):
                with open(os.path.join(csv_file_dir, filename)) as f:
                    reader = csv.reader(f)
                    self.data.extend(list(self._get_data(reader)))

        self.normalize = normalize
        self.mean = [0.0 for _ in range(input_cols)]
        self.std = [1.0 for _ in range(input_cols)]

        if self.normalize:
            self._compute_normalization_params()
            self._normalize_data()

    def __getitem__(self, index):
        src_start = index * self.src_len
        src_end = (index + 1) * self.src_len
        tgt_start = src_end
        tgt_end = min(tgt_start + self.tgt_len, len(self.data))  # Prevent IndexError

        src = [
            [float(col) for col in row[: self.input_cols]]
            for row in self.data[src_start:src_end]
        ]
        tgt = [
            [float(col) for col in row[: self.input_cols]]
            for row in self.data[tgt_start:tgt_end]
        ]
        output = [
            float(self.data[tgt_end - 1][self.output_col])
        ]  # Use last row of tgt for output

        return torch.tensor(src), torch.tensor(tgt), torch.tensor(output)

    def __len__(self):
        return math.floor((len(self.data) - self.tgt_len) / self.src_len)

    def _get_data(self, reader):
        next(reader)  # Skip header
        # remove incomplete data
        rows = list(reader)
        length = math.floor((len(rows) - self.tgt_len) / self.src_len) * self.src_len
        rows = rows[:length]
        return rows

    def _compute_normalization_params(self):
        # Compute mean and standard deviation for each column
        all_data = [row[: self.input_cols] for row in self.data]
        numpy_data = np.array(all_data, dtype=float)
        self.mean = np.mean(numpy_data, axis=0).tolist()
        self.std = np.std(numpy_data, axis=0).tolist()

    def _normalize_data(self):
        for row in self.data:
            for i in range(self.input_cols):
                if self.std[i] != 0:
                    row[i] = (float(row[i]) - self.mean[i]) / self.std[i]
            row[self.output_col] = float(row[self.output_col]) / 1e7

    def get_normalization_params(self):
        return self.mean, self.std


if __name__ == "__main__":
    dataset = TrafficDataset("data/csvfile")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, (src, tgt, output) in enumerate(dataloader):
        print(src)
        print(tgt)
        print(output)
        if i == 0:
            break

    print(len(dataset))
