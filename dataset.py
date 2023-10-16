import torch
from torch.utils.data import Dataset,dataloader
import csv
import math

class TrafficDataset(Dataset):
    # TODO: add param to specify the length of src and tgt
    def __init__(self, csv_file):
        self.csv_file = csv_file
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)
            # skip the header
            next(reader)
            # remove first column
            self.data = [row[1:] for row in reader]

    def __getitem__(self, index):
        start = index*100
        end = (index+1)*100

        # get column 0-4
        src = [[float(col) for col in row[:5]] for row in self.data[start:end]]
        # get column 5
        tgt = [[float(self.data[end][5])]]

        # transform to tensor
        src = torch.tensor(src)
        tgt = torch.tensor(tgt)

        return src, tgt

    def __len__(self):
        return math.floor(len(self.data)/100)
    
if __name__ == '__main__':
    dataset = TrafficDataset('data/100M50ms_bbr_Dldataset_202310.csv')
    dataloader = dataloader.DataLoader(dataset, batch_size=1, shuffle=True)
    for i, (src, tgt) in enumerate(dataloader):
        print(src)
        print(tgt)
        if i == 1:
            break

    print(len(dataset))