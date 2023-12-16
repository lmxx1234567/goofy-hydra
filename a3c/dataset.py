from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class A2CDataSet(Dataset):
    def __init__(
        self,
        csv_file_dir,
        input_cols: List[str]=["packet_count","1_TXrate","1_tcpi_lost","1_tcpi_reordering","1_tcpi_retransbyte","1_tcpi_retranscount","1_tcpi_rtt","1_tcpi_snd_cwnd","1_tcpi_unacked","1_tcpi_unsend","2_TXrate","2_tcpi_lost","2_tcpi_reordering","2_tcpi_retransbyte","2_tcpi_retranscount","2_tcpi_rtt","2_tcpi_snd_cwnd","2_tcpi_unacked","2_tcpi_unsend"],
        reward_cols: List[str]=["reward"],
        src_len=100,
    ):
        pd_data = pd.read_csv(csv_file_dir)

        self.input_data = pd_data[input_cols].values.reshape(-1, src_len, len(input_cols))
        self.reward_data = pd_data[reward_cols].values.reshape(-1, src_len, len(reward_cols))
        
    
    def __getitem__(self, index):
        return self.input_data[index], self.reward_data[index][0]

    def __len__(self):
        return len(self.input_data)

if __name__ == "__main__":
    dataset = A2CDataSet("data/DL2-40M40ms-90M90ms_82_bbr_202310.csv")
    print(dataset[0][0].shape)