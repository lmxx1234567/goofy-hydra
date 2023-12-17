from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class A2CDataSet(Dataset):
    def __init__(
        self,
        csv_file_dir,
        state_cols: List[str] = [
            "packet_count",
            "1_TXrate",
            "1_tcpi_lost",
            "1_tcpi_reordering",
            "1_tcpi_retransbyte",
            "1_tcpi_retranscount",
            "1_tcpi_rtt",
            "1_tcpi_snd_cwnd",
            "1_tcpi_unacked",
            "1_tcpi_unsend",
            "2_TXrate",
            "2_tcpi_lost",
            "2_tcpi_reordering",
            "2_tcpi_retransbyte",
            "2_tcpi_retranscount",
            "2_tcpi_rtt",
            "2_tcpi_snd_cwnd",
            "2_tcpi_unacked",
            "2_tcpi_unsend",
        ],
        reward_cols: List[str] = ["reward"],
        src_len=100,
    ):
        pd_data = pd.read_csv(csv_file_dir)
        self.data_length = pd_data.shape[0] // src_len

        state_data = pd_data[state_cols].values[: self.data_length * src_len]
        reward_data = pd_data[reward_cols].values[: self.data_length * src_len]

        self.state_data = state_data.reshape(-1, src_len, len(state_cols))
        self.reward_data = reward_data.reshape(-1, src_len, len(reward_cols))

    def __getitem__(self, index):
        return (
            torch.Tensor(self.state_data[index]),
            torch.Tensor(self.state_data[index + 1]),
            torch.Tensor(self.reward_data[index + 1]),
        )

    def __len__(self):
        return self.data_length - 1


if __name__ == "__main__":
    dataset = A2CDataSet("data/DL2-40M40ms-90M90ms_82_bbr_202310.csv")
    print(len(dataset))
    print(dataset[0][0].shape)
