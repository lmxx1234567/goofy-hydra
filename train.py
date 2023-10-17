from dataset import TrafficDataset
from torch.utils.data import DataLoader, random_split
import torch
from models import ThroughputPredictor
from tqdm import tqdm
import argparse
import os


def train(checkpoint: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TrafficDataset("data/csvfile")

    # split train and test data
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)

    # define model
    model = ThroughputPredictor(8, 256, 2, 2, 2, 2048).to(device)

    start_epoch = 0
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))
        # get epoch from checkpoint
        start_epoch = int(checkpoint.split("_")[-1].split(".")[0])

    # define loss function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    num_epochs = 10000
    # train
    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (src, tgt, y) in pbar:
            src, tgt, y = src.to(device), tgt.to(device), y.to(device)
            src = src.permute(1, 0, 2)
            tgt = tgt.permute(1, 0, 2)
            output = model(src, tgt)
            optimizer.zero_grad()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            pbar.set_description("Epoch: {}".format(epoch, loss.item()))
            pbar.set_postfix(loss=loss.item())

        # save only loss less than 1e-2
        if loss.item() < 1e-2:
            # save checkpoint
            save_path = "saved_models/throughput"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(
                model.state_dict(), save_path + "/throughput_{}.pth".format(epoch)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    train(checkpoint=args.checkpoint)
