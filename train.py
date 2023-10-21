from dataset import TrafficDataset
from torch.utils.data import DataLoader, random_split
import torch
from models import ThroughputPredictor
from tqdm import tqdm
import argparse
import os


def train(checkpoint: str, save_path="saved_models/throughput"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TrafficDataset("data/csvfile")

    # split train and test data
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    # define model
    model = ThroughputPredictor(8, 64, 4, 3, 3, 2048).to(device)

    start_epoch = 0
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model"])
        # get epoch from checkpoint
        start_epoch = int(checkpoint.split("_")[-1].split(".")[0])
        save_path = os.path.dirname(checkpoint)

    # define loss function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10000
    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(total=len(train_dataloader), desc="Training Epoch: {}".format(epoch))
        
        # train
        train_loss = 0
        for src, tgt, y in train_dataloader:
            src, tgt, y = src.to(device), tgt.to(device), y.to(device)
            src = src.permute(1, 0, 2)
            tgt = tgt.permute(1, 0, 2)
            output = model(src, tgt)
            optimizer.zero_grad()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            step_loss = loss.item()
            train_loss += step_loss
            pbar.update(1) # manually update progress bar by 1
            pbar.set_postfix(loss=step_loss)

        train_loss /= len(train_dataloader)
        
        pbar.n = 0  # Reset progress
        pbar.total = len(test_dataloader)
        pbar.set_description("Testing Epoch: {}".format(epoch))

        # test
        test_loss = 0
        with torch.no_grad():
            for src, tgt, y in test_dataloader:
                src, tgt, y = src.to(device), tgt.to(device), y.to(device)
                src = src.permute(1, 0, 2)
                tgt = tgt.permute(1, 0, 2)
                output = model(src, tgt)
                step_loss = criterion(output, y).item()
                test_loss += step_loss
                pbar.set_postfix(loss=step_loss)
                pbar.update(1) # manually update progress bar by 1
        
        test_loss /= len(test_dataloader)
        pbar.set_postfix(loss=train_loss,test_loss=test_loss)
        pbar.close()

            
        # save only loss less than 1e-2
        if train_loss < 1e-2:
            save_checkpoint(model, epoch, dataset, save_path)
            break

    save_checkpoint(model, num_epochs, dataset, save_path)


def save_checkpoint(model, epoch, dataset, save_path="saved_models/throughput"):
    mean, std = dataset.get_normalization_params()
    save_path = "saved_models/throughput"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save model and normalization params
    checkpoint = {"model": model.state_dict(), "mean": mean, "std": std}
    torch.save(checkpoint, os.path.join(save_path, "model_{}.pth".format(epoch)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save-path", type=str, default="saved_models/throughput")
    args = parser.parse_args()
    train(checkpoint=args.checkpoint, save_path=args.save_path)
