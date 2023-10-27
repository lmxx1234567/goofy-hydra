import torch
from models import ThroughputPredictor
import argparse
from dataset import TrafficDataset
from torch.utils.data import DataLoader
import time

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    checkpoint_data = torch.load(args.checkpoint)
    model = ThroughputPredictor(8, 256, 4, 3, 3, 2048).to(device)
    model.load_state_dict(checkpoint_data["model"])

    criterion = torch.nn.MSELoss()

    dataset = TrafficDataset("data/test")
    dataloader = DataLoader(dataset, 1, True)

    y_avg = 0
    output_avg = 0
    loss_avg = 0
    for src, tgt, y in dataloader:
        src,tgt,y = src.to(device),tgt.to(device),y.to(device)
        y_avg += y.item()
        # calculate time used
        start = time.time()
        output = model(src)
        end = time.time()
        loss = criterion(y, output)
        output_avg += output.item()
        loss_avg += loss.item()

    y_avg, output_avg, loss_avg = y_avg / \
        len(dataloader), output_avg/len(dataloader), loss_avg/len(dataloader)
    print(y_avg)
    print(output_avg)
    print(loss_avg)
    print(end-start)
