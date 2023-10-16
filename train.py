from dataset import TrafficDataset
from torch.utils.data import DataLoader,random_split
import torch
from models import ThroughputPredictor
from tqdm import tqdm
import argparse

def train(checkpoint:str):
    dataset = TrafficDataset('data/100M50ms_bbr_Dldataset_202310.csv')

    # split train and test data
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)

    # define model
    model = ThroughputPredictor(5, 1, 2, 2,2048)
    
    start_epoch = 0
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))
        # get epoch from checkpoint
        start_epoch = int(checkpoint.split('_')[-1].split('.')[0])

    # define loss function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    # train
    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (src, tgt) in pbar:
            optimizer.zero_grad()
            src = src.permute(1, 0, 2)
            tgt = tgt.permute(1, 0, 2)
            output = model(src, tgt)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            pbar.set_description('Epoch: {}; Loss: {:.4f}'.format(epoch, loss.item()))

        # save checkpoint
        torch.save(model.state_dict(), 'saved_models/throughput/epoch_{}.pth'.format(epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    train(checkpoint=args.checkpoint)