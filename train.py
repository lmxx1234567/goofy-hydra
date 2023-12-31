from dataset import TrafficDataset
from torch.utils.data import DataLoader, random_split
import torch
from models import ThroughputPredictor, ThroughputPredictorLiner
from tqdm import tqdm
import argparse
import os
from torch.utils.tensorboard import SummaryWriter


def train(
    checkpoint: str,
    save_path="saved_models/throughput",
    pretrained_model_path="saved_models/pretrain/transformer_1.pth",
):
    # init writer
    writer = SummaryWriter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TrafficDataset("data/chgebw")

    # define model
    model = ThroughputPredictor(8, 512, 8, 6, 6, 2048)
    if pretrained_model_path:
        pretrained_data = torch.load(pretrained_model_path)
        if isinstance(model, torch.nn.DataParallel):
            model.module.state_feature_extractor.load_state_dict(
                pretrained_data["model"]
            )
        else:
            model.state_feature_extractor.load_state_dict(pretrained_data["model"])
        model.state_feature_extractor.set_grad_requires(False)
    # model = ThroughputPredictor(8, 256, 4, 3, 3, 2048)
    # model = ThroughputPredictorLiner(8, 256)
    # If multiple GPUs are available, use DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)

    # split train and test data
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    batch_size = 8
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size * torch.cuda.device_count(), shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    start_epoch = 0
    if checkpoint:
        checkpoint_data = torch.load(checkpoint)
        # Careful handling of DataParallel model structure
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint_data["model"])
        else:
            model.load_state_dict(checkpoint_data["model"])
        # get epoch from checkpoint
        start_epoch = int(checkpoint.split("_")[-1].split(".")[0])
        save_path = os.path.dirname(checkpoint)

    # define loss function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 1000
    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(
            total=len(train_dataloader), desc="Training Epoch: {}".format(epoch)
        )

        # train
        train_loss = 0
        for src, tgt, y in train_dataloader:
            src, tgt, y = src.to(device), tgt.to(device), y.to(device)
            output = model(src)
            optimizer.zero_grad()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            step_loss = loss.item()
            train_loss += step_loss
            pbar.update(1)  # manually update progress bar by 1
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
                output = model(src, tgt)
                step_loss = criterion(output, y).item()
                test_loss += step_loss
                pbar.set_postfix(loss=step_loss)
                pbar.update(1)  # manually update progress bar by 1

        test_loss /= len(test_dataloader)
        pbar.set_postfix(loss=train_loss, test_loss=test_loss)
        pbar.close()

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Test", test_loss, epoch)

        # save every 50 epoch
        if epoch % 50 == 0:
            save_checkpoint(model, epoch, dataset, save_path)

        # save only loss less than 1e-4
        if train_loss < 1e-4:
            save_checkpoint(model, epoch, dataset, save_path)
            return

    save_checkpoint(model, num_epochs, dataset, save_path)


def save_checkpoint(model, epoch, dataset, save_path="saved_models/throughput"):
    mean, std = dataset.get_normalization_params()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # If the model is an instance of DataParallel, get the 'module' which contains the original model
    if isinstance(model, torch.nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model

    # save model and normalization params
    checkpoint = {"model": model_to_save.state_dict(), "mean": mean, "std": std}
    torch.save(checkpoint, os.path.join(save_path, "model_{}.pth".format(epoch)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save-path", type=str, default="saved_models/throughput")
    args = parser.parse_args()
    train(checkpoint=args.checkpoint, save_path=args.save_path)
