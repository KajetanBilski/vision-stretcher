import torch
from torch import nn
from stacker_net.train import MyDataset
from tqdm import tqdm
from torch.nn.functional import pad
from torch.utils.data import TensorDataset, DataLoader
from outpainter_net.net import OutpainterNet
import random

def construct_dataset(image_ds: MyDataset, chunk_size: int = 4, sample_size: int = None):
    xs = []
    ys = []
    if sample_size is None:
        sample_size = len(image_ds)
    for i in tqdm(random.sample(range(len(image_ds)), sample_size)):
        img = image_ds[i]
        img = img[...,-2 * chunk_size:]
        img = pad(img, (0, 0, chunk_size, chunk_size))
        for j in range(0, img.shape[1] - 3 * chunk_size, chunk_size):
            chunk = img[:, j:j+3*chunk_size].unsqueeze(0)
            xs.append(chunk[..., :chunk_size])
            ys.append(chunk[..., chunk_size:2*chunk_size, chunk_size:])
    return TensorDataset(torch.cat(xs, 0), torch.cat(ys, 0))

def prepare_dl(image_ds: MyDataset, chunk_size: int = 4, pin_memory: bool = False):
    ds = construct_dataset(image_ds, chunk_size)
    return DataLoader(ds, batch_size=256, shuffle=True, pin_memory=pin_memory)

@torch.no_grad()
def validate(
    model: OutpainterNet,
    loss_fn: nn.MSELoss,
    val_dl: DataLoader,
    device: torch.device = torch.device('cpu'),
) -> float:
    total_loss = 0
    count = 0
    for X_batch, y_batch in val_dl:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        total_loss += loss
        count += 1
    return total_loss.item() / count


def train(
    model: OutpainterNet,
    loss_fn: nn.MSELoss,
    optimizer: torch.optim.Optimizer,
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs: int = 50,
    device: torch.device = torch.device('cpu'),
    print_metrics: bool = False
):
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        count = 0
        for X_batch, y_batch in tqdm(train_dl):
            optimizer.zero_grad()
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)

            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach()
            count += 1
        
        if print_metrics:
            train_loss = total_loss.item() / count
            val_loss = validate(model, loss_fn, val_dl, device)
            print(f'Epoch {epoch}: train loss = {train_loss:.3f}, val loss = {val_loss:.3f}')
