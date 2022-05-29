from typing import Any, Callable, Optional
from torch import nn
import torch
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from .net import StackerNet
import torchvision
from PIL import Image

class MyDataset(torchvision.datasets.VisionDataset):

    def __init__(self, root: str, paths: list[str], transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.paths = paths

    def __getitem__(self, index: int) -> Any:
        item = Image.open(self.paths[index])
        if self.transform is not None:
                item = self.transform(item)
        return item
    
    def __len__(self) -> int:
        return len(self.paths)

@torch.no_grad()
def validate(
    model: StackerNet,
    loss_fn: nn.MSELoss,
    val_dl: DataLoader,
    pred_len: int,
    device: torch.device = torch.device('cpu'),
) -> float:
    total_loss = 0
    count = 0
    for X_batch in val_dl:
        X_batch = X_batch.to(device)
        y_batch = X_batch[..., -pred_len:]
        X_batch = X_batch[..., :-pred_len]
        y_pred = model.predict(X_batch, length=pred_len)
        loss = loss_fn(y_pred, y_batch)
        total_loss += loss
        count += 1
    return total_loss.item() / count


def train(
    model: StackerNet,
    loss_fn: nn.MSELoss,
    optimizer: torch.optim.Optimizer,
    train_dl: DataLoader,
    val_dl: DataLoader,
    pred_len: int,
    epochs: int = 50,
    device: torch.device = torch.device('cpu'),
    print_metrics: bool = False
):
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        count = 0
        for X_batch in tqdm(train_dl):
            optimizer.zero_grad()
            X_batch = X_batch.to(device)
            y_batch = X_batch[..., -pred_len:]
            X_batch = X_batch[..., :-pred_len]

            y_pred = model.predict(X_batch, length=pred_len)

            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach()
            count += 1
        
        if print_metrics:
            train_loss = total_loss.item() / count
            val_loss = validate(model, loss_fn, val_dl, pred_len, device)
            print(f'Epoch {epoch}: train loss = {train_loss:.3f}, val loss = {val_loss:.3f}')


@torch.no_grad()
def validate_autoencoder(
    model: StackerNet,
    loss_fn: nn.MSELoss,
    val_dl: DataLoader,
    pred_len: int,
    device: torch.device = torch.device('cpu'),
) -> float:
    total_loss = 0
    count = 0
    for X_batch in val_dl:
        X_batch = X_batch.to(device)
        X_batch = X_batch.transpose(1, 2).reshape(-1, 3, 256)
        X_pred = model.decoder(model.encoder(X_batch)) * 255
        loss = loss_fn(X_pred, X_batch)
        total_loss += loss
        count += 1
    return total_loss.item() / count


def train_autoencoder(
    model: StackerNet,
    loss_fn: nn.MSELoss,
    optimizer: torch.optim.Optimizer,
    train_dl: DataLoader,
    val_dl: DataLoader,
    pred_len: int,
    epochs: int = 50,
    device: torch.device = torch.device('cpu'),
    print_metrics: bool = False
):
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        count = 0
        for X_batch in tqdm(train_dl):
            optimizer.zero_grad()
            X_batch = X_batch.to(device)
            X_batch = X_batch.transpose(1, 2).reshape(-1, 3, 256)

            X_pred = model.decoder(model.encoder(X_batch)) * 255

            loss = loss_fn(X_pred, X_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach()
            count += 1
        
        if print_metrics:
            train_loss = total_loss.item() / count
            val_loss = validate_autoencoder(model, loss_fn, val_dl, pred_len, device)
            print(f'Epoch {epoch}: train loss = {train_loss:.3f}, val loss = {val_loss:.3f}')
