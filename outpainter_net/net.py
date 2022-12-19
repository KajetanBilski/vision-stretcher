import torch
from torch import nn
from torch.nn.functional import pad
from math import ceil

class OutpainterNet(nn.Module):
    def __init__(self, chunk_size: int = 4) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d((3, 1), stride=(3, 1)),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(self.chunk_size),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=self.chunk_size),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor):
        y = self.net[:6](x).view(x.shape[0], 16)
        y = self.net[6](y).view(x.shape[0], 16, 1, 1)
        y = self.net[7:](y)
        return y
    
    def image_forward(self, img: torch.Tensor):
        img = img[...,-self.chunk_size:]
        img = pad(img, (0, 0, self.chunk_size, self.chunk_size))
        x = []
        if img.shape[1] % self.chunk_size:
            for i in range(0, img.shape[1] - 2 * self.chunk_size - 1, self.chunk_size):
                x.append(img[:, i:i+3*self.chunk_size])
            x.append(img[:, -3*self.chunk_size:])
        else:
            for i in range(0, img.shape[1] - 2 * self.chunk_size, self.chunk_size):
                x.append(img[:, i:i+3*self.chunk_size])
        x = torch.stack(x)
        preds = self.forward(x)
        preds = preds.transpose(0, 1).reshape(3, -1, self.chunk_size)
        if img.shape[1] % self.chunk_size:
            preds = torch.cat((preds[:, :img.shape[1] - self.chunk_size], preds[:, -self.chunk_size:]), 1)
        return preds

    def image_outpaint(self, img: torch.Tensor, length: int):
        if length <= 0:
            return
        repeats = ceil(length / self.chunk_size)
        preds = [self.image_forward(img)]
        for _ in range(1, repeats):
            preds.append(self.image_forward(preds[-1]))
        preds = torch.cat(preds, dim=2)
        if preds.shape[2] > length:
            preds = preds[..., :length]
        return preds
