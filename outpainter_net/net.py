from typing import List, Tuple
import torch
from torch import nn
from torch.nn.functional import pad
from math import ceil, floor

class OutpainterNet(nn.Module):
    def __init__(self, chunk_size: int = 4) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((3, 1), stride=(3, 1)),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(self.chunk_size),
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
        batch_size = img.shape[0]
        height = img.shape[2]
        img = img[...,-self.chunk_size:]
        img = pad(img, (0, 0, self.chunk_size, self.chunk_size))
        x = []
        if img.shape[-2] % self.chunk_size:
            for i in range(0, img.shape[-2] - 3 * self.chunk_size + 1, self.chunk_size):
                x.append(img[:, :, i:i+3*self.chunk_size])
            x.append(img[:, :, -3*self.chunk_size:])
        else:
            for i in range(0, img.shape[-2] - 2 * self.chunk_size, self.chunk_size):
                x.append(img[:, :, i:i+3*self.chunk_size])
        x = torch.stack(x)
        preds = self.forward(x.reshape(-1, 3, 3*self.chunk_size, self.chunk_size))
        preds = preds.reshape(img.shape[0], -1, 3, self.chunk_size, self.chunk_size).transpose(1, 2).reshape(img.shape[0], 3, -1, self.chunk_size)
        if height % self.chunk_size:
            preds = torch.cat((preds[..., :height - self.chunk_size, :], preds[..., -self.chunk_size:, :]), -2)
        return preds

    def outpaint_right(self, img: torch.Tensor, length: int):
        if length <= 0:
            return img
        repeats = ceil(length / self.chunk_size)
        preds = [self.image_forward(img)]
        for _ in range(1, repeats):
            preds.append(self.image_forward(preds[-1]))
        preds = torch.cat(preds, dim=-1)
        if preds.shape[-1] > length:
            preds = preds[..., :length]
        return torch.cat((img, preds), dim=-1)

    def image_outpaint(self, img: torch.Tensor, target_size: List[int]):
        sizes = [
            floor((target_size[1] - img.shape[-2]) / 2), ceil((target_size[1] - img.shape[-2]) / 2),
            floor((target_size[0] - img.shape[-1]) / 2), ceil((target_size[0] - img.shape[-1]) / 2)
        ]
        img = self.outpaint_right(img.rot90(-1, dims=[-2, -1]), sizes[0])
        img = self.outpaint_right(img.rot90(2, dims=[-2, -1]), sizes[1])
        img = self.outpaint_right(img.rot90(1, dims=[-2, -1]), sizes[2])
        img = self.outpaint_right(img.rot90(2, dims=[-2, -1]), sizes[3])
        return img

    def perform_outpaint(self, img, target_size: List[int], device: torch.device):
        img = torch.tensor(img / 255, dtype=torch.float32).transpose(0, 2).unsqueeze(0).to(device)
        return (self.image_outpaint(img, target_size).squeeze().transpose(0, 2).cpu().detach().numpy() * 255).astype(int)
