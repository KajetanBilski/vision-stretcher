from copy import copy
from turtle import forward
from cv2 import transpose
import torch
from torch import nn


class StackerNet(nn.Module):
    def __init__(self, pixel_length = 256, sequence_length = 32, channels = 3) -> None:
        super().__init__()
        if pixel_length != 256 or channels != 3:
            raise NotImplementedError()
        self.pixel_length = pixel_length
        self.sequence_length = sequence_length
        self.channels = channels
        self.embedding_length = pixel_length * channels

        # self.encoder = nn.Sequential(
        #     nn.Conv1d(3, 12, kernel_size=7, stride=1, padding=3),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=4),
        #     nn.Conv1d(12, 48, kernel_size=7, stride=1, padding=3),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=4),
        #     nn.Conv1d(48, 192, kernel_size=7, stride=1, padding=3),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=4),
        #     nn.Conv1d(192, 768, kernel_size=4)
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose1d(768, 192, kernel_size=4),
        #     nn.ReLU(),
        #     nn.MaxUnpool1d(kernel_size=4),
        #     nn.ConvTranspose1d(192, 48, kernel_size=7, padding=3),
        #     nn.ReLU(),
        #     nn.MaxUnpool1d(kernel_size=4),
        #     nn.ConvTranspose1d(48, 12, kernel_size=7, padding=3),
        #     nn.ReLU(),
        #     nn.MaxUnpool1d(kernel_size=4),
        #     nn.ConvTranspose1d(12, 3, kernel_size=7, stride=1, padding=3),
        #     nn.Sigmoid()
        # )
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 12, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(12, 48, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(48, 192, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(192, 768, kernel_size=4)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(768, 192, kernel_size=4),
            nn.ReLU(),
            nn.ConvTranspose1d(192, 48, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(48, 12, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(12, 3, kernel_size=8, stride=4, padding=2),
            nn.Sigmoid()
        )
        self.predictor = nn.LSTM(self.embedding_length, self.embedding_length, num_layers=3, batch_first=True)
    
    def forward(self, x: torch.Tensor):
        # x.shape = (N, C, L, P)
        x = x.transpose(1, 2).reshape(-1, self.pixel_length, 3)
        # x.shape = (N * L, C, P)
        z = self.encoder(x).view(-1, self.sequence_length, self.embedding_length)
        # z = (N, L, C * P)
        z = self.predictor(z)[0][:, -1, :]
        return self.decoder(z.unsqueeze(-1))
    
    def sequential_forward(self, x: torch.Tensor, length = None):
        if not length:
            length = self.sequence_length
        embeddings = self.encoder(x.transpose(1, 2).reshape(-1, self.channels, self.pixel_length)).view(-1, self.sequence_length, self.embedding_length)
        z_pred = torch.zeros((embeddings.shape[0], length, self.embedding_length), dtype=embeddings.dtype, device=embeddings.device)
        outputs, (h, c) = self.predictor(embeddings)
        z_pred[:, 0, :] = outputs[:, -1, :]
        for i in range(1, length):
            outputs, (h, c) = self.predictor(outputs[:, -1:, :], (h, c))
            z_pred[:, i, :] = outputs[:, -1, :]
        y = self.decoder(z_pred.view(-1, z_pred.shape[-1]).unsqueeze(-1))
        return y.view(-1, length, *y.shape[-2:]).transpose(1, 2)
    
    def predict(self, x: torch.Tensor, direction = 0, length = None):
        if not (direction & 1):
            x = x.transpose(2, 3)
        if direction & 2:
            x = x.flip(2)
        preds = (self.sequential_forward(x[..., -self.sequence_length:, :], length) * 255.).round()
        if direction & 2:
            preds = preds.flip(2)
        if not (direction & 1):
            preds = preds.transpose(2, 3)
        return preds

        