import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import Tensor


class SquarePad(torch.nn.Module):
    def __init__(self, fill: int = 0, padding_mode: str = "constant"):
        super().__init__()
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img: Tensor):
        _, h, w = img.shape
        s = np.max([w, h])
        padding = [
            int(np.ceil((s - w) / 2)),
            int(np.ceil((s - h) / 2)),
            int(np.floor((s - w) / 2)),
            int(np.floor((s - h) / 2)),
        ]
        return F.pad(img, padding, self.fill, self.padding_mode)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fill={self.fill}, padding_mode={self.padding_mode})"
