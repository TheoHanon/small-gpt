import torch
import numpy as np

from pathlib import Path

from typing import Literal

class DataLoader:

    def __init__(
            self,
            B : int,
            ctx_size : int,
            dir_path : str,
            *,
            split : Literal["train", "val"] = "train", 
    ) -> None:
        
        path = (Path(__file__).cwd() / dir_path / (split + ".bin")).resolve()
        self.data = torch.from_numpy(np.fromfile(path, dtype = np.uint16)).to(torch.long)
    

        self.B = B
        self.ctx_size = ctx_size
        self.stride = self.B * (self.ctx_size)
        self._len = len(self.data)
        self.n_batches = (self._len - 1) // self.stride
        self.split = split

    def __iter__(self) :
        self.pos = 0
        return self
    
    def __next__(self):
        """
        Unused tokens here !
        """
        
        if self._len <= self.pos + self.stride:
            raise StopIteration

        
        buf = self.data[self.pos : self.pos + self.stride + 1]
        self.pos += self.stride
        
        x = buf[:-1].view(self.B, self.ctx_size)
        y = buf[1:].view(self.B, self.ctx_size)
        
        return x, y
        


