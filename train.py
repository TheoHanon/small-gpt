import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import DataLoader

from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path
import time


class TrainingConfig(BaseModel):
    num_epochs : int = Field(default = 10, ge = 1)
    device : Literal["cpu", "mps", "cuda"] = "cpu"
    distributed : bool = False
    save_dir : Path = Field(default=Path("."))
    learning_rate : float = Field(default=3e-4, gt = 0)
    batch_size : int = Field(default = 4, ge = 1)
    seq_length : int = Field(default = 64, ge=0)
    data_path : str = Field(default = "data/shakespeare")

    
class Training:

    def __init__(
        self,
        *,
        config : TrainingConfig,
        device,
    ) -> None:

        self.config = config

        self.data_train = DataLoader(
            B = config.batch_size,
            seq_length=config.seq_length,
            dir_path = config.data_path,
            split = "train"
        )
        

        self.optim = None
        self.device = device


    @classmethod
    def resume_from(cls):
        pass


    def run(self, model : nn.Module):
        
        self.optim = torch.optim.AdamW(
            model.parameters(),
            lr = self.config.learning_rate
        )

        train_loader = iter(self.data_train)

        for ep in range(self.config.num_epochs):

            t0 = time.time()
            
            self.optim.zero_grad()

            x, y = next(train_loader)
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(device_type = self.device.type, dtype = torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            loss.backward()
            self.optim.step()

            torch.mps.synchronize()

            t1 = time.time()
            dt = 1000*(t1 - t0)
            
            if ep % 20 == 0:
                print(f"{ep+1} / {self.config.num_epochs} | Loss : {loss.item():.4f} | dt : {dt:.2f}ms | {(self.config.batch_size * self.config.seq_length) / (t1 - t0):.2f} toks/sec")





    

    

