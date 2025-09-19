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
    distributed : bool = Field(default = False)
    learning_rate : float = Field(default=3e-4, gt = 0)
    batch_size : int = Field(default = 4, ge = 1)
    ctx_size : int = Field(default = 64, ge=0)
    
class Training:

    def __init__(
        self,
        *,
        config : TrainingConfig,
        run_dir : str,
        data_dir : str,
        verbose : Literal[0, 1] = 1,
    ) -> None:

        self.config = config
        self.data_dir = data_dir
        self.run_dir = run_dir
        self.verbose = verbose


    @classmethod
    def resume_from(cls):
        pass


    def run(self, *, model : nn.Module, device : torch.device):

        train_loader = DataLoader(
            B = self.config.batch_size,
            ctx_size=self.config.ctx_size,
            dir_path = self.data_dir,
            split = "train"
        ).__iter__()


        val_loader = DataLoader(
            B = self.config.batch_size*4,
            ctx_size=self.config.ctx_size,
            dir_path = self.data_dir,
            split = "val"
        ).__iter__()
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr = self.config.learning_rate,
        )

        for ep in range(self.config.num_epochs):
    
            t0 = time.time()
            
            optimizer.zero_grad()

            x, y = next(train_loader)
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type = device.type, dtype = torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            loss.backward()
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()

            elif device.type == "mps":
                torch.mps.synchronize()
            
            t1 = time.time()
            dt = 1000*(t1 - t0)

            if (self.verbose >= 1 and (ep + 50) % 50 == 0) or (ep == self.config.num_epochs - 1):
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    x_val, y_val = next(val_loader)
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    with torch.autocast(device_type = device.type, dtype = torch.bfloat16):
                        logits_val = model(x_val)
                        val_loss = F.cross_entropy(logits_val.view(-1, logits_val.size(-1)), y_val.view(-1)) 
                        
                model.train()
                print(f"\n=== Eval at epoch {ep+1} / {self.config.num_epochs} | Val Loss : {val_loss.item():.4f} ===\n")
            
            if self.verbose >= 1 and ep % 20 == 0:
                print(f"{ep+1} / {self.config.num_epochs} | Loss : {loss.item():.4f} | dt : {dt:.2f}ms | {(self.config.batch_size * self.config.ctx_size) / dt:.2f} toks/sec")
    






    

    

