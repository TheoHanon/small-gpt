import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import DataLoader

from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path

import time, json, logging

def _setup_logger(run_path : Path) -> logging.Logger:
    
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    # console
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # file
    fh = logging.FileHandler(run_path / "train.log", mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger



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
        resume : bool = False,
        save_ckpt : bool = True,
    ) -> None:

        self.config = config
        self.data_dir = data_dir
        self.run_dir = run_dir
        self.verbose = verbose

        self.save_ckpt = save_ckpt
        self.resume = resume


    def _load_checkpoint(self, device : torch.device) -> None:

        path = Path(self.run_dir) / "ckpt.pth"
        if not path.exists():
            print(f"=> No checkpoint found at {path}. Starting from scratch.")
            return None
        
        checkpoint = torch.load(path, map_location=device)    
        return checkpoint


    def run(self, *, model : nn.Module, device : torch.device) -> None:

        logger = _setup_logger(Path(self.run_dir))
        logger.info("=== Training started ===")


        model = model.to(device)

        if self.save_ckpt:
            ckpt_path = Path(self.run_dir)
            ckpt_path.mkdir(parents = True, exist_ok = True)
     
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

        start_epoch = 0
        best_val_loss = float("inf")

        if self.resume:
            checkpoint = self._load_checkpoint(device)
            if checkpoint is not None:
                model.load_state_dict(checkpoint["model_state_dict"])
                best_val_loss = checkpoint["val_loss"]
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1

                logger.info(f"[CHECKPOINT] Resumed training from epoch {start_epoch}/{self.config.num_epochs}")

        for ep in range(start_epoch, self.config.num_epochs):
    
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

            if (ep + 50) % 50 == 0 or (ep == self.config.num_epochs - 1):
                model.eval()
                # TODO: val error computed on a single batch only - improve
                with torch.no_grad():
                    val_loss = 0.0
                    x_val, y_val = next(val_loader)
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    with torch.autocast(device_type = device.type, dtype = torch.bfloat16):
                        logits_val = model(x_val)
                        val_loss = F.cross_entropy(logits_val.view(-1, logits_val.size(-1)), y_val.view(-1)) 


                if val_loss.item() < best_val_loss and self.save_ckpt:
                    best_val_loss = val_loss.item()

                    checkpoint = {
                        "model_state_dict" : model.state_dict(),
                        "optimizer_state_dict" : optimizer.state_dict(),
                        "epoch" : ep,
                        "val_loss" : val_loss.item(),
                    }
                    torch.save(checkpoint, ckpt_path / "ckpt.pth")

                    with open(ckpt_path / "config.json", "w") as f:
                        json.dump(model.config.model_dump(), f, indent = 4)


                    logger.info(
                        f"[CHECKPOINT] Saved best model at epoch {ep+1}/{self.config.num_epochs} "
                        f"| val_loss: {val_loss.item():.4f}"
                    )
                        
                model.train()
                logger.info(
                    f"[VALIDATION] === Eval at epoch {ep+1} / {self.config.num_epochs} | Val Loss : {val_loss.item():.4f} ==="
                )

            if ep % 20 == 0:
                logger.info(
                    "[EVAL] "
                    f"[{ep+1}/{self.config.num_epochs}] "
                    f"Loss: {loss.item():.4f} | dt: {dt:.2f}ms | "
                    f"{(self.config.batch_size * self.config.ctx_size) / dt:.2f} toks/sec"
                )
            

        logger.info("=== Training finished ===")






    

    

