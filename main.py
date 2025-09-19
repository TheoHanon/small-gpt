from train import TrainingConfig, Training
from transformers import AutoTokenizer
from model.gpt import GPT, GPTConfig

import torch
import argparse, yaml, os, pathlib


def load_data(data_dir, force_reload = False):
    
    p = pathlib.Path(data_dir)

    if not p.exists():
        raise RuntimeError(f"Data directory '{data_dir}' does not exist.")

    if not (p / "load.py").exists() :
        raise RuntimeError(f"Data directory '{data_dir}' does not contain 'load.py' script.")
        
    if not ((p / "val.bin").exists() and (p / "train.bin").exists()) or force_reload:
        print(f"=> Tokenized data not found in '{data_dir}'. Downloading and tokenizing ...")
        os.system(f"python {p / 'load.py'}")
        print("=> Done.")
        
    return 
    

def pick_device(device):

    if device == "mps" and torch.mps.is_available():
        print("=> Using MPS")
        print(f"=> Number of process available : {torch.mps.device_count()}")
        return torch.device("mps")

    if device == "cuda" and torch.cuda.is_available():
        print("=> Using MPS")
        print(f"=> Number of process available : {torch.cuda.device_count()}")
        return torch.device("cuda")
    
    
    print("=> Using CPU")
    return torch.device("cpu")


def parse_config():

    parser = argparse.ArgumentParser(
        prog = "TrainGPT",
        description = "Train the GPT model given the specifications.",
    )


    parser.add_argument("--eval", action = "store_true",
                        help = "Whether to eval and not train the model.", default=False)
    

    parser.add_argument("-c", "--config", type = str, default="config.yaml", 
                        help = "Path to your YAML config file.")
    
    parser.add_argument("--ckpt-dir", type = str, default = "runs/exp1", 
                        help = "Directory to save logs, checkpoints.")
    
    parser.add_argument("--save-ckpt", action = "store_true", default = True,
                        help = "Whether to save the model checkpoint.")

    parser.add_argument("--force-reload", action = "store_true",
                        help = "Force reloading the dataset (redownload and retokenize).", default=False)
    
    parser.add_argument("--compile", action = "store_true",
                        help = "Do not compile the model (if PyTorch version >= 2.0) or using mps.", default=False)

    parser.add_argument("--seed", type = int, default = 42,
                        help = "Random seed for reproducibility.")
    
    parser.add_argument("--device", type = str, default = "mps",
                        help = "Device to use (mps, cuda, cpu).")
    
    parser.add_argument("--data-dir", type = str, default = "data/shakespeare",
                        help = "Directory containing the dataset.")

    parser.add_argument("--resume", action = "store_true", default = False,
                        help = "Whether to resume training from a checkpoint.")
    
    parser.add_argument("--verbose", type = int, default = 1, choices = [0, 1],
                        help = "Verbosity level: 0 (silent), 1 (logs).")

    args = parser.parse_args()

    return args

    
def main():

    args = parse_config()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    load_data(args.data_dir, force_reload= args.force_reload)

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["tokenizer"]["name"],
        use_fast = False
    )

    train_config = TrainingConfig(**cfg["training"])
    gpt_config = GPTConfig(**cfg["model"])

    model = GPT(
        config = gpt_config
    )

    if args.compile:
        model.compile()

    if not args.eval:

        trainer = Training(
            config = train_config,
            run_dir = args.ckpt_dir,
            data_dir = args.data_dir,
            resume = args.resume,
            save_ckpt = args.save_ckpt,
            verbose = args.verbose
        )

        trainer.run(
            model = model, 
            device = device
        )

    else:
        
        path = pathlib.Path(args.ckpt_dir) / "ckpt.pth"
        if not path.exists():
            raise RuntimeError(f"No checkpoint found at '{path}'.")

        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"=> Loaded checkpoint from '{path}' (epoch {checkpoint['epoch']}, val loss {checkpoint['val_loss']:.4f})")

    model = model.to(device)
    model.eval()
    q = torch.tensor(tokenizer("I love ")["input_ids"], dtype = torch.long).unsqueeze(0).repeat(5, 1).to(device)
    out= model.generate(q, max_new_tokens = 20)
    resp = tokenizer.batch_decode(out, skip_special_tokens = True)

    print("\n")
    print("="*100)
    print("SAMPLES")
    print("="*100)
    for i, res in enumerate(resp):
        print(f"{i+1}. ", res)
    print("="*100)



if __name__ == "__main__":

    main()


