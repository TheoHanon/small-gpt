# 
# Code adapted from https://github.com/karpathy/nanoGPT/tree/master/data/shakespeare
#

import os, argparse, pathlib
import requests
from transformers import AutoTokenizer
import numpy as np


def main():
    parser = argparse.ArgumentParser(prog = "TokenizeDataset",
                                    description="Save the tokenize dataset using the provided tokenizer.")

    parser.add_argument(
        "-t", "--tokenizer", 
        default = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        type = str,
        help = "Provide the tokenizer's HuggingFace link."
    )

    args = parser.parse_args()

    # download the tiny shakespeare dataset
    input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()


    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    try :
        enc = AutoTokenizer.from_pretrained(args.tokenizer)

    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer '{args.tokenizer}': {e}")

    train_ids = enc(train_data)["input_ids"]
    val_ids = enc(val_data)["input_ids"]


    print(f"=> train has {len(train_ids):,} tokens")
    print(f"=> val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))


if __name__ == "__main__":

    main()