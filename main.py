from train import TrainingConfig, Training
from transformers import AutoTokenizer
from model.gpt import GPT, GPTConfig

import torch



def main(train_config, model, device):


    trainer = Training(
        config = train_config,
        device = device,
    )

    trainer.run(model)
    model.eval()




if __name__ == "__main__":

    torch.manual_seed(42)


    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("=> Using MPS")
    else:
        device = torch.device("cpu")
        print("=> Using CPU")

    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        use_fast = False
    )

    train_config = TrainingConfig(
        num_epochs= 5000,
        batch_size = 4,
        seq_length= 512,
    )

    gpt_config = GPTConfig(
        d_voc = tokenizer.vocab_size,
        d_emb = 128,
        n_layer = 4,
        max_ctx = 512,
        num_heads = 16,
    )

    model = GPT(
        config = gpt_config
    ).to(device)

    # model.compile()

    main(train_config, model, device)

    q = torch.tensor(tokenizer("My name is")["input_ids"], dtype = torch.long).unsqueeze(0).repeat(5, 1).to(device)
    out= model.generate(q, max_new_tokens = 20)
    resp = tokenizer.batch_decode(out, skip_special_tokens = True)

    print("\n")
    print("="*100)
    print("SAMPLES")
    print("="*100)
    for i, res in enumerate(resp):
        print(f"{i+1}. ", res)
    print("="*100)
    


