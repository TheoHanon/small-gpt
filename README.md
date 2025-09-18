# small-gpt

A minimal GPT-style transformer model implemented from scratch in PyTorch. Under development. 

This code is an independent implementation of a GPT-like model. For the moment the model is close to GPT-2 but will be subject to modifications. The excellent work in [nanoGPT](https://github.com/karpathy/nanoGPT) served as guidelines during development.

---

## Features

- GPT-like architecture: transformer blocks with self-attention and feed-forward layers  
- Training and inference pipeline included  
- Implemented to run on Apple Silicon MPS acceleration  

---

## Getting Started

1. **Clone the repository**  
   ```bash
   git clone https://github.com/TheoHanon/small-gpt.git
   cd small-gpt

2. **Install dependencies**
   Make sure you have Python and PyTorch installed.

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data**
   Place your training data in the `data/` folder. Some example data sets are provided (e.g. a Shakespeare text dataset).

---

## Status

This project is a work in progress.
Project is functional (model, training loop, inference) but is subject to further refinements and improvements.

---

## Contact

For any questions, suggestions, or contributions:
Theo Hanon — [GitHub: TheoHanon](https://github.com/TheoHanon)

