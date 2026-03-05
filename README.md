
# D2 V7 50M Sparse Gate LLM

A small-scale LLM experiment using causal Gated-D2 attention with sparsity constraints, trained from scratch on Chinese classical text.

## Features

* **Causal Gated-D2 Attention** with projection and residual connections
* **L1 sparsity regularization** on concept gates to prune unnecessary features
* Handles sequences up to **256 tokens**
* Trained from scratch with a **50M parameter model**
* Demonstrates **hierarchical feature extraction** and interpretable sparsity patterns

## Usage

1. Clone the repository
2. Make sure Python 3.10+ is installed, along with `torch`
3. Place your training text files in the `data` folder
4. Train the model:

```bash
python train_v7.py
```

5. **Confirm model and vocab paths**
   In `chat.py`, ensure the model file (e.g., `d2_v7_50m.pth`) and vocab file (e.g., `master_vocab_v7.json`) paths match the names you uploaded. The script will fail to load the model if these are incorrect.

6. Run inference:

```bash
python chat.py
```


