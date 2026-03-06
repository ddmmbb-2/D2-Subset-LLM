```markdown
# D2 V7 50M Sparse Gate LLM

A small-scale LLM experiment using **Causal Gated-D2 Attention** with sparsity constraints, trained from scratch on Chinese classical text.

This project explores whether **feature gating + sparsity regularization** can produce emergent specialization similar to **Mixture-of-Experts**, but **without explicit routing**.

---

# Model Architecture

The D2 architecture introduces a **Concept Gate** that selectively activates neural features before attention computation.

```

```
            ┌─────────────────────┐
            │      Input Token     │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │      Embedding       │
            │    (Token → Vector)  │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │     LayerNorm        │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │     QKV Projection   │
            │     Linear(d → 3d)   │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │     Concept Gate     │
            │     sigmoid(Wx)      │
            │  Sparse Activation   │
            └──────────┬──────────┘
                       │
            Gate Applied to Q,K
                       │
                       ▼
            ┌─────────────────────┐
            │   Causal Attention   │
            │  (Prefix Computation)│
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │   Projection Layer   │
            │      Linear(d,d)     │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │   Residual Add       │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │        MLP           │
            │   GELU Activation    │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │     Output Layer     │
            │     Vocabulary       │
            └─────────────────────┘
```

```

---

# Key Idea

Instead of routing tokens to explicit experts like **MoE**, D2 uses **continuous gating**:

```

Gate = sigmoid(Wx)
Q = Q * Gate
K = K * Gate

```

With **L1 regularization**, many gate dimensions become inactive:

```

Loss = CrossEntropy + λ * |Gate|

```

This encourages **feature sparsity**, allowing different tokens or domains to activate **different neural subspaces**.

---

# Features

* **Causal Gated-D2 Attention**
* **Sparse feature gating via L1 regularization**
* Residual + projection architecture
* Handles sequences up to **256 tokens**
* Trained from scratch on Chinese classical text
* Approximately **50M parameters**

Observed behavior:

* Classical Chinese tokens activate a **distinct subset of neurons**
* Emergent **hierarchical feature structure**
* Possible **implicit expert specialization**

---

# Training

1. Clone the repository

```

git clone <repo_url>

```

2. Install dependencies

```

pip install torch

```

3. Place your training text files inside

```

data/

```

4. Start training

```

python train_v7.py

```

---

# Inference

Make sure the model and vocab paths are correct.

Example:

```

model: d2_v7_50m.pth
vocab: master_vocab_v7.json

```

Run:

```

python chat.py

```

---

# Model Scale

| Parameter | Value |
|--------|--------|
| Model Size | ~50M |
| Context Length | 256 |
| Architecture | Gated-D2 |
| Training Data | Chinese Classical Text |

---

# Experiment Goal

This project investigates whether **sparse gating mechanisms** can produce:

* specialization similar to **Mixture-of-Experts**
* without router networks
* with **lower computational overhead**

---

# Future Work

Possible improvements:

* Linear Attention scaling
* Rotary Position Embedding
* Larger datasets (Wikipedia, multilingual corpora)
* Implicit expert activation analysis
```


```

甚至可以做成 **GitHub 上會很亮眼的圖版 README**。
那個會看起來真的像一個 **新模型 repo**。
