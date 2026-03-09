# D2-V11: Wave-Interference Implicit MoE Linear Attention LLM

**波干涉隱式專家線性注意力語言模型**

A lightweight experimental Large Language Model exploring **Causal Linear Attention** combined with **Wave-Interference Gating**, inspired by qllm2.

This model is trained **from scratch** on a mixed corpus including **Wikipedia**, **Classical Chinese**, and **Python code**, aiming to study whether **phase coherence and sparsity regularization** can produce **emergent specialization** similar to Mixture-of-Experts (MoE) — but **without explicit routing** and while maintaining **O(N) linear complexity**.

這是一個輕量級 LLM 架構實驗，探索由 **波干涉門控 (Wave-Interference Gating)** 驅動的 **因果線性注意力 (Causal Linear Attention)**。
模型使用 **混合語料（維基百科、古文、Python 程式碼）從零開始訓練**。

本專案的核心問題是：

> 在 **不使用明確 Expert Router** 的情況下，
> **相位干涉 + 稀疏正則化** 是否能自然湧現出類似 **Mixture-of-Experts** 的專家分工？

同時保持 **O(N)** 的線性計算複雜度。

---

# 🧠 Key Idea / 核心概念

Traditional MoE architectures **explicitly route tokens** to different experts.

D2-V11 instead treats **tokens as waves**.

Features are projected into two latent spaces:

* **Semantic Bank** – captures semantic amplitude and phase
* **Context Bank** – captures contextual amplitude and phase

Their **phase coherence** determines whether features are **amplified or cancelled** through wave interference.

不同於傳統 MoE 透過 Router 強制分配 token，
D2-V11 將 **Token 視為波 (Wave Representation)**。

模型將特徵投射到兩個潛在空間：

* **語意庫 (Semantic Bank)**
* **上下文庫 (Context Bank)**

透過兩者的 **相位差 (phase difference)** 形成 **波干涉效應**，
從而自然地 **放大或抑制特徵維度**。

---

# Mathematical Core / 數學核心

[
\text{Interference} =
A_{sem} \cdot A_{ctx} \cdot
\cos(\theta_{sem} - \theta_{ctx})
]

Where:

* (A) represents feature amplitude
* (\theta) represents feature phase

---

# Implementation Sketch / 實作邏輯

```python
# 1. Extract amplitude and phase
sem_amp, sem_phase = Semantic_Bank(x)
ctx_amp, ctx_phase = Context_Bank(x)

# 2. Wave interference
interference = sem_amp * ctx_amp * torch.cos(sem_phase - ctx_phase)

# 3. Soft gating
gate = torch.sigmoid(interference)

# 4. Apply gate to attention key
K = K * gate
```

To encourage specialization, the model uses:

* **L1 sparsity regularization**
* **Variance regularization**

These mechanisms gradually suppress inactive dimensions, allowing different domains (e.g., **code vs classical text**) to activate **distinct neural subspaces**.

透過 **L1 稀疏正則化** 與 **Variance 正則化**，
不必要的特徵維度會逐漸關閉，使不同領域資料（程式碼 / 古文）自然激活不同的神經元子空間。

---

# 🏗 Model Architecture / 模型架構

D2-V11 introduces a **Wave-Interference Concept Gate** before the **O(N) Linear Attention** computation.

```
            ┌─────────────────────┐
            │     Input Token     │
            └──────────┬──────────┘
                       ▼
            ┌─────────────────────┐
            │      Embedding      │
            └──────────┬──────────┘
                       ▼
            ┌──────────┴──────────┐
            ▼                     ▼
    ┌───────────────┐     ┌───────────────┐
    │ Semantic Bank │     │ Context Bank  │
    │ (Amp & Phase) │     │ (Amp & Phase) │
    └───────┬───────┘     └───────┬───────┘
            │                     │
            └─────────┬───────────┘
                      ▼
            ┌─────────────────────┐
            │ Wave Interference   │
            │ A_s * A_c * cos(Δθ) │
            └──────────┬──────────┘
                       ▼
            ┌─────────────────────┐
            │ Concept Gate        │
            │ sigmoid(interf)     │
            │ Sparse Activation   │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │ Linear Attention    │
            │ O(N) Prefix Memory  │
            └──────────┬──────────┘
                       ▼
            ┌─────────────────────┐
            │ Residual + MLP      │
            └──────────┬──────────┘
                       ▼
            ┌─────────────────────┐
            │ Output Layer        │
            └─────────────────────┘
```

---

# ✨ Features / 架構特色

### 🌊 Wave-Interference Implicit Experts

Inspired by qllm2, phase coherence is used for **continuous feature routing**, creating implicit expert behavior.

### ⚡ Causal Linear Attention

Provides **O(N)** memory complexity, enabling theoretically **unbounded context scaling**.

### 🧠 Emergent Domain Specialization

Training on **mixed domains (Wiki / Classical Chinese / Python)** encourages the network to dynamically reorganize neural representations.

### 💾 VRAM-Efficient Training

Optimized to run on **RTX 3060 12GB**, using:

* bfloat16 mixed precision
* gradient checkpointing
* gradient accumulation

---

# 🚀 Getting Started

## 1 Install Dependencies

```bash
pip install torch torchvision torchaudio tqdm
```

---

## 2 Prepare Dataset

Place training text files inside the `data/` directory.

Example datasets:

* Wikipedia
* Classical Chinese texts
* Python source code

The model will automatically construct a vocabulary.

---

## 3 Start Training

```bash
python V11_QLLM2_Genesis.py
```

---

## 4 Inference

```bash
python chat.py
```

---

# 📊 Model Scale

| Parameter      | Value                             |
| -------------- | --------------------------------- |
| Model Size     | ~184M                             |
| Layers         | 24                                |
| Heads          | 12                                |
| Context Length | 768 (O(N) scalable)               |
| Architecture   | Wave-Gated Linear Attention       |
| Training Data  | Wiki + Classical Chinese + Python |
| Hardware       | 1× RTX 3060 12GB                  |

---

# 🔮 Future Work

* Phase dynamics visualization (neuron phase rotation maps)
* Rotary Position Embedding (RoPE)
* Scaling experiments
* Complex-valued neural networks

---

## 📜 Research Question

> Can **phase-coherent interference + sparsity** produce
> **implicit expert specialization** without explicit routing?

D2-V11 is an exploration toward this direction.



這三個加上去，整個專案會 **直接從 hobby project 變 research project 等級**。
