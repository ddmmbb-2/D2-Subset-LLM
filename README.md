
# D2-V12: Wave-Interference Implicit MoE Linear Attention LLM

**波干涉隱式專家線性注意力語言模型 (V12 進化版)**

A lightweight experimental Large Language Model exploring **Causal Linear Attention** combined with **Wave-Interference Gating**. The core wave mechanics and phase interference formulations in this project are deeply inspired by and adapted from [**qllm2** by gowrav-vishwakarma](https://github.com/gowrav-vishwakarma/qllm2).

This V12 model is trained **from scratch** on a mixed corpus including **Wikipedia**, **Classical Chinese**, and **Python code**. It aims to study whether **phase coherence and sparsity regularization** can produce **emergent specialization** similar to Mixture-of-Experts (MoE) — but **without explicit routing** and while maintaining **O(N) linear complexity**. 

這是一個輕量級 LLM 架構實驗，探索由 **波干涉門控 (Wave-Interference Gating)** 驅動的 **因果線性注意力 (Causal Linear Attention)**。本專案的核心波函數機制與相位干涉概念，主要借鑒並改編自 [**qllm2** (gowrav-vishwakarma)](https://github.com/gowrav-vishwakarma/qllm2)。

本專案的核心問題是：
> 在 **不使用明確 Expert Router** 的情況下，
> **相位干涉 + 稀疏正則化** 是否能自然湧現出類似 **Mixture-of-Experts** 的專家分工？

同時保持 **O(N)** 的線性計算複雜度，並在 V12 導入了 **BPE Tokenizer** 與 **Weight Tying** 完美適配單張 RTX 3060 12GB。

---

## 🧠 Key Idea / 核心概念

Traditional MoE architectures **explicitly route tokens** to different experts. Building upon the quantum-inspired token mechanics from **qllm2**, D2-V12 instead treats **tokens as waves**.

Features are projected into two latent spaces:
* **Semantic Bank** – captures semantic amplitude and phase
* **Context Bank** – captures contextual amplitude and phase

Their **phase coherence** determines whether features are **amplified or cancelled** through wave interference. 透過兩者的 **相位差 (phase difference)** 形成 **波干涉效應**，從而自然地 **放大或抑制特徵維度**。

---

## 📐 Mathematical Core / 數學核心

$$\text{Interference} = A_{sem} \cdot A_{ctx} \cdot \cos(\theta_{sem} - \theta_{ctx})$$

Where:
* $A$ represents feature amplitude (forced to be positive via softplus).
* $\theta$ represents feature phase.

---

## ✨ V12 New Features / 架構特色

1. **BPE Tokenizer (Byte-Pair Encoding):** Replaced character-level tokenization with an efficient 16K BPE dictionary, drastically reducing VRAM usage and improving semantic context window.
2. **Weight Tying:** Tied embedding and output linear layers to save ~50MB of VRAM, stabilizing early training loss.
3. **Wave-Interference Implicit Experts:** Phase coherence is used for continuous feature routing.
4. **Causal Linear Attention:** Provides $O(N)$ memory complexity.
5. **VRAM-Efficient Training:** Fully optimized to run on **RTX 3060 12GB** using `bfloat16` mixed precision and gradient checkpointing.

---

## 🔬 Brain Scan Analysis / 大腦斷層掃描

By analyzing the internal activation rates (Gate Values) of the trained model, we observe clear emergent MoE behaviors. *(Run `analyze_gate.py` to generate these charts)*

### 1. Emergent Domain Specialization (專家分工湧現)
Different attention heads automatically specialize in different domains. For example, Head 3 (H3) shows significantly higher activation when processing Python code and Math logic compared to Classical Chinese.
![MoE Heatmap](moe_heatmap.png)

### 2. Layer Specialization Trend (神經層級分工)
The model exhibits a perfect "U-shaped" activation curve. Early layers process basic token recognition, middle layers (the routing center) sharply decrease activation to perform sparse expert routing, and deep layers re-activate to integrate outputs.
![Layer Specialization](layer_specialization.png)

### 3. Domain Clustering in Semantic Space (波干涉語意空間聚類)
Through PCA dimensionality reduction on the Wave Interference Gates, we can see that natural languages (Wiki, Classical) and logical languages (Python, Math) are projected into entirely distinct neural subspaces.
![Domain Clustering](domain_clustering.png)

### 4. Head Variance & Expert Specificity (頭部活化變異與專家評分)
![Expert Score](expert_score.png)
![Head Variance](head_variance.png)

---

## 🏗 Model Architecture / 模型架構

```text
            ┌─────────────────────┐
            │ BPE Encoded Token   │
            └──────────┬──────────┘
                       ▼
            ┌─────────────────────┐
            │ Tied Embedding      │
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
            │ Tied Output Layer   │
            └─────────────────────┘

```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install torch torchvision torchaudio tqdm tokenizers matplotlib seaborn scikit-learn

```

### 2. Prepare Dataset

Place training `.txt` files inside the `data/` directory. The model will automatically build a 16K BPE vocabulary on the first run.

### 3. Start Training

```bash
python d2-v12-qllm2-moe.py

```

### 4. Inference & Chat

```bash
python chat.py

```

### 5. Generate Brain Scan Analysis (Plots)

```bash
python analyze_gate.py

```

---

## 📊 Model Scale

| Parameter | Value |
| --- | --- |
| Model Size | ~239.4M |
| Vocab Size | 16,384 (BPE) |
| Layers | 24 |
| Heads | 12 |
| Context Length | 768 (O(N) scalable) |
| Architecture | Wave-Gated Linear Attention |
| Hardware | 1× NVIDIA RTX 3060 12GB |

---

## 🙏 Acknowledgments / 鳴謝

Special thanks to the **[qllm2](https://github.com/gowrav-vishwakarma/qllm2)** project by **gowrav-vishwakarma**.
The core concepts of treating tokens as waves, utilizing phase dynamics, and implementing wave-interference gating in D2-V12 were deeply inspired by their pioneering work on quantum-inspired LLM architectures.

特別感謝 **gowrav-vishwakarma** 開源的 **[qllm2](https://github.com/gowrav-vishwakarma/qllm2)** 專案。D2-V12 中將 Token 視為波的核心概念、相位動力學的運用，以及波干涉門控的機制，均深受其在量子啟發 (Quantum-inspired) 語言模型架構上先驅性工作的啟發。

