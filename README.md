
# D2-V10: Implicit MoE Linear Attention LLM
**隱式專家線性注意力語言模型 (180M)**

A lightweight LLM experiment exploring **Causal Gated Linear Attention** with sparsity constraints, trained from scratch on a mixed corpus (Wiki, Classical Chinese, Python). 
這是一個輕量級的 LLM 架構實驗，探討**因果門控線性注意力**與稀疏約束，並使用混合語料（維基百科、古文、Python 程式碼）從零訓練。

This project explores whether **feature gating + sparsity regularization** can produce emergent specialization similar to **Mixture-of-Experts (MoE)**, but **without explicit routing** and with **$O(N)$ linear complexity**.
本專案旨在驗證：**特徵門控 + 稀疏正則化** 是否能在**不使用明確路由網路**且具備 **$O(N)$ 線性複雜度** 的前提下，自然湧現出類似 MoE 的專家分工現象。

---

## 🧠 Key Idea / 核心概念

Instead of routing tokens to explicit experts, D2 uses **continuous gating** on both Query and Key:
不同於傳統 MoE 將 Token 強制分配給特定專家，D2 架構對 Query 與 Key 使用**連續門控 (Continuous Gating)**：


Gate = sigmoid(Wx)
Gate_Norm = Gate / mean(Gate)
Q = Q * Gate_Norm
K = K * Gate_Norm


With **L1 regularization**, inactive dimensions are forced to shut down:
透過 **L1 正規化** 懲罰，迫使不必要的特徵維度關閉：


Loss = CrossEntropy + λ * mean(Gate)



This encourages **feature sparsity**, allowing different domains (e.g., Code vs. Classical Chinese) to activate **different neural subspaces**.
這能促進**特徵稀疏性**，讓不同領域的輸入（如：程式碼 vs 古文）激發**不同的神經元子空間**。

---

## 🏗️ Model Architecture / 模型架構

The D2 architecture introduces a **Concept Gate** before the $O(N)$ Linear Attention computation.
D2 架構在 $O(N)$ 線性注意力計算前，引入了**概念門控 (Concept Gate)**。

```text
            ┌─────────────────────┐
            │     Input Token     │
            └──────────┬──────────┘
                       ▼
            ┌─────────────────────┐
            │      Embedding      │
            └──────────┬──────────┘
                       ▼
            ┌─────────────────────┐
            │  Q, K, V Projection │
            └──────────┬──────────┘
                       ▼
            ┌─────────────────────┐
            │ 🌟 Concept Gate 🌟 │
            │    sigmoid(Wx)      │
            │  Sparse Activation  │
            └──────────┬──────────┘
                       │ (Gate applied to Q & K)
                       ▼
            ┌─────────────────────┐
            │ ⚡ Linear Attention │
            │ O(N) Prefix Memory  │
            └──────────┬──────────┘
                       ▼
            ┌─────────────────────┐
            │    Residual + MLP   │
            └──────────┬──────────┘
                       ▼
            ┌─────────────────────┐
            │    Output Layer     │
            └─────────────────────┘

```

---

## ✨ Features / 架構特色

* **Causal Gated-D2 Linear Attention** (因果門控線性注意力)
* **Implicit MoE via Sparsity** (透過稀疏性達成隱式專家)
* **$O(N)$ Context Scaling** ($O(N)$ 長文本擴展能力)
* **VRAM Optimized for RTX 3060** (針對 RTX 3060 12GB 顯存極限優化：混合精度 + 梯度檢查點)
* **Emergent Specialization** (湧現領域分工：古文、百科、程式碼激發不同神經元)

---

## 🚀 Getting Started / 快速開始

### 1. Install Dependencies / 安裝依賴

```bash
pip install torch

```

### 2. Prepare Data / 準備語料

Place your training text files (e.g., Wiki, Classical, Python) inside the `data/` folder.
將你的訓練文本（如：維基百科、古文、Python）放入 `data/` 資料夾中。

### 3. Start Training / 開始訓練

```bash
python d2-v10.py

```

### 4. Inference / 文本生成

Make sure the model and vocab paths are correct, then run:
確認模型權重與字典檔路徑正確後，執行：

```bash
python chat.py

```

---

## 📊 Model Scale / 模型規模

| Parameter / 參數 | Value / 規格 |
| --- | --- |
| **Model Size** (模型大小) | ~180M |
| **Context Length** (上下文長度) | 768 (Scalable / 可擴充) |
| **Architecture** (核心架構) | Gated Linear Attention |
| **Training Data** (訓練語料) | Mixed (Wiki, Classical, Python) |
| **Hardware** (硬體要求) | 1x RTX 3060 12GB |

---

## 🔮 Future Work / 未來展望

* **Gate Activation Analysis** (神經元活化熱力圖分析)
* **Rotary Position Embedding (RoPE)** (旋轉位置編碼)
* **Soft Token Routing** (軟性 Token 路由)

```

你的模型現在應該已經穩穩地在 3060 上跑了好幾個 Step 了吧？**我們下一階段要直接來寫 `analyze_gates.py` 神經元觀測儀器，還是你想先放著讓它煉丹一個晚上看看 Loss 能降到多低？**

```
