D2-V11: Wave-Interference Implicit MoE Linear Attention LLM
波干涉隱式專家線性注意力語言模型 (180M)
A lightweight LLM experiment exploring Causal Linear Attention powered by Wave Interference Gating (inspired by QLLM2), trained from scratch on a mixed corpus (Wiki, Classical Chinese, Python).
這是一個輕量級的 LLM 架構實驗，探討由 波干涉門控 (受 QLLM2 啟發 https://github.com/gowrav-vishwakarma/qllm2 ) 驅動的因果線性注意力，並使用混合語料（維基百科、古文、Python 程式碼）從零訓練。
This project explores whether phase coherence + sparsity regularization can produce emergent specialization similar to Mixture-of-Experts (MoE), but without explicit routing and with $O(N)$ linear complexity.

本專案旨在驗證：相位干涉 + 稀疏正則化 是否能在不使用明確路由網路且具備 $O(N)$ 線性複雜度 的前提下，自然湧現出類似 MoE 的專家分工現象。
🧠 Key Idea / 核心概念
Instead of routing tokens to explicit experts or using a simple linear sigmoid, D2-V11 treats tokens as waves. It projects features into a Semantic Bank and a Context Bank, using their phase coherence to naturally amplify or cancel out features:
不同於傳統 MoE 的強制分配或單純的線性映射，D2-V11 將 Token 視為波。它將特徵投射為語意庫 (Semantic Bank) 與 上下文庫 (Context Bank)，並利用兩者的相位共相度，自然地放大或抵銷特徵：
Mathematical Core (數學核心):

$$\text{Interference} = A_{\text{sem}} \cdot A_{\text{ctx}} \cdot \cos(\theta_{\text{sem}} - \theta_{\text{ctx}})$$
Implementation (實作邏輯):

Python


# 1. Extract Amplitude and Phase / 提取振幅與相位
sem_amp, sem_phase = Semantic_Bank(x)
ctx_amp, ctx_phase = Context_Bank(x)

# 2. Phase Coherence (Wave Interference) / 波干涉
interference = sem_amp * ctx_amp * torch.cos(sem_phase - ctx_phase)

# 3. Apply to Key / 產生軟性門控並套用於 Key
Gate = torch.sigmoid(interference)
K = K * Gate


With L1 and Variance regularization, inactive dimensions are smoothly forced to shut down:
透過 L1 與變異數 (Variance) 正規化，迫使不必要的特徵維度平滑關閉，讓不同領域的輸入（程式碼 vs 古文）自然激發不同的神經元子空間。
🏗️ Model Architecture / 模型架構
The D2-V11 architecture introduces a Wave Interference Concept Gate before the $O(N)$ Linear Attention computation.
D2-V11 架構在 $O(N)$ 線性注意力計算前，引入了波干涉概念門控。

Plaintext


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
            │ 🌊 Wave Interference│
            │ A_s * A_c * cos(Δθ) │
            └──────────┬──────────┘
                       ▼
            ┌─────────────────────┐
            │ 🌟 Concept Gate 🌟 │
            │sigmoid(interference)│
            │  Sparse Activation  │
            └──────────┬──────────┘
                       │ (Gate applied to K)
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


✨ Features / 架構特色
Wave Interference Implicit MoE (波干涉隱式專家): QLLM2-inspired phase coherence for smooth, continuous feature routing.
Causal Linear Attention (因果線性注意力): $O(N)$ memory footprint, enabling infinite context scaling in theory.
VRAM Optimized for RTX 3060 (極限顯存優化): Engineered for 12GB VRAM using bfloat16 mixed precision, Gradient Checkpointing, and tailored accumulation steps.
Emergent Specialization (湧現領域分工): Mixed dataset (Wiki, Classical, Python) forces the model to dynamically shift neural states based on context.
🚀 Getting Started / 快速開始
1. Install Dependencies / 安裝依賴

Bash


pip install torch torchvision torchaudio tqdm


2. Prepare Data / 準備語料
Place your training text files (e.g., Wiki, Classical, Python) inside the data/ folder. The model will automatically build a dynamic vocabulary.
將你的訓練文本（如：維基百科、古文、Python）放入 data/ 資料夾中。模型會自動建立動態詞表。
3. Start Training / 開始訓練

Bash


python train.py


4. Inference / 文本生成
Make sure the model and vocab paths are correct, then run:
確認模型權重與字典檔路徑正確後，執行：

Bash


python chat.py


📊 Model Scale / 模型規模
Parameter / 參數
Value / 規格
Model Size (模型大小)
~184M
Layers / Heads (層數 / 頭數)
24 Layers / 12 Heads
Context Length (上下文長度)
768 (O(N) Scalable / 可無損擴充)
Architecture (核心架構)
Wave-Gated Linear Attention
Training Data (訓練語料)
Mixed (Wiki, Classical, Python)
Hardware (硬體要求)
1x RTX 3060 12GB

🔮 Future Work / 未來展望
Phase Tracking Analysis (神經元相位旋轉與活化熱力圖分析)
Rotary Position Embedding (RoPE) (旋轉位置編碼)
Scaling to Complex-Valued Networks (擴展至純複數神經網路)
