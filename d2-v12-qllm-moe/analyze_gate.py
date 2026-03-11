import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from tokenizers import Tokenizer # 🌟 新增 BPE 支援

# ==========================================
# 1. 分析儀配置 (對齊 V12)
# ==========================================
CONFIG = {
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 24,
    "model_path": "d2_v12_qllm2_moe.pth",     # 🆕 V12 權重檔名
    "vocab_path": "bpe_tokenizer_v12.json",   # 🆕 V12 BPE 字典檔名
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ==========================================
# 2. V12 核心模型架構 (含觀測探針)
# ==========================================
class CausalGatedD2Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.n_heads = CONFIG["n_heads"]
        self.d_head = d_model // self.n_heads
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.semantic_bank = nn.Linear(d_model, d_model * 2) 
        self.context_bank = nn.Linear(d_model, d_model * 2)  
        self.proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.last_gate = None # 🔬 觀測探針：用來攔截波干涉 Gate 值

    def forward(self, x):
        B, L, D = x.shape
        x_norm = self.ln(x)
        
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        
        # 🌊 波干涉隱式 MoE
        sem_amp, sem_phase = self.semantic_bank(x).chunk(2, dim=-1)
        sem_amp = F.softplus(sem_amp) 
        ctx_amp, ctx_phase = self.context_bank(x).chunk(2, dim=-1)
        ctx_amp = F.softplus(ctx_amp)
        
        interference = sem_amp * ctx_amp * torch.cos(sem_phase - ctx_phase)
        gate = torch.sigmoid(interference)
        
        self.last_gate = gate.detach() # 🔬 擷取並儲存目前的活化訊號
        k = k * gate 
        
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        q_f, k_f, v_f = q.float(), k.float(), v.float()
        q_f = F.elu(q_f) + 1.0
        k_f = F.elu(k_f) + 1.0
        
        kv_state = k_f.unsqueeze(-1) * v_f.unsqueeze(-2)  
        kv_cumsum = torch.cumsum(kv_state, dim=2) 
        out_num = torch.matmul(q_f.unsqueeze(-2), kv_cumsum).squeeze(-2) 
        k_cumsum = torch.cumsum(k_f, dim=2)
        out_den = (q_f * k_cumsum).sum(dim=-1, keepdim=True) + 1e-6
        
        attn_out = out_num / out_den
        attn_out = attn_out.to(x.dtype)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        
        return self.proj(attn_out)

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln = nn.LayerNorm(d_model)
    def forward(self, x): return self.net(self.ln(x))

class D2V12Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = CausalGatedD2Attention(d_model)
        self.mlp = MLP(d_model)
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class D2V12Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([D2V12Block(d_model) for _ in range(n_layers)])
        self.out_ln = nn.LayerNorm(d_model)
        
        # 🔗 權重綁定 (Weight Tying)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight
        
    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks: x = block(x)
        return self.head(self.out_ln(x))

# ==========================================
# 3. 測試語料與分析邏輯
# ==========================================
test_cases = {
    "Wiki (Modern)": "維基百科是一個自由內容的百科全書計畫，其目標是向全人類提供完整的知識。",
    "Classical (Ancient)": "子曰：學而時習之，不亦說乎？有朋自遠方來，不亦樂乎？",
    "Python (Code)": "def train_model(data):\n    optimizer.zero_grad()\n    loss = criterion(model(data))\n    return loss",
    "Logic (Math)": "問題：阿芳有7個香蕉，給了小明2個。請問阿芳現在有幾個香蕉？\n【思考過程】\n1. 初始狀態：阿芳有 7 個。\n2. 動作：給出 2 個，所以要用減法。\n3. 計算：7 - 2 = 5。\n【答案】"
}

def run_analysis():
    print("🔬 啟動 V12 隱式專家深度分析儀...")
    
    # 載入 BPE 分詞器
    if not os.path.exists(CONFIG["vocab_path"]):
        raise FileNotFoundError(f"❌ 找不到 BPE 分詞器：{CONFIG['vocab_path']}")
    tokenizer = Tokenizer.from_file(CONFIG["vocab_path"])
    vocab_size = tokenizer.get_vocab_size()
    
    # 載入模型與權重 (支援 V12 斷點續傳格式)
    model = D2V12Model(vocab_size, CONFIG["d_model"], CONFIG["n_layers"]).to(CONFIG["device"])
    
    if os.path.exists(CONFIG["model_path"]):
        checkpoint = torch.load(CONFIG["model_path"], map_location=CONFIG["device"], weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            step = checkpoint.get('step', '未知')
            print(f"✅ 成功載入 V12 權重！(來自 Step {step})")
        else:
            model.load_state_dict(checkpoint)
            print("✅ 成功載入 V12 權重！")
    else:
        print(f"❌ 找不到模型檔案: {CONFIG['model_path']}")
        return

    model.eval()

    all_layer_data = [] # 儲存每層的平均活化
    domain_results = {} # 紀錄特定層(Layer 7)的詳細活化

    with torch.no_grad():
        for name, text in test_cases.items():
            # 🆕 使用 BPE 編碼
            encoded = tokenizer.encode(text)
            tokens = torch.tensor([encoded.ids], dtype=torch.long).to(CONFIG["device"])
            _ = model(tokens)
            
            layer_activations = []
            for i, block in enumerate(model.blocks):
                gate_val = block.attn.last_gate[0].cpu().numpy() # (T, D)
                layer_activations.append(gate_val.mean())
                
                # 選取中間層觀測專家分工 (可自由調整，例如改看 Layer 12)
                if i == 7: 
                    # 計算每個 Head 的平均活化度 (T, D) -> (Heads,)
                    head_act = gate_val.reshape(-1, CONFIG["n_heads"], CONFIG["d_model"] // CONFIG["n_heads"]).mean(axis=(0, 2))
                    domain_results[name] = {"gate": gate_val, "head_act": head_act}
            
            all_layer_data.append(layer_activations)

    # ==========================================
    # 4. 生成圖表
    # ==========================================
    print("📊 正在繪製並儲存分析圖表...")
    
    # 圖 1: MoE Heatmap
    plt.figure(figsize=(10, 6))
    heat_data = np.array([domain_results[n]["head_act"] for n in test_cases.keys()])
    sns.heatmap(heat_data, annot=True, cmap="YlOrRd", yticklabels=test_cases.keys(), 
                xticklabels=[f"H{i+1}" for i in range(CONFIG["n_heads"])])
    plt.title("1. MoE Heatmap - Head Activation by Domain (Layer 7)")
    plt.tight_layout()
    plt.savefig("moe_heatmap.png")

    # 圖 2: Head Variance
    plt.figure(figsize=(8, 5))
    variances = [np.var(d["head_act"]) for d in domain_results.values()]
    plt.bar(test_cases.keys(), variances, color='skyblue')
    plt.title("2. Head Activation Variance (Specialization Score)")
    plt.tight_layout()
    plt.savefig("head_variance.png")

    # 圖 3: Domain Clustering (PCA)
    plt.figure(figsize=(8, 6))
    pca = PCA(n_components=2)
    all_points = []
    labels = []
    for name in test_cases.keys():
        points = domain_results[name]["gate"] # (T, D)
        all_points.append(points)
        labels.extend([name] * points.shape[0])
    pca_res = pca.fit_transform(np.concatenate(all_points))
    sns.scatterplot(x=pca_res[:, 0], y=pca_res[:, 1], hue=labels, palette="deep")
    plt.title("3. Domain Clustering (Gate Semantic Space)")
    plt.tight_layout()
    plt.savefig("domain_clustering.png")

    # 圖 4: Layer Specialization
    plt.figure(figsize=(10, 5))
    avg_layers = np.mean(all_layer_data, axis=0)
    plt.plot(range(CONFIG["n_layers"]), avg_layers, marker='o', linewidth=2, color='purple')
    plt.title("4. Layer Specialization Trend (Mean Gate Activation)")
    plt.xlabel("Layer Index")
    plt.ylabel("Activation Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("layer_specialization.png")

    # 圖 5: Expert Score
    plt.figure(figsize=(12, 4))
    expert_scores = np.std(heat_data, axis=0)
    plt.bar([f"H{i+1}" for i in range(CONFIG["n_heads"])], expert_scores, color='salmon')
    plt.title("5. Expert Score (Specificity per Head)")
    plt.tight_layout()
    plt.savefig("expert_score.png")

    print("✅ 分析完成，已在目前資料夾生成 5 張觀測圖表！")

if __name__ == "__main__":
    run_analysis()