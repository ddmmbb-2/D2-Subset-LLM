import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


class CausalGatedLinearAttentionV10(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.concept_gate = nn.Linear(d_model, d_model) 
        self.proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.last_gate = None # 用於觀測紀錄

    def forward(self, x):
        B, T, D = x.shape
        x_norm = self.ln(x)
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        
        gate_sig = torch.sigmoid(self.concept_gate(x))
        self.last_gate = gate_sig.detach() # 擷取活化訊號
        
        gate_norm = gate_sig / (gate_sig.mean(dim=-1, keepdim=True) + 1e-5)
        q, k = q * gate_norm, k * gate_norm
        
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        with torch.amp.autocast('cuda', enabled=False):
            q, k, v = q.float(), k.float(), v.float()
            q, k = F.elu(q) + 1.0, F.elu(k) + 1.0
            kv = torch.einsum('b h t d, b h t m -> b h t d m', k, v)
            kv_cumsum = torch.cumsum(kv, dim=2) 
            out_num = torch.einsum('b h t d, b h t d m -> b h t m', q, kv_cumsum)
            k_cumsum = torch.cumsum(k, dim=2)
            out_den = torch.einsum('b h t d, b h t d -> b h t', q, k_cumsum).unsqueeze(-1) + 1e-5
            out = out_num / out_den
            
        out = out.to(x_norm.dtype).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)

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

class D2V10Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = CausalGatedLinearAttentionV10(d_model, n_heads)
        self.mlp = MLP(d_model)
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class D2V10Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(*[D2V10Block(d_model, n_heads) for _ in range(n_layers)])
        self.out_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks: x = block(x)
        return self.head(self.out_ln(x))

# --- 2. 分析儀配置 ---
CONFIG = {
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 24,
    "model_path": "d2_v11_genesis_180m.pth",
    "vocab_path": "master_vocab_v11.json",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# 測試案例
# 測試案例 (已加入邏輯推導)
test_cases = {
    "Wiki (Modern)": "維基百科是一個自由內容的百科全書計畫，其目標是向全人類提供完整的知識。",
    "Classical (Ancient)": "子曰：學而時習之，不亦說乎？有朋自遠方來，不亦樂乎？",
    "Python (Code)": "def train_model(data):\n    optimizer.zero_grad()\n    loss = criterion(model(data))\n    return loss",
    "Logic (Math)": "問題：阿芳有7個香蕉，給了小明2個。請問阿芳現在有幾個香蕉？\n【思考過程】\n1. 初始狀態：阿芳有 7 個。\n2. 動作：給出 2 個，所以要用減法。\n3. 計算：7 - 2 = 5。\n【答案】"
}

def run_analysis():
    print("🔬 啟動 V10 隱式專家深度分析儀...")
    
    # 載入詞表與模型
    with open(CONFIG["vocab_path"], "r", encoding="utf-8") as f:
        vocab = json.load(f)
        char_to_int = vocab["char_to_int"]
        unk_id = char_to_int.get("[UNK]", 0)
    
    model = D2V10Model(len(char_to_int), CONFIG["d_model"], CONFIG["n_layers"], CONFIG["n_heads"]).to(CONFIG["device"])
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=CONFIG["device"]))
    model.eval()

    all_layer_data = [] # 儲存每層的平均活化
    domain_results = {} # 紀錄特定層(Layer 7)的詳細活化

    with torch.no_grad():
        for name, text in test_cases.items():
            tokens = torch.tensor([[char_to_int.get(c, unk_id) for c in text]], dtype=torch.long).to(CONFIG["device"])
            _ = model(tokens)
            
            layer_activations = []
            for i, block in enumerate(model.blocks):
                gate_val = block.attn.last_gate[0].cpu().numpy() # (T, D)
                layer_activations.append(gate_val.mean())
                
                if i == 7: # 選取中間層觀測專家分工
                    # 計算每個 Head 的平均活化度 (T, D) -> (Heads,)
                    head_act = gate_val.reshape(-1, CONFIG["n_heads"], CONFIG["d_model"] // CONFIG["n_heads"]).mean(axis=(0, 2))
                    domain_results[name] = {"gate": gate_val, "head_act": head_act}
            
            all_layer_data.append(layer_activations)

    # --- 3. 生成圖表 ---
    
    # 圖 1: MoE Heatmap
    plt.figure(figsize=(10, 6))
    heat_data = np.array([domain_results[n]["head_act"] for n in test_cases.keys()])
    sns.heatmap(heat_data, annot=True, cmap="YlOrRd", yticklabels=test_cases.keys(), 
                xticklabels=[f"H{i+1}" for i in range(CONFIG["n_heads"])])
    plt.title("1. MoE Heatmap - Head Activation by Domain (Layer 7)")
    plt.savefig("moe_heatmap.png")

    # 圖 2: Head Variance
    plt.figure(figsize=(8, 5))
    variances = [np.var(d["head_act"]) for d in domain_results.values()]
    plt.bar(test_cases.keys(), variances, color='skyblue')
    plt.title("2. Head Activation Variance (Specialization Score)")
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
    sns.scatterplot(x=pca_res[:, 0], y=pca_res[:, 1], hue=labels)
    plt.title("3. Domain Clustering (Gate Semantic Space)")
    plt.savefig("domain_clustering.png")

    # 圖 4: Layer Specialization
    plt.figure(figsize=(10, 5))
    avg_layers = np.mean(all_layer_data, axis=0)
    plt.plot(range(CONFIG["n_layers"]), avg_layers, marker='o', linewidth=2)
    plt.title("4. Layer Specialization Trend (Mean Gate Activation)")
    plt.xlabel("Layer Index")
    plt.ylabel("Activation Rate")
    plt.grid(True)
    plt.savefig("layer_specialization.png")

    # 圖 5: Expert Score
    plt.figure(figsize=(12, 4))
    # 專家分：跨領域活化的標準差（越高代表越是特定領域專家）
    expert_scores = np.std(heat_data, axis=0)
    plt.bar([f"H{i+1}" for i in range(CONFIG["n_heads"])], expert_scores, color='salmon')
    plt.title("5. Expert Score (Specificity per Head)")
    plt.savefig("expert_score.png")

    print("✅ 分析完成，已生成 5 張觀測圖表。")

if __name__ == "__main__":
    if os.path.exists(CONFIG["model_path"]):
        run_analysis()
    else:
        print(f"❌ 找不到模型檔案: {CONFIG['model_path']}")
