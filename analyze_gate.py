import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 載入與訓練時相同的 V10 配置 ---
config = {
    "d_model": 896,
    "n_heads": 14,
    "n_layers": 16,
    "model_path": "d2_v10_genesis.pth",
    "vocab_path": "master_vocab_v10.json",
    "block_size": 256
}

device = "cuda" if torch.cuda.is_available() else "cpu"

class CausalGatedLinearAttentionV10(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.concept_gate = nn.Linear(d_model, d_model) 
        self.proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
    def forward(self, x):
        B, T, D = x.shape
        x_norm = self.ln(x)
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        gate_raw = self.concept_gate(x)
        gate_sig = torch.sigmoid(gate_raw)
        gate_norm = gate_sig / (gate_sig.mean(dim=-1, keepdim=True) + 1e-5)
        gate_norm = gate_norm.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        q = q * gate_norm
        k = k * gate_norm
        
        with torch.amp.autocast('cuda', enabled=False):
            q, k, v = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
            q, k = F.elu(q) + 1.0, F.elu(k) + 1.0
            kv = torch.einsum('b h t d, b h t m -> b h t d m', k, v)
            kv_cumsum = torch.cumsum(kv, dim=2) 
            out_num = torch.einsum('b h t d, b h t d m -> b h t m', q, kv_cumsum)
            k_cumsum = torch.cumsum(k, dim=2)
            out_den = torch.einsum('b h t d, b h t d -> b h t', q, k_cumsum).unsqueeze(-1) + 1e-5
            out = out_num / out_den
            
        out = out.to(x_norm.dtype)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
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

# --- 2. 工具啟動與全面 Hook 註冊 ---
print("🔬 正在初始化神經元觀測儀...")

with open(config["vocab_path"], "r", encoding="utf-8") as f:
    vocab_data = json.load(f)
    char_to_int = vocab_data["char_to_int"]
    unk_id = char_to_int.get("[UNK]", 0)
    vocab_size = len(char_to_int)

model = D2V10Model(vocab_size, config["d_model"], config["n_layers"], config["n_heads"]).to(device)
model.load_state_dict(torch.load(config["model_path"], map_location=device))
model.eval()

# 儲存每一層 Gate 的啟動值
gate_activations = {}
def get_activation_hook(layer_name):
    def hook(module, input, output):
        gate_activations[layer_name] = torch.sigmoid(output).detach().cpu().numpy()
    return hook

# 註冊所有層 (All Layers) 以進行 Layer Specialization 分析
for i in range(config["n_layers"]):
    model.blocks[i].attn.concept_gate.register_forward_hook(get_activation_hook(f"Layer_{i}"))

def text_to_tensor(text):
    return torch.tensor([[char_to_int.get(c, unk_id) for c in text[:config["block_size"]]]], dtype=torch.long).to(device)

# --- 3. 領域測試文本 ---
test_cases = {
    "Wiki (Modern)": "維基百科是一個自由內容的百科全書計畫，其目標是向全人類提供完整的知識。它由來自世界各地的志願者合作編寫。",
    "Classical (Ancient)": "子曰：學而時習之，不亦說乎？有朋自遠方來，不亦樂乎？人不知而不慍，不亦君子乎？",
    "Python (Code)": "def calculate_loss(predictions, targets):\n    loss = F.cross_entropy(predictions, targets)\n    return loss.item()"
}

# 結構: domain_results[domain][layer_name] = head_activations_array
domain_results = {domain: {} for domain in test_cases.keys()}

print("🚀 正在注入測試文本，擷取神經元全層活化訊號...")
with torch.no_grad():
    for domain, text in test_cases.items():
        input_tensor = text_to_tensor(text)
        _ = model(input_tensor)
        
        for i in range(config["n_layers"]):
            activation = gate_activations[f"Layer_{i}"][0] 
            avg_activation = activation.mean(axis=0).reshape(config["n_heads"], config["d_model"] // config["n_heads"])
            head_activation = avg_activation.mean(axis=1)
            domain_results[domain][f"Layer_{i}"] = head_activation

# --- 4. 計算五大指標 ---
print("📊 正在計算分析指標並生成綜合儀表板...")

domains = list(test_cases.keys())
layer_to_observe = 7  # 擷取單層觀察的目標層

# [Metric 1] MoE Heatmap (Layer 7)
moe_heatmap_data = np.array([domain_results[d][f"Layer_{layer_to_observe}"] for d in domains])

# [Metric 2] Head Variance (Layer 7)
# 變異數越高，代表這個 Head 對不同領域切換最敏感
head_variance = np.var(moe_heatmap_data, axis=0)

# [Metric 3] Expert Score (Layer 7)
# 專家分數：(最高活化度) / (平均活化度)，分數越高代表神經元有極端偏好
max_act = np.max(moe_heatmap_data, axis=0)
mean_act = np.mean(moe_heatmap_data, axis=0)
expert_score = max_act / (mean_act + 1e-5)

# [Metric 4] Layer Specialization (All Layers)
# 計算每一層所有 Head 在領域間變異數的均值，觀察特徵在哪一層進行分化
layer_specialization = []
for i in range(config["n_layers"]):
    layer_data = np.array([domain_results[d][f"Layer_{i}"] for d in domains])
    layer_var = np.mean(np.var(layer_data, axis=0))
    layer_specialization.append(layer_var)

# [Metric 5] Domain Clustering
# 將每個領域在所有層的特徵攤平為一維向量，計算餘弦相似度 (Cosine Similarity)
domain_vectors = []
for d in domains:
    vec = np.concatenate([domain_results[d][f"Layer_{i}"] for i in range(config["n_layers"])])
    domain_vectors.append(vec)
domain_vectors = np.array(domain_vectors)
norms = np.linalg.norm(domain_vectors, axis=1, keepdims=True)
normalized_vecs = domain_vectors / (norms + 1e-8)
domain_clustering_sim = np.dot(normalized_vecs, normalized_vecs.T)

# --- 5. 繪製多圖綜合儀表板 ---
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(18, 10))
fig.suptitle("Gated-D2 Concept Gate Specialization Dashboard", fontsize=18, fontweight='bold')

# 1. MoE Heatmap
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(moe_heatmap_data, annot=True, cmap="YlOrRd", 
            xticklabels=[f"H{i+1}" for i in range(config["n_heads"])],
            yticklabels=domains, ax=ax1, cbar_kws={'label': 'Activation'})
ax1.set_title(f"[1] MoE Heatmap (Layer {layer_to_observe})")

# 2. Head Variance
ax2 = plt.subplot(2, 3, 2)
x_heads = np.arange(config["n_heads"])
ax2.bar(x_heads, head_variance, color='skyblue', edgecolor='black')
ax2.set_xticks(x_heads)
ax2.set_xticklabels([f"H{i+1}" for i in range(config["n_heads"])])
ax2.set_title(f"[2] Head Variance (Domain Sensitivity in L{layer_to_observe})")
ax2.set_ylabel("Variance")

# 3. Expert Score
ax3 = plt.subplot(2, 3, 3)
ax3.bar(x_heads, expert_score, color='lightgreen', edgecolor='black')
ax3.set_xticks(x_heads)
ax3.set_xticklabels([f"H{i+1}" for i in range(config["n_heads"])])
ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.5) # 基準線 1.0 = 不偏好
ax3.set_title(f"[3] Expert Score (Max/Mean Ratio in L{layer_to_observe})")
ax3.set_ylabel("Score")

# 4. Layer Specialization
ax4 = plt.subplot(2, 3, 4)
x_layers = np.arange(config["n_layers"])
ax4.plot(x_layers, layer_specialization, marker='o', linewidth=2, color='purple')
ax4.set_xticks(x_layers)
ax4.set_title("[4] Layer Specialization (Domain Divergence by Layer)")
ax4.set_xlabel("Layer Index")
ax4.set_ylabel("Average Domain Variance")

# 5. Domain Clustering (Similarity Matrix)
ax5 = plt.subplot(2, 3, 5)
sns.heatmap(domain_clustering_sim, annot=True, cmap="coolwarm", vmin=0.8, vmax=1.0,
            xticklabels=domains, yticklabels=domains, ax=ax5, cbar_kws={'label': 'Cosine Similarity'})
ax5.set_title("[5] Domain Clustering (Global Similarity)")
ax5.tick_params(axis='y', rotation=0)

# 空白區域可留作文字說明或排版用
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
info_text = (
    "Metrics Explained:\n\n"
    "- MoE Heatmap: Raw gate activation rates across domains.\n"
    "- Head Variance: Identifies which heads shift most between contexts.\n"
    "- Expert Score: Higher score = head is heavily specialized for one domain.\n"
    "- Layer Specialization: Identifies which layer diverges features the most.\n"
    "- Domain Clustering: Overall architectural feature similarity."
)
ax6.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", edgecolor="gray"))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("d2_gate_analysis_dashboard.png", dpi=300, bbox_inches='tight')
print("✅ 分析完成！請查看產生的 'd2_gate_analysis_dashboard.png'。")
