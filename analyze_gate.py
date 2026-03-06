import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# --- 1. 載入與訓練時相同的 V10 配置 ---
config = {
    "d_model": 896,
    "n_heads": 14,
    "n_layers": 16,
    "model_path": "d2_v10_genesis.pth",
    "vocab_path": "master_vocab_v10.json",
    "block_size": 256 # 測試時不需要太長
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# (為了獨立執行，這裡簡化貼上 V10 的模型結構，但移除了訓練用代碼)
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
        for block in self.blocks: x = block(x) # 觀測時不需 checkpoint
        return self.head(self.out_ln(x))

# --- 2. 工具啟動與 Hook 註冊 ---
print("🔬 正在初始化神經元觀測儀...")

with open(config["vocab_path"], "r", encoding="utf-8") as f:
    vocab_data = json.load(f)
    char_to_int = vocab_data["char_to_int"]
    unk_id = char_to_int.get("[UNK]", 0)
    vocab_size = len(char_to_int)

model = D2V10Model(vocab_size, config["d_model"], config["n_layers"], config["n_heads"]).to(device)
model.load_state_dict(torch.load(config["model_path"], map_location=device))
model.eval()

# 儲存每層 Gate 的啟動值
gate_activations = {}
def get_activation_hook(layer_name):
    def hook(module, input, output):
        # output 是 gate_raw，我們套用 sigmoid 取得真實啟動率
        gate_sig = torch.sigmoid(output).detach().cpu().numpy()
        gate_activations[layer_name] = gate_sig
    return hook

# 這裡我們觀察網路「中間層」(Layer 7) 的特徵分工最明顯
layer_to_observe = 7
model.blocks[layer_to_observe].attn.concept_gate.register_forward_hook(get_activation_hook("Layer_7"))

def text_to_tensor(text):
    return torch.tensor([[char_to_int.get(c, unk_id) for c in text[:config["block_size"]]]], dtype=torch.long).to(device)

# --- 3. 領域測試文本 ---
test_cases = {
    "Wiki (Modern)": "維基百科是一個自由內容的百科全書計畫，其目標是向全人類提供完整的知識。它由來自世界各地的志願者合作編寫。",
    "Classical (Ancient)": "子曰：學而時習之，不亦說乎？有朋自遠方來，不亦樂乎？人不知而不慍，不亦君子乎？",
    "Python (Code)": "def calculate_loss(predictions, targets):\n    loss = F.cross_entropy(predictions, targets)\n    return loss.item()"
}

results = []

print("🚀 正在注入測試文本，擷取神經元活化訊號...")
with torch.no_grad():
    for name, text in test_cases.items():
        input_tensor = text_to_tensor(text)
        _ = model(input_tensor)
        
        # 取出該文本在 Layer 7 的 Gate 啟動矩陣: shape (1, Seq, d_model)
        activation = gate_activations["Layer_7"][0] 
        
        # 1. 沿著時間軸 (Seq) 平均，得到每個維度對這段文本的平均關注度
        # 2. Reshape 成 (n_heads, d_head) 來觀察各個 Head 的活化狀況
        avg_activation = activation.mean(axis=0).reshape(config["n_heads"], config["d_model"] // config["n_heads"])
        
        # 再次平均，取得每個 Head 整體的活化程度
        head_activation = avg_activation.mean(axis=1)
        results.append(head_activation)

# --- 4. 繪製熱力圖 ---
print("📊 正在生成隱式專家分工熱力圖...")
plt.figure(figsize=(10, 6))
sns.heatmap(np.array(results), annot=True, cmap="YlOrRd", 
            xticklabels=[f"Head {i+1}" for i in range(config["n_heads"])],
            yticklabels=list(test_cases.keys()),
            cbar_kws={'label': 'Gate Activation Rate'})

plt.title(f"Concept Gate Activation by Domain (Layer {layer_to_observe})")
plt.xlabel("Attention Heads (Implicit Experts)")
plt.ylabel("Input Domain")
plt.tight_layout()
plt.savefig("moe_heatmap.png", dpi=300)
print("✅ 觀測完成！請查看產生的 'moe_heatmap.png'。")
