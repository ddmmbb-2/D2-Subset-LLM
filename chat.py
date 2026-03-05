import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# --- 1. 必須與訓練完全一致的模型定義 ---
class CausalGatedD2Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.concept_gate = nn.Linear(d_model, d_model) 
        self.ln = nn.LayerNorm(d_model)
        
    def forward(self, x):
        B, T, D = x.shape
        x = self.ln(x)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        gate = torch.sigmoid(self.concept_gate(x))
        k = k * gate
        
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # 矩陣結合律推理
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, T, T)
        attn_weights = attn_weights * mask
        
        out_num = torch.matmul(attn_weights, v)
        k_cumsum = torch.cumsum(k, dim=1)
        out_den = (q * k_cumsum).sum(dim=-1, keepdim=True) + 1e-6
        
        return out_num / out_den

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln = nn.LayerNorm(d_model)
    def forward(self, x): return self.net(self.ln(x))

class D2V3Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = CausalGatedD2Attention(d_model)
        self.mlp = MLP(d_model)
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class D2V3Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(*[D2V3Block(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        return self.head(x)

# --- 2. 載入模型與字典 ---
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("master_vocab_v6.json", "r", encoding="utf-8") as f:
    v = json.load(f)
chars, char_to_int = v["chars"], v["char_to_int"]
int_to_char = {int(k): val for k, val in v["int_to_char"].items()}
vocab_size = len(chars)

model = D2V3Model(vocab_size, 512, 6).to(device) # 確認 n_layers = 6
model.load_state_dict(torch.load("d2_v6_scratch.pth", weights_only=True))
model.eval()

# --- 3. 對話生成 ---
def chat(prompt, length=150, temp=0.7, top_k=5):
    input_ids = torch.tensor([[char_to_int.get(c, 0) for c in prompt]], device=device)
    res = prompt
    
    for _ in range(length):
        with torch.no_grad():
            logits = model(input_ids)[:, -1, :] / temp
            
            # Top-K 採樣
            v_val, _ = torch.topk(logits, top_k)
            logits[logits < v_val[:, [-1]]] = -float('Inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            
            char = int_to_char.get(next_id.item(), "？")
            res += char
            input_ids = torch.cat([input_ids, next_id], dim=1)
            
            # 保持窗口長度在 128 以內
            if input_ids.shape[1] > 128:
                input_ids = input_ids[:, 1:]
                
    return res

print("\n--- 🧠 融合了 Gated-D2 子集空間的 AI 回應 ---")
print(chat("雖然寫程式很累，但是", length=150, temp=0.8, top_k=5))
