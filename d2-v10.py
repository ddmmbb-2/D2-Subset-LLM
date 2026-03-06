import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from collections import Counter

# --- V10.1 強化配置區 ---
config = {
    "d_model": 896,          
    "n_heads": 14,           
    "n_layers": 16,
    "batch_size": 2,         
    "accum_steps": 6,        
    "block_size": 768,       
    "lr": 1e-4,              # 續訓建議調低 LR 進行微雕
    "epochs": 40000,         # 目標步數
    "data_dir": "data",      
    "save_model": "d2_v10_genesis.pth",
    "vocab_name": "master_vocab_v10.json",
    "l1_lambda": 0.0001,     
    "balance_lambda": 0.01,  # 防止專家崩潰的力道
    "min_freq": 5            
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. 字典建立與資料讀取 ---
print(f"🔍 正在掃描 {config['data_dir']} 資料夾...")
all_text = ""
txt_files = glob.glob(os.path.join(config["data_dir"], "*.txt"))
if not txt_files:
    raise FileNotFoundError(f"❌ 在 {config['data_dir']} 資料夾中找不到任何 .txt 檔案！")

for file_path in txt_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        all_text += f.read() + "\n\n" 

counter = Counter(all_text)
valid_chars = sorted([ch for ch, count in counter.items() if count >= config["min_freq"]])
vocab = ["[UNK]"] + valid_chars
vocab_size = len(vocab)
char_to_int = {ch: i for i, ch in enumerate(vocab)}
int_to_char = {i: ch for i, ch in enumerate(vocab)}
unk_id = char_to_int["[UNK]"]
data = torch.tensor([char_to_int.get(c, unk_id) for c in all_text], dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data) - config["block_size"], (config["batch_size"],))
    x = torch.stack([data[i:i+config["block_size"]] for i in ix])
    y = torch.stack([data[i+1:i+config["block_size"]+1] for i in ix])
    return x.to(device), y.to(device)

# --- 2. Gated-D2 架構 (V10 線性注意力 + Load Balance) ---
class CausalGatedLinearAttentionV10(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.concept_gate = nn.Linear(d_model, d_model) 
        self.proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.gate_sparsity = 0.0 
        self.gate_balance = 0.0 

    def forward(self, x):
        B, T, D = x.shape
        x_norm = self.ln(x)
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        
        # Concept Gate
        gate_sig = torch.sigmoid(self.concept_gate(x))
        self.gate_sparsity = gate_sig.mean()
        
        # Load Balance: 獲取每個維度的平均啟動率
        gate_usage = gate_sig.mean(dim=(0, 1)) 
        self.gate_balance = torch.var(gate_usage) # 懲罰分配不均
        
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
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
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
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        return self.head(self.out_ln(x))

# --- 3. 點火訓練 ---
model = D2V10Model(vocab_size, config["d_model"], config["n_layers"], config["n_heads"]).to(device)

if os.path.exists(config["save_model"]):
    print(f"♻️ 正在喚醒 V10 大腦進行續訓...")
    model.load_state_dict(torch.load(config["save_model"], map_location=device, weights_only=True))

optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.05)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
scaler = torch.amp.GradScaler('cuda') 

model.train()
optimizer.zero_grad()

try:
    for epoch in range(config["epochs"]):
        xb, yb = get_batch() 
        with torch.amp.autocast('cuda'):
            logits = model(xb)
            loss_ce = criterion(logits.view(-1, vocab_size), yb.view(-1))
            
            l1_val = sum(b.attn.gate_sparsity for b in model.blocks) / config["n_layers"]
            bal_val = sum(b.attn.gate_balance for b in model.blocks) / config["n_layers"]
            
            # 總損失：包含交叉熵、L1 稀疏與負載均衡
            total_loss = (loss_ce + config["l1_lambda"] * l1_val + config["balance_lambda"] * bal_val) / config["accum_steps"]
        
        scaler.scale(total_loss).backward()
        
        if (epoch + 1) % config["accum_steps"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        if epoch % 100 == 0:
            print(f"🚀 V10+ | Step {epoch:05d} | Loss: {loss_ce.item():.4f} | Sparse: {l1_val.item():.4f} | Bal: {bal_val.item():.5f}")

except KeyboardInterrupt:
    print("\n⚠️ 訓練中斷，正在保存結晶...")

torch.save(model.state_dict(), config["save_model"])
with open(config["vocab_name"], "w", encoding="utf-8") as f:
    json.dump({"chars": vocab, "char_to_int": char_to_int, "int_to_char": {str(k): v for k, v in int_to_char.items()}}, f, ensure_ascii=False)
