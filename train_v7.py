import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import json

# --- V7 配置區：50M 規模 ---
config = {
    "d_model": 640,         # 增加寬度
    "n_layers": 10,         # 增加深度
    "batch_size": 32,
    "block_size": 256,      # 增加上下文長度，讓邏輯更長遠
    "lr": 4e-4,             # 50M 模型建議用稍微溫和一點的 LR
    "epochs": 20000,        # 規模變大，需要更多步數來磨練
    "data_dir": "data",
    "save_model": "d2_v7_50m.pth",
    "vocab_name": "master_vocab_v7.json",
    "l1_lambda": 0.001      # 延用我們驗證過的稀疏化黃金比例
}
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. 數據加載 (與之前相同，確保 data 資料夾已清洗) ---
all_text = ""
for f in os.listdir(config["data_dir"]):
    if f.endswith(".txt"):
        with open(os.path.join(config["data_dir"], f), 'r', encoding='utf-8') as file:
            all_text += file.read() + "\n"

chars = sorted(list(set(all_text)))
vocab_size = len(chars)
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}
data = torch.tensor([char_to_int[c] for c in all_text], dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data) - config["block_size"], (config["batch_size"],))
    x = torch.stack([data[i:i+config["block_size"]] for i in ix])
    y = torch.stack([data[i+1:i+config["block_size"]+1] for i in ix])
    return x.to(device), y.to(device)

# --- 2. V7 進階版：帶 Projection 的 Causal Gated-D2 ---
class CausalGatedD2Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.concept_gate = nn.Linear(d_model, d_model) 
        self.proj = nn.Linear(d_model, d_model) # 🌟 新增：輸出投影層
        self.ln = nn.LayerNorm(d_model)
        self.current_gate = None 
        
    def forward(self, x):
        B, T, D = x.shape
        x = self.ln(x)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        gate = torch.sigmoid(self.concept_gate(x))
        self.current_gate = gate
        k = k * gate
        
        q, k = F.elu(q) + 1, F.elu(k) + 1
        
        # 矩陣結合律優化
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, T, T)
        attn_weights = attn_weights * mask
        
        out_num = torch.matmul(attn_weights, v)
        k_cumsum = torch.cumsum(k, dim=1)
        out_den = (q * k_cumsum).sum(dim=-1, keepdim=True) + 1e-6
        
        # 🌟 投影後輸出，增加模型特徵重組能力
        return self.proj(out_num / out_den)

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1) # 🌟 50M 模型開始需要一點 Dropout 防止過擬合
        )
        self.ln = nn.LayerNorm(d_model)
    def forward(self, x): return self.net(self.ln(x))

class D2V7Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = CausalGatedD2Attention(d_model)
        self.mlp = MLP(d_model)
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class D2V7Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(*[D2V7Block(d_model) for _ in range(n_layers)])
        self.out_ln = nn.LayerNorm(d_model) # 🌟 輸出前多一層 LN 增加穩定性
        self.head = nn.Linear(d_model, vocab_size)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        return self.head(self.out_ln(x))

# --- 3. 初始化與訓練 ---
model = D2V7Model(vocab_size, config["d_model"], config["n_layers"]).to(device)
optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.05)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"]) # 🌟 改用餘弦退火，訓練更平滑

print(f"🌟 V7 啟動！目標 50M 參數... (當前參數估算: {sum(p.numel() for p in model.parameters())/1e6:.1f}M)")

model.train()
try:
    for epoch in range(config["epochs"]):
        xb, yb = get_batch() 
        logits = model(xb)
        loss_ce = criterion(logits.view(-1, vocab_size), yb.view(-1))
        
        # L1 稀疏化
        l1_loss = sum(block.attn.current_gate.mean() for block in model.blocks)
        total_loss = loss_ce + config["l1_lambda"] * l1_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch % 500 == 0:
            print(f"🚀 V7 | Step {epoch:05d} | Total: {total_loss.item():.4f} (CE: {loss_ce.item():.4f}, L1: {l1_loss.item():.4f}) | LR: {optimizer.param_groups[0]['lr']:.6e}")

except KeyboardInterrupt:
    print("\n⚠️ 手動中斷，保存中...")

torch.save(model.state_dict(), config["save_model"])
with open(config["vocab_name"], "w", encoding="utf-8") as f:
    json.dump({"chars": chars, "char_to_int": char_to_int, "int_to_char": {str(k): v for k, v in int_to_char.items()}}, f, ensure_ascii=False)
