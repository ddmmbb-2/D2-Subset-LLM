import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from collections import Counter

# --- V10 創世紀配置區 (專為 RTX 3060 12GB 設計) ---
config = {
    "d_model": 896,          
    "n_heads": 14,           # 🌟 新增：多頭機制，d_head = 64，這對 O(N) 狀態計算至關重要
    "n_layers": 16,
    "batch_size": 4,         # 降低 Batch Size，改用梯度累積 (Gradient Accumulation) 模擬 12
    "accum_steps": 3,        # 4 * 3 = 12 (等效 Batch Size)
    "block_size": 768,       # 長序列測試基準
    "lr": 4e-4,
    "epochs": 20000,         # Stage 1 架構驗證
    "data_path": "data/wiki_50MB.txt",  
    "save_model": "d2_v10_genesis.pth",
    "vocab_name": "master_vocab_v10.json",
    "l1_lambda": 0.0001,     # 稍微調低，因為我們改為懲罰 pre-norm gate
    "min_freq": 3            
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. 字典建立 (省略重複的 Print，與 V9 邏輯相同) ---
with open(config["data_path"], 'r', encoding='utf-8') as f:
    all_text = f.read()

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

# --- 2. Gated-D2 架構 (V10 Linear Attention 版) ---
class CausalGatedLinearAttentionV10(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必須能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.concept_gate = nn.Linear(d_model, d_model) 
        self.proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        
        self.gate_sparsity = 0.0 # 用於儲存 L1 懲罰值
        
    def forward(self, x):
        B, T, D = x.shape
        x_norm = self.ln(x)
        
        # 1. 產生 Q, K, V 並拆分多頭
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2) # (B, H, T, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # 2. 🌟 Concept Gate 處理 (Q 與 K 雙向控制)
        gate_raw = self.concept_gate(x)
        gate_sig = torch.sigmoid(gate_raw)
        
        # 記錄未正規化的 sparsity 供 L1 Loss 使用 (解決 L1 與 Norm 的衝突)
        self.gate_sparsity = gate_sig.mean()
        
        # Gate Normalization (避免神經元全部休眠)
        gate_norm = gate_sig / (gate_sig.mean(dim=-1, keepdim=True) + 1e-6)
        gate_norm = gate_norm.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # 套用 Gate
        q = q * gate_norm
        k = k * gate_norm
        
        # 3. 🌟 V10 Kernel: ReLU(x)^2
        q = F.relu(q)**2 + 1e-6
        k = F.relu(k)**2 + 1e-6
        
        # 4. 🌟 O(N) Linear Attention 因果計算 (Prefix Memory)
        # S = sum(k ⊗ v) 
        # 使用 einsum 進行外積，並用 cumsum 沿著時間維度 T 累加
        kv = torch.einsum('b h t d, b h t m -> b h t d m', k, v)
        kv_cumsum = torch.cumsum(kv, dim=2) 
        out_num = torch.einsum('b h t d, b h t d m -> b h t m', q, kv_cumsum)
        
        # z = sum(k)
        k_cumsum = torch.cumsum(k, dim=2)
        out_den = torch.einsum('b h t d, b h t d -> b h t', q, k_cumsum).unsqueeze(-1) + 1e-6
        
        out = out_num / out_den
        
        # 合併多頭
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(), # V11 若要升級可考慮 SwiGLU
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

# --- 3. V10 創世紀點火 (加入 3060 榨汁機優化) ---
model = D2V10Model(vocab_size, config["d_model"], config["n_layers"], config["n_heads"]).to(device)

# 推薦使用 torch.compile 提升 15-30% 速度 (若你的 PyTorch 版本支援)
# model = torch.compile(model) 

optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.05)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
scaler = torch.cuda.amp.GradScaler() # 🌟 加入混合精度，節省顯存

print(f"🚀 V10 Implicit MoE 創世紀啟動！當前規模: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

model.train()
optimizer.zero_grad()

try:
    for epoch in range(config["epochs"]):
        xb, yb = get_batch() 
        
        # 🌟 使用 autocast 啟動混合精度計算
        with torch.cuda.amp.autocast():
            logits = model(xb)
            loss_ce = criterion(logits.view(-1, vocab_size), yb.view(-1))
            
            # 收集未正規化的 Gate Sparsity
            l1_loss = sum(block.attn.gate_sparsity for block in model.blocks) / config["n_layers"]
            
            # 梯度累積計算 (Loss 平均化)
            total_loss = (loss_ce + config["l1_lambda"] * l1_loss) / config["accum_steps"]
        
        # 使用 Scaler 反向傳播
        scaler.scale(total_loss).backward()
        
        # 🌟 梯度累積 (Gradient Accumulation) 達到等效大 Batch Size
        if (epoch + 1) % config["accum_steps"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        if epoch % 100 == 0:
            print(f"🚀 V10 | Step {epoch:05d} | Loss: {loss_ce.item():.4f} | Gate_Act: {l1_loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

except KeyboardInterrupt:
    print("\n⚠️ 訓練中斷，正在封裝 V10 大腦...")

torch.save(model.state_dict(), config["save_model"])
with open(config["vocab_name"], "w", encoding="utf-8") as f:
    json.dump({"chars": vocab, "char_to_int": char_to_int, "int_to_char": {str(k): v for k, v in int_to_char.items()}}, f, ensure_ascii=False)
