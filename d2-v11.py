import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import glob
import math
from collections import Counter
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

# ==========================================
# 🚀 V11 穩定轉生配置 (180M 級別)
# 完美適配 RTX 3060 12GB 
# ==========================================
config = {
    "d_model": 768,          # 🧠 降至 768 (標準 180M 級別尺寸)
    "n_heads": 12,           # 🧠 配合 768 維度 (768 / 64 = 12頭)
    "n_layers": 24,          # 🧠 深度維持 24 層，強化邏輯推導 (總參數約 184M)
    "batch_size": 2,         
    "block_size": 768,       
    "accum_steps": 12,       
    "lr": 2e-4,              # 模型變小，可以稍微提速至 2e-4
    "epochs": 40000,         
    "warmup_steps": 2000,    # 前 2000 步熱身防爆
    "data_dir": "data",      
    "save_model": "d2_v11_genesis_180m.pth", # 更改檔名避免覆蓋 330M 的權重
    "vocab_name": "master_vocab_v11.json", 
    "l1_lambda": 0.0003,     
    "balance_lambda": 0.01,  
    "min_freq": 5            
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 運行設備: {device}")

# ==========================================
# 1. 資料讀取與全新詞表建立
# ==========================================
print(f"🔍 正在掃描 {config['data_dir']} 資料夾...")
all_text = ""
txt_files = glob.glob(os.path.join(config["data_dir"], "*.txt"))
if not txt_files:
    raise FileNotFoundError(f"❌ 在 {config['data_dir']} 資料夾中找不到任何 .txt 檔案！")

for file_path in txt_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        all_text += f.read() + "\n\n" 

if os.path.exists(config["vocab_name"]):
    print(f"🔒 偵測到舊有字典 {config['vocab_name']}，載入詞表！")
    with open(config["vocab_name"], "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
        vocab = vocab_data["chars"]
        char_to_int = vocab_data["char_to_int"]
        int_to_char = {int(k): v for k, v in vocab_data["int_to_char"].items()}
else:
    print("⚠️ 找不到舊字典，為 V11 重新建立全新詞表...")
    counter = Counter(all_text)
    valid_chars = sorted([ch for ch, count in counter.items() if count >= config["min_freq"]])
    vocab = ["[UNK]"] + valid_chars
    char_to_int = {ch: i for i, ch in enumerate(vocab)}
    int_to_char = {i: ch for i, ch in enumerate(vocab)}
    with open(config["vocab_name"], "w", encoding="utf-8") as f:
        json.dump({"chars": vocab, "char_to_int": char_to_int, "int_to_char": int_to_char}, f, ensure_ascii=False)

vocab_size = len(vocab)
unk_id = char_to_int.get("[UNK]", 0)
data = torch.tensor([char_to_int.get(c, unk_id) for c in all_text], dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data) - config["block_size"], (config["batch_size"],))
    x = torch.stack([data[i:i+config["block_size"]] for i in ix])
    y = torch.stack([data[i+1:i+config["block_size"]+1] for i in ix])
    return x.to(device), y.to(device)

# ==========================================
# 2. V11 模型架構 (修正 NaN 與多頭機制)
# ==========================================
class CausalGatedD2Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.d_head = d_model // self.n_heads
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.concept_gate = nn.Linear(d_model, d_model) 
        self.proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        
    def forward(self, x):
        B, L, D = x.shape
        x_norm = self.ln(x)
        
        # 取得 Q, K, V
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        
        # 隱式 MoE: 概念門控 (Gating)
        gate = torch.sigmoid(self.concept_gate(x))
        k = k * gate 
        
        # 切分為多頭 [Batch, Heads, SeqLen, d_head]
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        # --- 🌟 正確的因果線性注意力 (Causal Linear Attention) ---
        # 為了避免溢位，強制轉為 FP32 進行核心運算
        q_f, k_f, v_f = q.float(), k.float(), v.float()
        
        # 確保 Q, K 為正數 (核函數映射)
        q_f = F.elu(q_f) + 1.0
        k_f = F.elu(k_f) + 1.0
        
        # 【分子計算】計算 K^T * V 的因果累積 (這是 d x d 的核心！)
        # 利用 unsqueeze 創造 [B, H, L, d, d] 的狀態矩陣
        kv_state = k_f.unsqueeze(-1) * v_f.unsqueeze(-2)  
        kv_cumsum = torch.cumsum(kv_state, dim=2) # 沿著時間維度 L 進行因果累積
        
        # Q 去乘上每一刻的累積狀態矩陣
        out_num = torch.matmul(q_f.unsqueeze(-2), kv_cumsum).squeeze(-2) # [B, H, L, d]
        
        # 【分母計算】計算 K 的因果累積
        k_cumsum = torch.cumsum(k_f, dim=2)
        out_den = (q_f * k_cumsum).sum(dim=-1, keepdim=True) + 1e-6
        
        # 兩者相除 (這時分子和分母的因果時間軸完全對齊，絕不會爆炸)
        attn_out = out_num / out_den
        
        # --- 轉回原資料型態並合併多頭 ---
        attn_out = attn_out.to(x.dtype)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        
        return self.proj(attn_out), gate

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

class D2V11Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = CausalGatedD2Attention(d_model)
        self.mlp = MLP(d_model)
    def forward(self, x):
        attn_out, gate = self.attn(x)
        x = x + attn_out
        x = x + self.mlp(x)
        return x, gate

class D2V11Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([D2V11Block(d_model) for _ in range(n_layers)])
        self.out_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        gates = []
        
        for block in self.blocks:
            # 🛡️ 啟動梯度檢查點！
            # PyTorch 不會再死記硬背那龐大的 [L, d, d] 矩陣
            # 這會直接把 VRAM 佔用砍掉一半以上！
            x, gate = checkpoint(block, x, use_reentrant=False)
            gates.append(gate)
            
        logits = self.head(self.out_ln(x))
        
        all_gates = torch.stack(gates) 
        sparse_loss = all_gates.mean()
        balance_loss = all_gates.mean(dim=(0, 1, 2)).var() 
        
        return logits, sparse_loss, balance_loss

# ==========================================
# 3. 初始化與防爆優化器設定
# ==========================================
model = D2V11Model(vocab_size, config["d_model"], config["n_layers"]).to(device)

if os.path.exists(config["save_model"]):
    model.load_state_dict(torch.load(config["save_model"], map_location=device, weights_only=True))
    print("✅ 成功載入 V11 舊有權重！")
else:
    print(f"🌟 初始化全新 V11 大腦！(參數規模約 {sum(p.numel() for p in model.parameters())/1e6:.1f}M)")

optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)


# --- 自動學習率排程器 (Warmup + Cosine Decay) ---
def lr_lambda(current_step):
    if current_step < config["warmup_steps"]:
        return float(current_step) / float(max(1, config["warmup_steps"]))
    progress = float(current_step - config["warmup_steps"]) / float(max(1, config["epochs"] - config["warmup_steps"]))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = LambdaLR(optimizer, lr_lambda)

# ==========================================
# 4. 訓練迴圈 (BF16 絕對防爆版)
# ==========================================
print("🚀 V11 轉生計畫啟動！(啟動 RTX 3060 專屬 BF16 絕對防護)")
model.train()

global_step = 0
while global_step < config["epochs"]:
    optimizer.zero_grad(set_to_none=True)
    
    total_loss = 0
    total_sparse = 0
    total_bal = 0
    
    for _ in range(config["accum_steps"]):
        xb, yb = get_batch()
        
        # 🌟 核心關鍵：強制使用 bfloat16，天花板提升至 10^38，永不溢位！
        with autocast('cuda', dtype=torch.bfloat16):
            logits, sparse_loss, balance_loss = model(xb)
            ce_loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            loss = ce_loss + config["l1_lambda"] * sparse_loss + config["balance_lambda"] * balance_loss
            loss = loss / config["accum_steps"]
            
        # 直接 backward，不需要 scaler！
        loss.backward()
        
        total_loss += ce_loss.item() 
        total_sparse += sparse_loss.item()
        total_bal += balance_loss.item()

    # --- 🌟 梯度裁剪 (防滾籠) ---
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

    # 乾淨俐落的 step，沒有警告，沒有囉嗦
    optimizer.step()
    scheduler.step()

    global_step += 1
    
    # 測試階段，我們先每 5 步印一次，確認它真的跑起來了！
    if global_step % 100 == 0:
        current_lr = scheduler.get_last_lr()[0] 
        avg_loss = total_loss / config["accum_steps"]
        avg_sparse = total_sparse / config["accum_steps"]
        avg_bal = total_bal / config["accum_steps"]
        print(f"🚀 V11 | Step {global_step:05d} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Sparse: {avg_sparse:.4f} | Bal: {avg_bal:.5f}")
        
    if global_step % 2000 == 0:
        torch.save(model.state_dict(), config["save_model"])
        print(f"💾 Step {global_step} 模型已儲存至 {config['save_model']}")

print("🎉 V11 訓練完成！")