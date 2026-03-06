import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from collections import Counter

# --- V10 創世紀配置區 (專為 RTX 3060 12GB 設計) ---
config = {
    "d_model": 896,          
    "n_heads": 14,           
    "n_layers": 16,
    "batch_size": 2,         
    "accum_steps": 6,        
    "block_size": 768,       
    "lr": 4e-4,
    "epochs": 20000,         
    "data_dir": "data",      # 🌟 修改這裡：指向整個 data 資料夾
    "save_model": "d2_v10_genesis.pth",
    "vocab_name": "master_vocab_v10.json",
    "l1_lambda": 0.0001,     
    "min_freq": 5            # 🌟 建議設為 5：因為古文生僻字多，提高門檻保護顯存
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. 字典建立與資料讀取 ---
print(f"🔍 正在掃描 {config['data_dir']} 資料夾，準備吸收集體知識...")
all_text = ""

# 自動尋找資料夾下所有的 txt 檔案
txt_files = glob.glob(os.path.join(config["data_dir"], "*.txt"))

if not txt_files:
    raise FileNotFoundError(f"❌ 在 {config['data_dir']} 資料夾中找不到任何 .txt 檔案！")

for file_path in txt_files:
    print(f"   📖 正在讀取: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        # 用換行符號隔開不同檔案的內容，避免句意連在一起
        all_text += f.read() + "\n\n" 

print("📊 正在統計字頻並啟動『顯存保護機制』...")
counter = Counter(all_text)

# 過濾掉出現次數低於 min_freq 的生僻字
valid_chars = sorted([ch for ch, count in counter.items() if count >= config["min_freq"]])

# 加入 [UNK] (Unknown) 作為兜底符號
vocab = ["[UNK]"] + valid_chars
vocab_size = len(vocab)
char_to_int = {ch: i for i, ch in enumerate(vocab)}
int_to_char = {i: ch for i, ch in enumerate(vocab)}
unk_id = char_to_int["[UNK]"]

print(f"✅ 字典淬鍊完成！")
print(f"   - 總文本長度: {len(all_text) / (1024*1024):.2f} MB")
print(f"   - 原始字元總數: {len(counter)}")
print(f"   - 過濾後字典大小: {vocab_size} (已消除 {len(counter) - vocab_size} 個生僻垃圾字元)")

# --- 2. 數據轉換 (下方接續原本的程式碼) ---
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
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # 2. 🌟 Concept Gate 處理
        gate_raw = self.concept_gate(x)
        gate_sig = torch.sigmoid(gate_raw)
        
        self.gate_sparsity = gate_sig.mean()
        
        # EPS 稍微調大一點點，避免除以極小值
        gate_norm = gate_sig / (gate_sig.mean(dim=-1, keepdim=True) + 1e-5)
        gate_norm = gate_norm.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        q = q * gate_norm
        k = k * gate_norm
        
        # ==========================================
        # 🚑 救命仙丹：強制將 Q, K, V 轉為 FP32 進行累加計算
        # ==========================================
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        
        # 3. 🌟 V10 Kernel: ReLU(x)^2
        q = F.relu(q)**2 + 1e-5
        k = F.relu(k)**2 + 1e-5
        
        # 4. 🌟 O(N) Linear Attention 因果計算
        kv = torch.einsum('b h t d, b h t m -> b h t d m', k, v)
        kv_cumsum = torch.cumsum(kv, dim=2) 
        out_num = torch.einsum('b h t d, b h t d m -> b h t m', q, kv_cumsum)
        
        k_cumsum = torch.cumsum(k, dim=2)
        out_den = torch.einsum('b h t d, b h t d -> b h t', q, k_cumsum).unsqueeze(-1) + 1e-5
        
        out = out_num / out_den
        
        # 算完之後，把結果轉回原本的資料型態 (FP16)
        out = out.to(x.dtype)
        # ==========================================
        
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
        # Block 的任務是單純的特徵提取與殘差連接
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
        # Model 才是負責把 Token 轉 Embedding，然後進入 Checkpoint 迴圈的地方
        x = self.embedding(x)
        
        # 🌟 啟動 Gradient Checkpointing，完美壓制顯存
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            
        return self.head(self.out_ln(x))

# --- 3. V10 創世紀點火 (加入 3060 榨汁機優化) ---
model = D2V10Model(vocab_size, config["d_model"], config["n_layers"], config["n_heads"]).to(device)

# 推薦使用 torch.compile 提升 15-30% 速度 (若你的 PyTorch 版本支援)
# model = torch.compile(model) 

optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.05)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
scaler = torch.amp.GradScaler('cuda') 

print(f"🚀 V10 Implicit MoE 創世紀啟動！當前規模: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

model.train()
optimizer.zero_grad()

try:
    for epoch in range(config["epochs"]):
        xb, yb = get_batch() 
        
        # 🌟 使用 autocast 啟動混合精度計算
        with torch.amp.autocast('cuda'):
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
