import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tokenizers import Tokenizer

# ==========================================
# 1. V12 規格配置 (對齊你的 d2-v12-qllm2-moe.py)
# ==========================================
config = {
    "d_model": 768,          
    "n_heads": 12,           
    "n_layers": 24,          
    "load_model": "d2_v12_qllm2_moe.pth", 
    "vocab_name": "bpe_tokenizer_v12.json"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. 載入 BPE 分詞器
# ==========================================
if not os.path.exists(config["vocab_name"]):
    raise FileNotFoundError(f"❌ 找不到 BPE 分詞器：{config['vocab_name']}")
tokenizer = Tokenizer.from_file(config["vocab_name"])
vocab_size = tokenizer.get_vocab_size()

# ==========================================
# 3. V12 推理專用模型架構 (拔除訓練用的累贅)
# ==========================================
class CausalGatedD2Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.d_head = d_model // self.n_heads
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.semantic_bank = nn.Linear(d_model, d_model * 2) 
        self.context_bank = nn.Linear(d_model, d_model * 2)  
        self.proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        
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
        
        return self.proj(attn_out) # 推理時不需要回傳 gate

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        ) # 推理時可拔除 Dropout
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
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight # 權重綁定
        
    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x) # 推理時不使用 checkpoint
        return self.head(self.out_ln(x))

# ==========================================
# 4. 載入模型權重
# ==========================================
model = D2V12Model(vocab_size, config["d_model"], config["n_layers"]).to(device)

if os.path.exists(config["load_model"]):
    print("⏳ 正在讀取 V12 權重...")
    checkpoint = torch.load(config["load_model"], map_location=device, weights_only=False)
    # 處理斷點續傳的字典結構
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        step = checkpoint.get('step', '未知')
        print(f"✅ 成功載入 V12 (180M) 權重！(來自 Step {step})")
    else:
        model.load_state_dict(checkpoint)
        print("✅ 成功載入 V12 (180M) 權重！")
else:
    raise FileNotFoundError(f"❌ 找不到權重檔：{config['load_model']}")

model.eval()

# ==========================================
# 5. 生成邏輯 (BPE 支援版)
# ==========================================
def generate(prompt, max_new_tokens=150, temp=0.7, top_k=10, rep_penalty=1.2): # 🌟 加入 rep_penalty
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    print(f"\nAI: {prompt}", end="", flush=True)
    generated_ids = []
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids[:, -512:]) 
            logits = logits[:, -1, :] / temp
            
            # 🛡️ 啟動重複懲罰 (Repetition Penalty)
            # 讓已經講過的字，出現機率降低，強迫它講新的字！
            for token_id in set(generated_ids + input_ids[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= rep_penalty
                else:
                    logits[0, token_id] *= rep_penalty
            
            # Top-K 採樣
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_id], dim=1)
            generated_ids.append(next_id.item())
            
            char = tokenizer.decode([next_id.item()])
            print(char, end="", flush=True)
            
            if char.strip() == "<|endoftext|>" or (len(generated_ids) >= 2 and tokenizer.decode(generated_ids[-2:]) == "\n\n"):
                break
                
    print("\n" + "-"*40)
    print("\n" + "-"*40)

# ==========================================
# 6. 互動介面
# ==========================================
print("--- 🚀 D2-V12 (BPE + MoE) 測試啟動 ---")
print("提示：模型目前訓練到 Step 6000 (Loss ~4.9)，可能會有語法框架，但邏輯尚未完全成熟。")
while True:
    p = input("\n✍️ 請輸入 prompt (輸入 quit 離開)：")
    if p.lower() == 'quit': break
    if not p.strip(): continue
    generate(p)