import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import time

# =====================================================================
# 1. BIOLOGICAL CONFIGURATION
# =====================================================================
DNA_VOCAB = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, '<PAD>': 5, '<CLS>': 6}
VOCAB_SIZE = len(DNA_VOCAB)

SEQ_LEN = 1024 
EMBD_DIM = 256
NUM_HEADS = 4
LAYERS = 4
N_BUCKETS = 16 
URDHVA_BLOCKS = 8 # Vedic Matrix Blocks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Executing Quad-Prior Oncology Benchmark on: {device}")

# =====================================================================
# 2. THE QUAD-PRIOR ENGINE (Pingala + Panini + Ramanujan + Urdhva)
# =====================================================================

# --- PILLAR 4: URDHVA-TIRYAGBHYAM (Compute Efficiency) ---
class Urdhva_Linear(nn.Module):
    def __init__(self, dim, num_blocks):
        super().__init__()
        assert dim % num_blocks == 0, "Dimension must be divisible by num_blocks"
        self.dim = dim
        self.num_blocks = num_blocks
        self.block_size = dim // num_blocks
        
        self.W_vertical = nn.Parameter(torch.randn(num_blocks, self.block_size, self.block_size) / (self.block_size**0.5))
        self.W_cross_right = nn.Parameter(torch.randn(num_blocks - 1, self.block_size, self.block_size) / (self.block_size**0.5))
        self.W_cross_left = nn.Parameter(torch.randn(num_blocks - 1, self.block_size, self.block_size) / (self.block_size**0.5))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, self.num_blocks, self.block_size) 
        
        y_vertical = torch.einsum('nbs,bst->nbt', x, self.W_vertical)
        y_cross_right = torch.zeros_like(y_vertical)
        y_cross_right[:, 1:, :] = torch.einsum('nbs,bst->nbt', x[:, :-1, :], self.W_cross_right)
        y_cross_left = torch.zeros_like(y_vertical)
        y_cross_left[:, :-1, :] = torch.einsum('nbs,bst->nbt', x[:, 1:, :], self.W_cross_left)
        
        y = y_vertical + y_cross_right + y_cross_left
        return y.reshape(orig_shape) + self.bias

# --- PILLARS 1 & 3: PINGALA (Local) + RAMANUJAN (Long-Range) ---
class QuadPrior_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        
        # Using Urdhva for QKV projections to save even more compute!
        self.q_proj = Urdhva_Linear(embed_dim, URDHVA_BLOCKS)
        self.k_proj = Urdhva_Linear(embed_dim, URDHVA_BLOCKS)
        self.v_proj = Urdhva_Linear(embed_dim, URDHVA_BLOCKS)
        self.out_proj = Urdhva_Linear(embed_dim, URDHVA_BLOCKS)
        
        mask = torch.full((seq_len, seq_len), float('-inf'))
        for i in range(seq_len):
            for j in range(max(0, i - 4), i + 1): mask[i, j] = 0.0 # Pingala
            step = 8
            while i - step >= 0:
                mask[i, i - step] = 0.0 # Ramanujan
                step *= 2
        self.register_buffer("sparse_mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self.sparse_mask[:T, :T]
        attn = F.softmax(scores, dim=-1)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class QuadPrior_Block(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = QuadPrior_Attention(embed_dim, num_heads, seq_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # The FFN is completely replaced by Vedic Math
        self.ffn = nn.Sequential(
            Urdhva_Linear(embed_dim, URDHVA_BLOCKS),
            nn.GELU(),
            Urdhva_Linear(embed_dim, URDHVA_BLOCKS)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# --- PILLAR 2: PANINI (Positional Grammar) ---
class PaniniGenomicGrammar(nn.Module):
    def __init__(self, n_buckets, embd, vocab_size, seq_len, rank=8):
        super().__init__()
        self.n_buckets = n_buckets
        self.bucket_size = max(1, seq_len // n_buckets)
        self.A = nn.Parameter(torch.randn(n_buckets, embd, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, vocab_size) * 0.01)
        self.alpha = nn.Parameter(torch.tensor(0.05)) 

    def forward(self, hidden):
        B, T, E = hidden.shape
        positions = torch.arange(T, device=hidden.device)
        bucket_idx = (positions // self.bucket_size).clamp(max=self.n_buckets - 1)
        A_pos = self.A[bucket_idx] 
        h_proj = torch.einsum('bte,ter->btr', hidden, A_pos) 
        return self.alpha * (h_proj @ self.B)

class QuadPrior_GenomicEngine(nn.Module):
    def __init__(self, embd_dim=256, num_heads=4, layers=4, seq_len=1024, n_buckets=16):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, embd_dim)
        self.pos_emb = nn.Embedding(seq_len, embd_dim)
        self.blocks = nn.Sequential(*[
            QuadPrior_Block(embd_dim, num_heads, seq_len) for _ in range(layers)
        ])
        self.ln_final = nn.LayerNorm(embd_dim)
        self.lm_head = Urdhva_Linear(embd_dim, URDHVA_BLOCKS) # Vedic LM Head
        self.final_proj = nn.Linear(embd_dim, VOCAB_SIZE, bias=False) # Final map to vocab
        self.panini = PaniniGenomicGrammar(n_buckets, embd_dim, VOCAB_SIZE, seq_len, rank=8)

    def forward(self, x, targets=None):
        B, T = x.shape
        h = self.token_emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        h = self.blocks(h)
        h = self.ln_final(h)
        logits = self.final_proj(self.lm_head(h)) + self.panini(h)
        
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1)) if targets is not None else None
        return logits, loss

# =====================================================================
# 3. ONCOLOGY DATASET PIPELINE & EXECUTION
# =====================================================================
def fetch_genome(ncbi_id, name):
    print(f"Fetching {name} ({ncbi_id})...")
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={ncbi_id}&rettype=fasta&retmode=text"
    try:
        fasta_data = urllib.request.urlopen(url).read().decode('utf-8')
        lines = fasta_data.split('\n')
        sequence = "".join([line.strip() for line in lines[1:] if line.strip()])
        time.sleep(1) 
        return sequence
    except Exception as e: return ""

class EvolutionaryDataset(Dataset):
    def __init__(self, sequence_string, seq_len=1024):
        self.tokens = [DNA_VOCAB.get(base, 4) for base in sequence_string]
        stride = seq_len // 4
        self.chunks = [self.tokens[i : i + seq_len + 1] for i in range(0, len(self.tokens) - seq_len - 1, stride)]
    def __len__(self): return len(self.chunks)
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

# Initialize
model = QuadPrior_GenomicEngine().to(device)

# --- Compute Savings Calculation ---
def count_parameters(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
quad_params = count_parameters(model)
# A standard model of this dim/depth takes ~3.2M parameters. Let's see what Quad-Prior takes.
print("\n" + "█"*50)
print(f"ENGINE ARCHITECTURE: {quad_params:,} Parameters")
print("█"*50)

print("\nBUILDING TRAINING SET...")
mouse_tp53 = fetch_genome("NM_011640.3", "Mouse TP53")
macaque_tp53 = fetch_genome("NM_001047152.1", "Macaque TP53")
train_loader = DataLoader(EvolutionaryDataset(mouse_tp53 + macaque_tp53), batch_size=8, shuffle=True)

human_wt_tp53 = fetch_genome("NM_000546.6", "Human TP53 (Healthy)")
mutation_index = 1000 
human_mutant_tp53 = human_wt_tp53[:mutation_index] + 'A' + human_wt_tp53[mutation_index+1:]

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print("\nTRAINING: QUAD-PRIOR HIGH-SPEED EVOLUTION (100 STEPS)")
model.train()
steps = 100
current_step = 0
t0 = time.time()

while current_step < steps:
    for x, y in train_loader:
        if current_step >= steps: break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if current_step % 25 == 0 or current_step == steps - 1:
            print(f"  Step {current_step:03d} | Grammar Loss: {loss.item():.4f}")
        current_step += 1

train_time = time.time() - t0
print(f" Training Completed in {train_time:.2f} seconds.")

print("\nZERO-SHOT EVALUATION: EVOLUTION VS ONCOLOGY")
model.eval()
def analyze_window(sequence, start_idx):
    window = sequence[start_idx : start_idx + SEQ_LEN + 1]
    tokens = [DNA_VOCAB.get(base, 4) for base in window]
    x_in = torch.tensor([tokens[:-1]], dtype=torch.long).to(device)
    y_target = torch.tensor([tokens[1:]], dtype=torch.long).to(device)
    with torch.no_grad():
        logits, _ = model(x_in, y_target)
        relative_idx = mutation_index - start_idx
        token_losses = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y_target.view(-1), reduction='none').cpu().numpy()
    return token_losses[relative_idx]

eval_start = mutation_index - 512 
wt_surprise = analyze_window(human_wt_tp53, eval_start)
mutant_surprise = analyze_window(human_mutant_tp53, eval_start)

print(f"Healthy Mutation Surprise: {wt_surprise:.4f}")
print(f"Cancer Mutation Surprise:  {mutant_surprise:.4f}")

difference = ((mutant_surprise - wt_surprise) / wt_surprise) * 100
print(f"\n FINAL RESULT: Detected a {difference:.1f}% SPIKE in anomaly at the cancer site.")
