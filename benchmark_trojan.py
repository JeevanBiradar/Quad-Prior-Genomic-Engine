import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import time

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
DNA_VOCAB = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, '<PAD>': 5, '<CLS>': 6}
VOCAB_SIZE = len(DNA_VOCAB)

SEQ_LEN = 1024 
EMBD_DIM = 256
NUM_HEADS = 4
LAYERS = 4
N_BUCKETS = 16 
URDHVA_BLOCKS = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Executing Trojan Horse Benchmark on: {device}")

# =====================================================================
# 2. QUAD-PRIOR ENGINE (Urdhva + Pingala + Ramanujan + Panini)
# =====================================================================
class Urdhva_Linear(nn.Module):
    def __init__(self, dim, num_blocks):
        super().__init__()
        self.dim, self.num_blocks, self.block_size = dim, num_blocks, dim // num_blocks
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
        return (y_vertical + y_cross_right + y_cross_left).reshape(orig_shape) + self.bias

class QuadPrior_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len):
        super().__init__()
        self.head_dim, self.num_heads = embed_dim // num_heads, num_heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = [Urdhva_Linear(embed_dim, URDHVA_BLOCKS) for _ in range(4)]
        
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
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) + self.sparse_mask[:T, :T]
        return self.out_proj((F.softmax(scores, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C))

class QuadPrior_Block(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.attn = QuadPrior_Attention(embed_dim, num_heads, seq_len)
        self.ffn = nn.Sequential(Urdhva_Linear(embed_dim, URDHVA_BLOCKS), nn.GELU(), Urdhva_Linear(embed_dim, URDHVA_BLOCKS))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))

class PaniniGenomicGrammar(nn.Module):
    def __init__(self, n_buckets, embd, vocab_size, seq_len, rank=8):
        super().__init__()
        self.bucket_size, self.n_buckets = max(1, seq_len // n_buckets), n_buckets
        self.A = nn.Parameter(torch.randn(n_buckets, embd, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, vocab_size) * 0.01)
        self.alpha = nn.Parameter(torch.tensor(0.05)) 

    def forward(self, hidden):
        bucket_idx = (torch.arange(hidden.shape[1], device=hidden.device) // self.bucket_size).clamp(max=self.n_buckets - 1)
        return self.alpha * (torch.einsum('bte,ter->btr', hidden, self.A[bucket_idx]) @ self.B)

class QuadPrior_GenomicEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb, self.pos_emb = nn.Embedding(VOCAB_SIZE, EMBD_DIM), nn.Embedding(SEQ_LEN, EMBD_DIM)
        self.blocks = nn.Sequential(*[QuadPrior_Block(EMBD_DIM, NUM_HEADS, SEQ_LEN) for _ in range(LAYERS)])
        self.ln_final = nn.LayerNorm(EMBD_DIM)
        self.lm_head = Urdhva_Linear(EMBD_DIM, URDHVA_BLOCKS)
        self.final_proj = nn.Linear(EMBD_DIM, VOCAB_SIZE, bias=False)
        self.panini = PaniniGenomicGrammar(N_BUCKETS, EMBD_DIM, VOCAB_SIZE, SEQ_LEN)

    def forward(self, x, targets=None):
        h = self.blocks(self.token_emb(x) + self.pos_emb(torch.arange(x.shape[1], device=x.device)))
        logits = self.final_proj(self.lm_head(self.ln_final(h))) + self.panini(h)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1)) if targets is not None else None
        return logits, loss

# =====================================================================
# 3. DATASET & EXECUTION
# =====================================================================
def fetch_genome(ncbi_id, name):
    print(f"Fetching {name} ({ncbi_id})...")
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={ncbi_id}&rettype=fasta&retmode=text"
    try:
        lines = urllib.request.urlopen(url).read().decode('utf-8').split('\n')
        time.sleep(1) 
        return "".join([line.strip() for line in lines[1:] if line.strip()])
    except: return ""

class GenomicDataset(Dataset):
    def __init__(self, sequence_string, seq_len=1024):
        self.tokens = [DNA_VOCAB.get(base, 4) for base in sequence_string]
        self.chunks = [self.tokens[i : i + seq_len + 1] for i in range(0, len(self.tokens) - seq_len - 1, seq_len // 2)]
    def __len__(self): return len(self.chunks)
    def __getitem__(self, idx):
        return torch.tensor(self.chunks[idx][:-1]), torch.tensor(self.chunks[idx][1:])

print("\n" + "█"*50)
print("BUILDING HOST (HUMAN) AND ALIEN (HIV) GENOMES")
print("█"*50)
# Training on Human Chromosome 1 segment (Host)
human_host = fetch_genome("NC_000001.11&seq_start=1000000&seq_stop=1100000", "Human Host DNA")
# The payload we will secretly inject later
hiv_virus = fetch_genome("NC_001802.1", "HIV-1 Genome (Alien Payload)")

train_loader = DataLoader(GenomicDataset(human_host), batch_size=8, shuffle=True)
model = QuadPrior_GenomicEngine().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print("\nTRAINING: LEARNING THE HUMAN HOST GRAMMAR (150 STEPS)")
model.train()
for step, (x, y) in enumerate(train_loader):
    if step >= 150: break
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(); _, loss = model(x, y); loss.backward(); optimizer.step()
    if step % 30 == 0: print(f"  Step {step:03d} | Human Grammar Loss: {loss.item():.4f}")

print("\n" + "█"*50)
print("THE TROJAN HORSE TEST: VIRAL INTEGRATION")
print("█"*50)
# We take a 1024-token window. The first 512 tokens are Human. The last 512 are HIV.
# Can the model detect the exact moment the virus takes over?
spliced_seq = human_host[5000:5512] + hiv_virus[1000:1512]
test_tokens = [DNA_VOCAB.get(b, 4) for b in spliced_seq]

model.eval()
with torch.no_grad():
    x_in = torch.tensor([test_tokens[:-1]], dtype=torch.long).to(device)
    y_target = torch.tensor([test_tokens[1:]], dtype=torch.long).to(device)
    logits, _ = model(x_in, y_target)
    losses = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y_target.view(-1), reduction='none').cpu().numpy()

# Analyze the exact boundary
human_region_surprise = np.mean(losses[100:400]) # Deep in human territory
integration_site_surprise = np.mean(losses[508:516]) # The exact cut-and-paste boundary
alien_region_surprise = np.mean(losses[600:900]) # Deep in viral territory

print(f"Deep Human Territory Surprise:  {human_region_surprise:.4f}")
print(f"Viral Integration Boundary:     {integration_site_surprise:.4f}")
print(f"Deep Alien (HIV) Territory:     {alien_region_surprise:.4f}")

gap = ((alien_region_surprise - human_region_surprise) / human_region_surprise) * 100
print(f"\n🔥 RESULT: The Quad-Prior Engine detected a {gap:.1f}% massive grammar shift entering the HIV sequence.")
