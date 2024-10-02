import torch
import torch.nn.functional as F
import torch.nn as nn

# hparams -----------
batch_size = 32
block_size = 8
lr = 1e-3
max_iters = 5000
eval_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
head_size = 32
num_heads = 4
# ------------
torch.manual_seed(1337)

text = open("input.txt", 'r').read()

vocab = sorted(list(set(''.join(text))))
vocab_size = len(vocab)
stoi = {s:i for i,s in enumerate(vocab)}
itos = {i:s for i,s in enumerate(vocab)}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text))
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+1+block_size] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split)
            logits, loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLM(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_heads = MultiHeadedSelfAttention(num_heads=num_heads)
        # self.ffwd = FeedForward(n_embd=n_embd)
        self.blocks = nn.Sequential(
            Block(num_heads=num_heads),
            Block(num_heads=num_heads),
            Block(num_heads=num_heads),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,n_embd)
        x = tok_emb + pos_emb # (B,T,n_embd)
        # x = self.sa_heads(x) # (B,T,n_embd)
        # x = self.ffwd(x) # (B,T,n_embd)
        x = self.blocks(x) # (B,T,n_embd)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # getting the predictions
            logits, loss = self(idx_cond)
            # focus on the last time step
            logits = logits[:,-1,:] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from distribution
            idx_nxt = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat([idx, idx_nxt], dim=1) # becomes (B, T+1)
        return idx

class Head(nn.Module):
    # one head of self-attention
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        # calculating attention scores ('affinities')
        wei = q @ k.transpose(-2,-1) * head_size**-0.5 # (B,T,head_size) @ # (B,head_size,T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B,T,T) @ (B,T,head_size) ---> (B,T,head_size)
        return out

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.head_size = head_size // num_heads
        self.heads = nn.ModuleList([Head(self.head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    # TRANSFORMER BLOCK: communication followed by computation
    def __init__(self, num_heads):
        super().__init__()
        self.sa = MultiHeadedSelfAttention(num_heads=num_heads)
        self.ffwd = FeedForward(n_embd=n_embd)
    
    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x


# training -----------------
model = BigramLM()
m = model.to(device)
optimizer = torch.optim.AdamW(params=m.parameters(), lr=lr)

# training the model
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generation ---------------
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))