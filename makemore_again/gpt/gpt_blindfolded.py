import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## hparams
eval_iters = 200
eval_interval = 200
block_size = 8
batch_size = 4
learning_rate = 1e-3
max_iters = 10000
n_embd = 32
num_heads = 4
dropout = 0.2
n_layer = 4

## reading data and processing
text = open('../../makemore/gpt/input.txt', 'r').read()
# building vocab
chars = sorted(list(set(''.join(text))))
vocab_size = len(chars)
# building string - int mapping & vice-versa
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
# tokenizer & inverse
encode = lambda s: [stoi[i] for i in s]
decode = lambda list_i: ''.join([itos[i] for i in list_i])
# tokenizing + vectorizing
data = torch.tensor(encode(text), dtype=torch.long)
# train-test split of data
n = int(len(data) * 0.8)
train_data = data[:n]
val_data = data[n:]
# get_batch method
def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(0, len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i  :i+  block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x,y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x,y = get_batch(split)
            logits, loss = model(x,y)
            losses[i] = loss
        out[split] = losses.mean()
    model.train()
    return out


## self attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,head_size) @ (B,head_size,T) --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)

        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B,T,T) @ (B,T,head_size) --> (B,T,head_size)
        return out


## multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B,T,head_size*num_heads)
        out = self.dropout(self.proj(out))
        return out


## feed forward
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x) # (B,T,n_embd)
    

## block
class Blocks(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads=num_heads, head_size=head_size)
        self.ffwd = FeedForward(n_embd=n_embd)
        self.ln1 = nn.LayerNorm(normalized_shape=n_embd)
        self.ln2 = nn.LayerNorm(normalized_shape=n_embd)

    def forward(self, x):
        x = x + self.ln1(self.sa(x)) # (B,T,n_embd)
        x = x + self.ln2(self.ffwd(x)) # (B,T,n_embd)
        return x


## model class
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd)
        self.position_embedding_table = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embd)
        self.blocks = nn.Sequential(
            *[Blocks(n_embd=n_embd, num_heads=num_heads) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(normalized_shape=n_embd)
        self.lm_head = nn.Linear(in_features=n_embd, out_features=vocab_size)

    # This would get (B,T) dimensional idx/targets to predict next tokens
    def forward(self, idx, targets=None):

        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,n_embd)
        x = tok_emb + pos_emb # (B,T,n_embd)
        x = self.blocks(x) # (B,T,n_embd)
        x = self.ln_f(x) # (B,T,n_embd)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C) # (B*T,C)
            targets = targets.view(B*T) # (B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens = 100):
        # idx is (B,T) tensor, which is T time dims for each context in a batch
        # And we wish to generate T+1, T+2,  ..so on tokens for each context in a batch
        for _ in range(max_new_tokens):
            # crop the idx for last block_size (T) timesteps
            idx_cond = idx[ : , -block_size : ]
            logits, loss = self(idx_cond)
            # focus only on the last timestep: as this is bigram model
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


## model init
model = GPTLanguageModel()
model = model.to(device)
print(f"Number of parameters in this model: {sum(p.numel() for p in model.parameters())/1e6}M.")


## training
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss: {losses['train']:.4f} val loss: {losses['val']:.4f}")

    xb,yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))