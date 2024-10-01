import torch
import torch.nn.functional as F
import torch.nn as nn

# hparams -----------
batch_size = 32
block_size = 8
lr = 1e-2
max_iters = 3000
eval_interval = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
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
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd)
        logits = self.lm_head(tok_emb) # (B,T,vocab_size)
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
            # getting the predictions
            logits, loss = self(idx)
            # focus on the last time step
            logits = logits[:,-1,:] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from distribution
            idx_nxt = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat([idx, idx_nxt], dim=1) # becomes (B, T+1)
        return idx

model = BigramLM()
m = model.to(device)
optimizer = torch.optim.AdamW(params=m.parameters(), lr=lr)

# training the model
for iter in range(max_iters):
    if iter%eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))