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
            

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size)
    
    # This would get (B,T) dimensional idx/targets to predict next tokens
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B,T,C)
        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C) # (B*T,C)
            targets = targets.view(B*T) # (B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens=200):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # focus only on last timestep
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next))
        return idx

model = GPTLanguageModel()
print(f"Number of parameters in this model: {sum(p.numel() for p in model.parameters())/1e6}M.")

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
