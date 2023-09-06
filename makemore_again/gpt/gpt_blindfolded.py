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
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd)
        self.position_embedding_table = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embd)
        self.lm_head = nn.Linear(in_features=n_embd, out_features=vocab_size)

    # This would get (B,T) dimensional idx/targets to predict next tokens
    def forward(self, idx, targets=None):

        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,n_embd)
        x = tok_emb + pos_emb
        logits = self.lm_head(x)
        
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

model = GPTLanguageModel()
model = model.to(device)
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


# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))