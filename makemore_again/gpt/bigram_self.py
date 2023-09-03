import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 32
block_size = 8
torch.manual_seed(1337)
max_iters = 10000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
eval_interval = 200

# data prep
text = open("../../makemore/gpt/input.txt", "r").read()
chars = sorted(list(set(''.join(text))))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda s: ''.join([itos[i] for i in s])
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(text) * 0.9)
train_data = data[:n]
val_data   = data[n:]

def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i  :  i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y
xb,yb = get_batch("train")


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) # initializing zeros tensor for storing losses of eval_iters number of batches
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# simple bigram LM
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) #(vocab_size, vocab_size) <--> (C,C)
    
    def forward(self, idx, targets=None): # passing idx of shape (B,T)
        
        # idx & targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C) : C is vocab_size here, but can differ
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C) # (B*T, C)
            targets = targets.view(B*T) # (B*T)
            # print(f'Shape of logits (after reshape)is: {logits.shape} and of targets (after reshape) is: {targets.shape}')
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens = 100):
        # idx is (B,T) tensor, which is T time dims for each context in a batch
        # And we wish to generate T+1, T+2,  ..so on tokens for each context in a batch
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            # focus only on the last timestep: as this is bigram model
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# initialize the model
model = BigramLanguageModel(vocab_size=vocab_size)
model = model.to(device)
logits, loss = model(xb, yb)
# print(f'Shape of xb is: {xb.shape} and of yb is: {yb.shape}')
print(logits.shape)
print(loss.item())

# train the model
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb,yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
# print(loss.item())


# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))
