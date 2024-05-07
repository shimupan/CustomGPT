import torch
import torch.nn as nn
from torch.nn import functional as F
from prepare_data import Data

BLOCK_SIZE = 8
BATCH_SIZE = 32
N_EMBD = 32
MAX_ITERATION = 5000
LEARNING_RATE = 1e-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200

torch.manual_seed(1)

    

text = None
with open("test.txt","r",encoding="utf-8") as file:
    text = file.read()

data = Data(text)
print(data.vocabulary_size)

# Generate batches of data from the testing data
# With each testing block following an expected result
def get_batch(split):
    curr_split = None
    if split == 'test':
        curr_split = data.testing_data
    else:
        curr_split = data.training_data
    
    random_sample = torch.randint(len(curr_split) - BLOCK_SIZE, (BATCH_SIZE,))

    testing_block = torch.stack([curr_split[i:i+BLOCK_SIZE] for i in random_sample])
    expected_block = torch.stack([curr_split[i+1:i+BLOCK_SIZE+1] for i in random_sample])
    testing_block, expected_block = testing_block.to(DEVICE), expected_block.to(DEVICE)
    return testing_block, expected_block

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Single head of self attention
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute attention scores
        weight = q @ k.transpose(-2,-1) * C**-0.5
        # Decoder Block
        weight = weight.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        v = self.value(x)
        out = weight @ v
        return out

# Multi-head Self Attention
class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self,x):
        return torch.cat([h(x) for h in self.heads], dim=-1)   

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(data.vocabulary_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.sa_head = MultiHeadAttention(4,N_EMBD//4)
        self.lm_head = nn.Linear(N_EMBD, data.vocabulary_size)
    
    def forward(self, idx, targets=None):

        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.sa_head(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -BLOCK_SIZE:]
            # get the predictions
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(DEVICE)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for iter in range(MAX_ITERATION):

    if iter % EVAL_ITERS == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(data.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    