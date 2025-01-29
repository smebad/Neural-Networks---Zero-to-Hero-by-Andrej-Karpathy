import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # number of samples in each batch
block_size = 8 # length of each input
max_iters = 3000 # number of training iterations
eval_interval = 300 # interval to evaluate the model
learning_rate = 1e-2 # learning rate for the optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200 # number of iterations to evaluate the model
n_embd = 32 # dimensionality of the character embeddings

torch.manual_seed(1337)

# Reading the data from the file
with open(r"C:\Users\DC\Desktop\Neural Networks - Zero to Hero by Andrej Karpathy\Neural Networks - Zero to Hero by Andrej Karpathy\lecture-07-building-GPT-from-scratch\input.txt", 'r', encoding='utf-8') as f:
    text = f.read()

# Unique characters that appear in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create mappings from characters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

# Encoding the entire dataset and storing it into a torch tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Splitting the data into training and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Implementing the Transformer model using PyTorch (Bigram Language Model)
class BigramLanguageModel(nn.Module):
    def __init__(self): # initialize the model
        super().__init__() # initialize the base class
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # embedding table
        self.lm_head = nn.Linear(n_embd, vocab_size) # linear layer to predict the next token

    def forward(self, idx, targets=None): # forward function for the model
        tok_emb = self.token_embedding_table(idx) # token embeddings
        logits = self.lm_head(tok_emb) # (B, T, C) - B is the batch size, T is the sequence length, C is the number of characters

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

model = BigramLanguageModel()
m = model.to(device)


# Training the model and creating a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
   
   # every once in a while we need to evaluate the model
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")


    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluating the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# Generating text using the model
context = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
