import torch # get_batches
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from data_preprocessing import preprocess_corpus, get_batches
import configparser

argparser = argparse.ArgumentParser(description="A Language Model that generates Burna Boy like lyrics")
subparsers = argparser.add_subparsers(title="Mode", dest="command",required=True, help='Language Model modes')

#training specifics
train_parser = subparsers.add_parser('train',help="Train the model with specific configs")
train_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for training")
train_parser.add_argument('--epoch', type=int, default=2000,help="Number of trainning epochs")
train_parser.add_argument('--b_size', type=int, default=64, help="Batch size for training")
train_parser.add_argument('--n_heads', type=int, default=6, help="Number of head in the masked self-attention")
train_parser.add_argument('--em_dims', type=int, default=384, help="Models embedding dimensions ")
train_parser.add_argument('--n_layers', type=int, default=6, help="Number of trasformer layers")
train_parser.add_argument('--cntx_len', type=int, default=256, help="Transformers context length")
train_parser.add_argument('--save', type=bool, default=True, help="Save model weights")
train_parser.add_argument('--fresh', type=bool, default=False, help="To start training fresh")

#testing specifics
test_parser = subparsers.add_parser('test',help="Run inference on the Model")
test_parser.add_argument('--check_pt', type=str, help="Path to a checkpoint file for inference")
test_parser.add_argument('--phrase', type=str, help="Starting phrase for inference")

args = argparser.parse_args()


BASE_DIR = os.getcwd()
WEIGHTS_CHKPT = 'BBLM.pth'
dsplt, encode, decode, chars = preprocess_corpus('lyrics.txt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
em_dims = args.em_dims
context_len = args.cntx_len
batch_size = args.b_size
n_head = args.n_heads
drop_fc = 0.2
vocab_size = len(chars)
n_layers = args.n_layers
learning_rate = args.lr
epochs = args.epoch





class SelfAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(em_dims, head_size, bias=False)
        self.query = nn.Linear(em_dims, head_size, bias=False)
        self.value = nn.Linear(em_dims, head_size, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(context_len, context_len, device=device)))
        self.dropout = nn.Dropout(drop_fc)

    def forward(self, x):
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)
        weights = query @ key.transpose(-2, -1) * (query.size(-1) ** -0.5)
        weights = weights.masked_fill(self.mask[:x.size(1), :x.size(1)] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        return weights @ value

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(em_dims, em_dims)
        self.dropout = nn.Dropout(drop_fc)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, em_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(em_dims, em_dims * 4),
            nn.ReLU(),
            nn.Linear(em_dims * 4, em_dims)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, em_dims, n_head):
        super().__init__()
        head_size = em_dims // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedForward(em_dims)
        self.layer_norm1 = nn.LayerNorm(em_dims)
        self.layer_norm2 = nn.LayerNorm(em_dims)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x

class BurnaBoyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, em_dims)
        self.position_encodings = nn.Embedding(context_len, em_dims)
        self.blocks = nn.Sequential(*[Block(em_dims, n_head) for _ in range(n_layers)])
        self.final_layernorm = nn.LayerNorm(em_dims)
        self.lm_head = nn.Linear(em_dims, vocab_size)

    def forward(self, idx, targets=None):
        token_emb = self.embedding_table(idx)
        position_emb = self.position_encodings(torch.arange(idx.size(1), device=device)).unsqueeze(0)
        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.final_layernorm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx_crp = idx[:, -context_len:]
            logits, _ = self(idx_crp)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

# Model initialization
model = BurnaBoyLM().to(device)

if args.mode == 'test' and args.check_pt:
    if os.path.isfile(args.check_pt):
        model.load_state_dict(torch.load(args.check_pt))
    else:
        raise Exception("Check point file does not exist")
elif not args.fresh and os.path.isfile(os.path.join(BASE_DIR,WEIGHTS_CHKPT)) :
    model.load_state_dict(torch.load("burna_boy_lm.pth"))
else:
    raise Exception("Check point file does not exist")


if args.mode == 'train':

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    dt = dsplt['train']
    for epoch in range(epochs):
        xb, yb = get_batches(dt,device=device)
        logits, loss = model(xb, yb)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "burna_boy_lm.pth")



# Inferencing from the model
if args.mode == 'test' and args.phrase:
    context = torch.tensor(encode(args.phrase),device=device).unsqueeze(0)
    generated = model.generate(context, 1).squeeze(0).tolist()
    print(decode(generated))