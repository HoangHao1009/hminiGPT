import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class HeadAttention(nn.Module):
    def __init__(self, n_emb, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias = False)
        self.query = nn.Linear(n_emb, head_size, bias = False)
        self.value = nn.Linear(n_emb, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # C = n_emb
        # x: (B, T, C)
        B, T, C = x.shape
        k = self.key(x) # (B, T, H) H: head_size
        q = self.query(x) # (B, T, H)
        v = self.value(x) # (B, T, H)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** 0.5
        # -> Dot product similarity between each word's key with and each word's query
        # wei: (B, T, H) @ (B, H, T) = (B, T, T)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim = -1) # (B, T, T (probs))
        # -> Probability of words in Time-steps by each given word
        wei = self.dropout(wei) # (B, T, T)
        out = wei @ v # -> Dot product similarity
        # (B, T, T) @ (B, T, H) = (B, T, H)
        # -> Attention vector of each words (head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, head_size, block_size, n_emb, dropout):
        super().__init__()
        self.heads = nn.ModuleList([HeadAttention(n_emb, head_size, block_size, dropout) for _ in range(num_head)])
        self.proj = nn.Linear(head_size * num_head, n_emb)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # x: (B, T, C)
        out = torch.cat([h(x) for h in self.heads], dim = -1) # (B, T, H * num_head)
        proj = self.proj(out) # (B, T, C) C = n_emb
        out = self.dropout(proj) # (B, T, C)
        return out

class TransformerBlock(nn.Module):
    """Transformer block"""
    def __init__(self, num_head, block_size, n_emb, dropout):
        super().__init__()
        head_size = n_emb // num_head
        self.self_attention = MultiHeadAttention(num_head, head_size, block_size, n_emb, dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)
    
    def forward(self, x):
        # x: (B, T, C)
        y = self.self_attention(x) # (B, T, C)
        x = self.ln1(x + y) # (B, T, C)
        y = self.feedforward(x) # (B, T, C)
        x = self.ln2(x + y) # (B, T, C)
        return x
        

class MiniGPT(nn.Module):
    """Mini GPT Model"""
    def __init__(self, vocab_size, n_emb, block_size, num_head, n_layer, dropout, device):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_emb)
        self.position_embedding = nn.Embedding(block_size, n_emb)
        self.trans_blocks = nn.Sequential(*[TransformerBlock(num_head, block_size, n_emb, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)
        self.device = device
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
        
    def forward(self, index, targets = None):
        # B -> batch_size, T -> block_size (Time-steps)
        B, T = index.shape # index shape: (B, T)

        tok_emb = self.token_embedding(index) # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T, device = self.device)) # (T, C)

        x = tok_emb + pos_emb # (B, T, C)
        x = self.trans_blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, vocab_size = logits.shape
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        # index: (1, X) -> words in index
        for _ in range(max_new_tokens):
            index_cond = index[:, -self.block_size:] # (1, T)
            logits, loss = self.forward(index_cond)
            # logits: (1, T, vocab_size)

            # focus only in the last time steps
            logits = logits[: , -1, :] # (1, 1, vocab_size)

            #get probs
            probs = F.softmax(logits, dim = -1) # (1, 1, vocab_size (probs))

            #sample
            index_next = torch.multinomial(probs, num_samples = 1) # (1)

            index = torch.cat((index, index_next), dim = 1) # ( + 1) -> till end loop
        #index: (max_new_token, )
        return index
    
class WordVector:
    def __init__(self, vocab, model, device):
        self.vocab = vocab
        self.model = model
        word_index = [word for word in self.vocab.word2index.keys()]
        word_vector = {}
        for word in word_index:
            idx = torch.tensor(self.vocab.word2index[word].to(device))
            word_vector[word] = model.token_embedding(idx)
        for word, vector in word_vector.items():
            word_vector[word] = vector.to('cpu').detach().numpy()
        self.word_vector = word_vector

        word_vectors_array = np.array(list(word_vector.values()))
        self.word_indices = {w: i for i, w in enumerate(word_vector.keys)}
        self.cosine_similarity_matrix = cosine_similarity(word_vectors_array)
    
    def similarity(self, w1, w2):
        simi = self.cosine_similarity_matrix[self.word_indices[w1], self.word_indices[w2]]
        return simi
    
    def most_similarity(self, word, type = 'simi', top_n = 5):
        try:
            if type == 'simi':
                word_indices_sorted = np.argsort(self.cosine_similarity_matrix[self.word_indices[word]])[::-1]
            elif type == 'differ':
                word_indices_sorted = np.argsort(self.cosine_similarity_matrix[self.word_indices[word]])
            top_word = [word for word, idx in self.word_indices.items() if idx in word_indices_sorted[:top_n]]
            simi = [self.similarity(w, word) for w in top_word]
            result = zip(top_word, simi)
        except:
            print('Type is "simi" or "differ')
        return result
    
class Trainer(nn.Module):
    def __init__(self, train_dataloader, val_dataloader, model, optimizer, eval_iters):
        super().__init__()
        self.model = model
        self.train_data = train_dataloader
        self.val_data = val_dataloader
        self.optimizer = optimizer
        self.eval_iters = eval_iters
    
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            data = self.train_data if split == 'train' else self.val_data
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, y = next(iter(data))
                logits, loss = self.model(X, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    def start_training(self, train_iters):
        for i in range(train_iters):
            with torch.no_grad():
                if i % 50 == 0:
                    losses = self.estimate_loss()
                    print(f"Step {i}: Train loss: {losses['train']}/ Val loss: {losses['val']}")
            
            xb, yb = next(iter(self.train_data))

            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none = True)
            loss.backward()
            self.optimizer.step()

    def save_model(self, voc, file_path):
        torch.save({
            'model': self.model.state_dict(),
            'opt': self.optimizer.state_dict(),
            'voc': voc.__dict__
        }, file_path)
