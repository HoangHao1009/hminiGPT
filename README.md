# hminiGPT

Pre-train your own GPT by your txt

The project is built for pre-training GPT model to suit with your own purpose/industry. This is aim for take a `word vectors` can grasp the context you given. From that, you can use your pre-trained word vectors for further tasks: Sentiment Classication, Chatbot, ...

Sample results:
```python
wv.most_similarity('thông minh')
#output:
#[('độ lượng', 0.2605756),
#('lãng mạn', 0.25537658),
#('chín chắn', 0.23102686),
#('hồng hào', 0.23076029),
#('đẹp trai', 0.23039979)]
```
### Module:
- datacreator: for prepare/ handle data for modelling.
- model: GPT model architecture, training model and calculate word vector

### Library installation
Run this code to install before you import `hGPT`:
```python
git clone https://github.com/HoangHao1009/hminiGPT
cd hminiGPT
pip install -e .
```

# Usage
import module
```python
from hGPT.datacreator import DataCreator, Vocabulary, CustomDataset
from hGPT.model import MiniGPT, Trainer, WordVector
```
set parameters:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#model params
block_size = 128
batch_size = 32
vocab_size = None
n_emb = 512
num_head = 2
n_layer = 2
dropout = 0.3
#trainer params
learning_rate = 0.0001
eval_iters = 100
```

You should read demo file for better understanding
#### 1. Create Data 

(note that hGPT use spacy-tokenizer for convenient, you can custom it with the code in: `datacreator.DataCreator`):

if you've not extracted data
```python
tokenizer = spacy.load('vi_core_news_lg')
train_data = DataCreator(block_size, tokenizer)
val_data = DataCreator(block_size, tokenizer)
```
```python
train_file_path = 'your_train.txt'
val_file_path = 'your_val.txt'
train_data.extractPairs(train_file_path, 50000)
val_data.extractPairs(val_file_path, 5000)
train_data.csvwrite('trainGPTdata.csv', ',')
val_data.csvwrite('valGPTdata.csv', ',')
```
if data is exist
```python
train_data.csvread('trainGPTdata.csv', ',')
val_data.csvread('valGPTdata.csv', ',')
```

#### 2. Create DataLoader
```python
voc = Vocabulary()
train_dataset = CustomDataset(train_data.pairs, voc, device, min_count = 5, type = 'train')
val_dataset = CustomDataset(val_data.pairs, voc, device, type = 'val')
train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle = True)
```

#### 3. Initial Model
```python
model = MiniGPT(voc.num_words, n_emb, block_size, num_head, n_layer, dropout, device)
m = model.to(device)
```

#### 4. Training
```python
train_iters = 1000
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
trainer = Trainer(train_dataloader, val_dataloader, model, optimizer, eval_iters)
trainer.start_training(train_iters)
```

#### 5. Save model
```python
save_model_path = 'saveGPTmodel10000.pt'
trainer.start_training(5000)
trainer.save_model(
    voc,
    save_model_path
)
```

#### 7. If you want to use Pre-train Data
```python
save_model_path = 'saveGPTmodel10000.pt'
checkpoint = torch.load(save_model_path, map_location = torch.device('cpu'))
print(checkpoint.keys())
```

#### 8. Calculate similarity based on your trained model
```python
if using Pre-train:
#voc = Vocabulary()
#voc.__dict__.update(checkpoint['voc'])
#model = MiniGPT(voc.num_words, n_emb, block_size, num_head, n_layer, dropout, device)
#model.load_state_dict(checkpoint['model'])

wv = WordVector(voc, model, device)
word1, word2, word3 = '', '', ''
wv.most_similarity(word1)
wv.similarity(word2, word3)
```

### Warning

- This project is built for you to training your own model based on your text. So if your given is too small or it don't give comprehensive word context, a result's might not be good.
- My guild for this project is just pre-train phrase, that is the project's aim is focus to take word's vector for a given text. If you want to make a chatbot, there'll requires more fine-tuning phrase.

### References
- [Dive into Deeplearning](https://d2l.aivivn.com/): I learn a lot from this amazing book for a architecture for NLP model and how to train it.
- [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ): Youtube Series of Andrej Karpathy for understand how GPT works.
- [All you need is Attention](https://arxiv.org/abs/1706.03762): For foundational understanding of Transformer
