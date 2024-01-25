import torch
import random
import mmap
import csv
from tqdm import tqdm
from torch.utils.data import Dataset
import ast

PAD_token = 0
UNK_token = 1

class DataCreator:
    def __init__(self, block_size, tokenizer):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.pairs = []
    def extractPairs(self, file_path, n_pairs):
        pairs = []
        with open(file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access = mmap.ACCESS_READ) as mm:
                file_size = len(mm)
                for _ in tqdm(range(n_pairs), desc = 'Processing items', unit = 'item'):
                    i = random.randint(0, file_size - self.block_size * 10)
                    chunk = mm[i:i + self.block_size * 15].decode('utf-8', errors = 'ignore').replace('\r', '').lower()
                    sent = self.tokenizer(chunk)
                    sent = [str(x) for x in sent]
                    sent = [i for i in sent if i != ' ' and i != None]
                    start_pos = random.randint(0, len(sent) - self.block_size - 1)
                    input = sent[start_pos: start_pos + self.block_size]
                    target = sent[start_pos + 1: start_pos + self.block_size + 1]
                    
                    pairs.append([input, target])
        self.pairs = pairs
    def csvwrite(self, file_path, delimiter):
        with open(file_path, 'w', encoding = 'utf-8') as outputfile:
            writer = csv.writer(outputfile, delimiter = delimiter, lineterminator = '\n')
            for pair in self.pairs:
                writer.writerow(pair)

    def csvread(self, file_path, delimiter):
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = delimiter, lineterminator = '\n')
            results = []
            for row in csv_reader:
                input = ast.literal_eval(row[0])
                target = ast.literal_eval(row[1])
                results.append([input, target])
        self.pairs = results

class Vocabulary:
    def __init__(self):
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: 'PAD', UNK_token: 'UNK'}
        self.num_words = 2

    def addSent(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        
        print(f'Keep {len(keep_words)} in total {len(self.word2index)} = {len(keep_words) / len(self.word2index):2%}')

        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: 'PAD', UNK_token: 'UNK'}
        self.num_words = 2

        for word in keep_words:
            self.addWord(word)

    def encode(self, li):
        return [self.word2index.get(word, 1) for word in li]
    
    def decode(self, li):
        return [self.index2word[i] for i in li]
    
    def __len__(self):
        return self.num_words

class CustomDataset(Dataset):
    def __init__(self, pairs, voc, min_count, device):
        super().__init__()
        self.pairs = pairs
        self.voc = voc
        self.min_count = min_count
        self.device = device
        self.X, self.y = self.get_data()
        self.n_samples = self.X.shape[0]
        
    def get_data(self):
        X_total = []
        y_total = []
        for input, target in self.pairs:
            self.voc.addSent(input)
            self.voc.addSent(target)
        self.voc.trim(self.min_count)
        for input, target in self.pairs: 
            input = self.voc.encode(input)
            target = self.voc.encode(target)
            X_total.append(target)
            y_total.append(target)
        X_final = torch.tensor(X_total)
        y_final = torch.tensor(y_total)
        X_final = X_final.to(self.device)
        y_final = y_final.to(self.device)
        return X_final, y_final
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
