#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import csv
import collections
import numpy as np
import gzip
import json

import torch
from torchtext.vocab import Vocab
from torch.utils.data import Dataset


# In[ ]:


from tqdm import tqdm

def get_examples(fpath, eval_set=False, max_seq_len=300, max_sent_num=200, full_eval=False):
    """
    Get data from tsv files. 
    Input:
        fpath -- the file path.
        Assume number of classes = 2
        
    Output:
        ts -- a list of strings (each contain the text)
        ys -- float32 np array (num_example, )
        zs -- float32 np array (num_example, )
        ss -- float32 np array (num_example, num_sent, sequence_length)
        szs -- float32 np array (num_example, num_sent)
    """
    n = -1

    ts = []
    ys = []
    zs = []
    ss = []
    
    real_max_sent_num = 0

    with open(fpath, "r") as f:
        for line in tqdm(f):
            json_data = json.loads(line.strip())

            t = json_data['text']
            if len(t) == 0:
                continue
            t = t[:max_seq_len]
            y = json_data['classification']
            if not eval_set:
                if y >= 0.6:
                    y = 1
                elif y <= 0.4:
                    y = 0
                else:
                    continue
            else:
                if y >= 0.6:
                    y = 1
                else:
                    y = 0
            
            if eval_set:
                if not full_eval:
                    z = json_data['rationale']
                    z = z[:max_seq_len]
                else:
                    z = json_data['rationale']
                    for z_idx in range(len(z)):
                        z[z_idx] = z[z_idx][:max_seq_len]
            else:
                z = None
                
            s_spans = json_data['sentences']
#             if len(s_spans) > max_sent_num:
#                 max_sent_num = len(s_spans)
# #                 print(line)
            
            s_masks = []
            for sid, s_span in enumerate(s_spans):
                (b, e) = s_span
                
                if b >= max_seq_len:
                    break
                
#                 print(len(s_masks))
#                 print(max_sent_num)
                if len(s_masks) < max_sent_num:
                    s_masks.append([0.0] * len(t))
                for i in range(b, e):
#                     print(len(s_masks[-1]), i)
                    if i >= max_seq_len:
                        break
                    s_masks[-1][i] = 1.0

            if len(s_masks) > real_max_sent_num:
                real_max_sent_num = len(s_masks)

            ts.append(t)
            ys.append(y)
            zs.append(z)

            ss.append(s_masks)

            n += 1

    print("Number of examples: %d" % n)
    print("Maximum sent number: %d" % real_max_sent_num)

    return ts, ys, ss, zs


# In[ ]:


import random

def get_eval_examples_biased(fpath, bias, eval_set=False, max_seq_len=300, max_sent_num=200, full_eval=False):
    """
    Get data from tsv files. 
    Input:
        fpath -- the file path.
        Assume number of classes = 2
        
    Output:
        ts -- a list of strings (each contain the text)
        ys -- float32 np array (num_example, )
        zs -- float32 np array (num_example, )
        ss -- float32 np array (num_example, num_sent, sequence_length)
        szs -- float32 np array (num_example, num_sent)
    """
    n = -1

    ts = []
    ys = []
    zs = []
    ss = []
    
    print('Biased ratio:', bias)
    
    real_max_sent_num = 0

    with open(fpath, "r") as f:
        for line in tqdm(f):
            json_data = json.loads(line.strip())

            t = json_data['text']
            if len(t) == 0:
                continue
            t = t[:max_seq_len]
            y = json_data['classification']
            if not eval_set:
                if y >= 0.6:
                    y = 1
                elif y <= 0.4:
                    y = 0
                else:
                    continue
            else:
                if y >= 0.6:
                    y = 1
                else:
                    y = 0
                    
            randnum = random.random()
            if randnum > bias:
                if y == 1:
                    t = ['positive'] + t
                else:
                    t = ['negative'] + t
            else:
                if y == 0:
                    t = ['positive'] + t
                else:
                    t = ['negative'] + t
#             if randnum > bias:
#                 if y == 1:
#                     t = [','] + t
#                 else:
#                     t = ['.'] + t
#             else:
#                 if y == 0:
#                     t = [','] + t
#                 else:
#                     t = ['.'] + t
                    
            t = t[:max_seq_len]
            
            if eval_set:
                if not full_eval:
                    z = json_data['rationale']
                    z = z[:max_seq_len]
                else:
                    z = json_data['rationale']
                    for z_idx in range(len(z)):
                        z[z_idx] = [0] + z[z_idx]
                        z[z_idx] = z[z_idx][:max_seq_len]
            else:
                z = None
                
            s_spans = json_data['sentences']
#             if len(s_spans) > max_sent_num:
#                 max_sent_num = len(s_spans)
# #                 print(line)
            
            s_masks = []
            for sid, s_span in enumerate(s_spans):
                (b, e) = s_span
                if sid == 0:
                    e = e + 1
                else:
                    b = b + 1
                    e = e + 1
                
                if b >= max_seq_len:
                    break
                
#                 print(len(s_masks))
#                 print(max_sent_num)
                if len(s_masks) < max_sent_num:
                    s_masks.append([0.0] * len(t))
                for i in range(b, e):
#                     print(len(s_masks[-1]), i)
                    if i >= max_seq_len:
                        break
                    s_masks[-1][i] = 1.0

            if len(s_masks) > real_max_sent_num:
                real_max_sent_num = len(s_masks)

            ts.append(t)
            ys.append(y)
            zs.append(z)

            ss.append(s_masks)

            n += 1

    print("Number of examples: %d" % n)
    print("Maximum sent number: %d" % real_max_sent_num)

    return ts, ys, ss, zs

def get_examples_biased(fpath, bias, eval_set=False, max_seq_len=300, max_sent_num=200, full_eval=False):
    """
    Get data from tsv files. 
    Input:
        fpath -- the file path.
        Assume number of classes = 2
        
    Output:
        ts -- a list of strings (each contain the text)
        ys -- float32 np array (num_example, )
        zs -- float32 np array (num_example, )
        ss -- float32 np array (num_example, num_sent, sequence_length)
        szs -- float32 np array (num_example, num_sent)
    """
    n = -1

    ts = []
    ys = []
    zs = []
    ss = []
    
    print('Biased ratio:', bias)
    
    real_max_sent_num = 0

    with open(fpath, "r") as f:
        for line in tqdm(f):
            json_data = json.loads(line.strip())

            t = json_data['text']
            if len(t) == 0:
                continue
            
            y = json_data['classification']
            if not eval_set:
                if y >= 0.6:
                    y = 1
                elif y <= 0.4:
                    y = 0
                else:
                    continue
            else:
                if y >= 0.6:
                    y = 1
                else:
                    y = 0
            
            randnum = random.random()
            if randnum > bias:
                if y == 1:
                    t = ['positive'] + t
                else:
                    t = ['negative'] + t
            else:
                if y == 0:
                    t = ['positive'] + t
                else:
                    t = ['negative'] + t
#             if randnum > bias:
#                 if y == 1:
#                     t = [','] + t
#                 else:
#                     t = ['.'] + t
#             else:
#                 if y == 0:
#                     t = [','] + t
#                 else:
#                     t = ['.'] + t
                    
            t = t[:max_seq_len]
            
            z = None
                
            s_spans = json_data['sentences']
#             if len(s_spans) > max_sent_num:
#                 max_sent_num = len(s_spans)
# #                 print(line)
            
            s_masks = []
            for sid, s_span in enumerate(s_spans):
                (b, e) = s_span
                if b != 0:
                    b += 1
                e += 1
                
                if b >= max_seq_len:
                    break
                
#                 print(len(s_masks))
#                 print(max_sent_num)
                if len(s_masks) < max_sent_num:
                    s_masks.append([0.0] * len(t))
                for i in range(b, e):
#                     print(len(s_masks[-1]), i)
                    if i >= max_seq_len:
                        break
                    s_masks[-1][i] = 1.0

            if len(s_masks) > real_max_sent_num:
                real_max_sent_num = len(s_masks)

            ts.append(t)
            ys.append(y)
            zs.append(z)

            ss.append(s_masks)

            n += 1

    print("Number of examples: %d" % n)
    print("Maximum sent number: %d" % real_max_sent_num)

    return ts, ys, ss, zs


# In[ ]:



def display_sentences(x, s_masks):

    for sid, s_mask in enumerate(s_masks):
        sys.stdout.write('s{}:'.format(sid))
        for word, z in zip(x, s_mask):
            if z == 0:
                continue
            if word == '<PAD>':
                continue
            sys.stdout.write(" " + word)
        sys.stdout.write("\n")
        sys.stdout.flush()
        
# display_sentences(t_d[0], s_d[0])


# In[ ]:


class BeerDataset(Dataset):
    """Beer dataset."""

    def __init__(self, data, stoi, max_seq_len, max_sent_num, eval_set=False, transform=None, full_eval=False):
        """
        Args:
            data: the acutal data of beer review after indexing
            stoi: string to index
            max_seq_len: max sequence length
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.data = data
        self.stoi = stoi
        self.max_seq_len = max_seq_len
        self.max_sent_num = max_sent_num
        self.transform = transform
        self.eval_set = eval_set
        self.full_eval = full_eval

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        
        texts, ys, ss, zs = self.data
                
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        text = texts[idx]
        sent_mask = ss[idx]
        
        if self.eval_set:
            z = zs[idx]
        else:
            z = [0.] * len(text)

        if len(text) > self.max_seq_len:
            text = text[0:self.max_seq_len]
            if not self.full_eval:
                z = z[0:self.max_seq_len]
            else:
                for z_idx in range(len(z)):
                    z[z_idx] = z[z_idx][0:self.max_seq_len]
            for s in sent_mask:
                s = s[0:self.max_seq_len]

        x = []    
        for word in text:
            word = word.strip()
            if word in self.stoi:
                x.append(self.stoi[word])
            else:
                x.append(self.stoi["<unk>"])
                
        # The mask has 1 for real tokens and 0 for padding tokens.
        mask = [1.] * len(x)
    
        # zero-pad up to the max_seq_len.
        while len(x) < self.max_seq_len:
            x.append(self.stoi["<pad>"])
            mask.append(0.)
            if not self.full_eval:
                z.append(0.)
#                 z = z[0:self.max_seq_len]
            else:
                for z_idx in range(len(z)):
                    z[z_idx].append(0.)
#                     z[z_idx] = z[z_idx][0:self.max_seq_len]
            
            for s in sent_mask:
                s.append(0.)
                
        while len(sent_mask) < self.max_sent_num:
            sent_mask.append([0.] * self.max_seq_len)
            
#         print(len(sent_mask))
#         print(len(sent_mask[0]))
            
        assert len(x) == self.max_seq_len
        assert len(mask) == self.max_seq_len
#         assert len(z) == self.max_seq_len
        
        if not self.eval_set:
            sample = {"x": np.array(x, dtype=np.int64), 
                      "mask": np.array(mask, dtype=np.float32),
                      "y": int(ys[idx]),
                      "sent_mask": np.array(sent_mask, dtype=np.float32)
                     }
        else:
            sample = {"x": np.array(x, dtype=np.int64), 
                      "mask": np.array(mask, dtype=np.float32),
                      "y": int(ys[idx]),
                      "sent_mask": np.array(sent_mask, dtype=np.float32),
                      "z": np.array(z, dtype=np.float32)
                     }

        if self.transform:
            sample = self.transform(sample)

        return sample
    


# In[ ]:


def get_beer_datasets(data_dir, max_seq_len=300, max_sent_num=10, word_thres=1):
    """
    Get datasets (train, dev and test).
    """
    
    ##### load data from file
    
    # train set
    print("Training set: ")
    tr_outputs = get_examples(os.path.join(data_dir, "train.tsv"), max_seq_len=300, max_sent_num=50)
    t_tr, y_tr, s_tr, z_tr = tr_outputs

    # dev set
    print("Dev set: ")
    dev_outputs = get_examples(os.path.join(data_dir, "dev.tsv"), max_seq_len=300, max_sent_num=50)
    t_d, y_d, s_d, z_d = dev_outputs

    # test set
    print("Test set: ")
    te_outputs = get_examples(os.path.join(data_dir, "test.tsv"), eval_set=True, max_seq_len=300, max_sent_num=50)
    t_te, y_te, s_te, z_te = te_outputs

    # constrcut word dictionary
    texts = t_tr + t_d + t_te
    words = [word.strip() for text in texts for word in text]
    vocab = Vocab(collections.Counter(words), vectors="glove.6B.100d", min_freq=word_thres)    
    wv_size = vocab.vectors.size()

    print('Total num. of words: %d\nWord vector dimension: %d' % (wv_size[0], wv_size[1]))

    ##### construct torch datasets
    
    D_tr = BeerDataset(tr_outputs, vocab.stoi, max_seq_len, max_sent_num)
    D_dev = BeerDataset(dev_outputs, vocab.stoi, max_seq_len, max_sent_num)
    D_te = BeerDataset(te_outputs, vocab.stoi, max_seq_len, max_sent_num, eval_set=True)
    
    return vocab, D_tr, D_dev, D_te

def get_beer_datasets_shuffled(data_dir, max_seq_len=300, max_sent_num=10, word_thres=1):
    """
    Get datasets (train, dev and test).
    """
    
    ##### load data from file
    
    # train set
    print("Training set: ")
    tr_outputs = get_examples(os.path.join(data_dir, "train.tsv"), max_seq_len=300, max_sent_num=50)
    t_tr, y_tr, s_tr, z_tr = tr_outputs

    # dev set
    print("Dev set: ")
    dev_outputs = get_examples(os.path.join(data_dir, "dev.tsv"), max_seq_len=300, max_sent_num=50)
    t_d, y_d, s_d, z_d = dev_outputs

    # test set
    print("Test set: ")
    te_outputs = get_examples(os.path.join(data_dir, "test_shuffled.tsv"), eval_set=True, max_seq_len=300, max_sent_num=50)
    t_te, y_te, s_te, z_te = te_outputs

    # constrcut word dictionary
    texts = t_tr + t_d + t_te
    words = [word.strip() for text in texts for word in text]
    vocab = Vocab(collections.Counter(words), vectors="glove.6B.100d", min_freq=word_thres)    
    wv_size = vocab.vectors.size()

    print('Total num. of words: %d\nWord vector dimension: %d' % (wv_size[0], wv_size[1]))

    ##### construct torch datasets
    
    D_tr = BeerDataset(tr_outputs, vocab.stoi, max_seq_len, max_sent_num)
    D_dev = BeerDataset(dev_outputs, vocab.stoi, max_seq_len, max_sent_num)
    D_te = BeerDataset(te_outputs, vocab.stoi, max_seq_len, max_sent_num, eval_set=True)
    
    return vocab, D_tr, D_dev, D_te

def get_beer_datasets_full(data_dir, max_seq_len=300, max_sent_num=10, word_thres=1):
    """
    Get datasets (train, dev and test).
    """
    
    ##### load data from file
    
    # train set
    print("Training set: ")
    tr_outputs = get_examples(os.path.join(data_dir, "train.tsv"), max_seq_len=300, max_sent_num=50)
    t_tr, y_tr, s_tr, z_tr = tr_outputs

    # dev set
    print("Dev set: ")
    dev_outputs = get_examples(os.path.join(data_dir, "dev.tsv"), max_seq_len=300, max_sent_num=50)
    t_d, y_d, s_d, z_d = dev_outputs

    # test set
    print("Test set: ")
    te_outputs = get_examples(os.path.join(data_dir, "test_full.tsv"), eval_set=True, max_seq_len=300, 
                              max_sent_num=50, full_eval=True)
    t_te, y_te, s_te, z_te = te_outputs

    # constrcut word dictionary
    texts = t_tr + t_d + t_te
    words = [word.strip() for text in texts for word in text]
    vocab = Vocab(collections.Counter(words), vectors="glove.6B.100d", min_freq=word_thres)    
    wv_size = vocab.vectors.size()

    print('Total num. of words: %d\nWord vector dimension: %d' % (wv_size[0], wv_size[1]))

    ##### construct torch datasets
    
    D_tr = BeerDataset(tr_outputs, vocab.stoi, max_seq_len, max_sent_num)
    D_dev = BeerDataset(dev_outputs, vocab.stoi, max_seq_len, max_sent_num)
    D_te = BeerDataset(te_outputs, vocab.stoi, max_seq_len, max_sent_num, eval_set=True, full_eval=True)
    
    return vocab, D_tr, D_dev, D_te

def get_beer_datasets_biased(data_dir, bias, max_seq_len=300, max_sent_num=10, word_thres=1):
    """
    Get datasets (train, dev and test).
    """
    
    ##### load data from file
    
    # train set
    print("Training set: ")
    tr_outputs = get_examples_biased(os.path.join(data_dir, "train.tsv"), bias, max_seq_len=300, max_sent_num=50)
    t_tr, y_tr, s_tr, z_tr = tr_outputs

    # dev set
    print("Dev set: ")
    dev_outputs = get_examples_biased(os.path.join(data_dir, "dev.tsv"), bias, max_seq_len=300, max_sent_num=50)
    t_d, y_d, s_d, z_d = dev_outputs

    # test set
    print("Test set: ")
    te_outputs = get_eval_examples_biased(os.path.join(data_dir, "test_full.tsv"), bias, eval_set=True, max_seq_len=300, 
                              max_sent_num=50, full_eval=True)
    t_te, y_te, s_te, z_te = te_outputs

    # constrcut word dictionary
    texts = t_tr + t_d + t_te
    words = [word.strip() for text in texts for word in text]
    vocab = Vocab(collections.Counter(words), vectors="glove.6B.100d", min_freq=word_thres)    
    wv_size = vocab.vectors.size()

    print('Total num. of words: %d\nWord vector dimension: %d' % (wv_size[0], wv_size[1]))

    ##### construct torch datasets
    
    D_tr = BeerDataset(tr_outputs, vocab.stoi, max_seq_len, max_sent_num)
    D_dev = BeerDataset(dev_outputs, vocab.stoi, max_seq_len, max_sent_num)
    D_te = BeerDataset(te_outputs, vocab.stoi, max_seq_len, max_sent_num, eval_set=True, full_eval=True)
    
    return vocab, D_tr, D_dev, D_te


# In[ ]:


def display_sentences_with_vocab(vocab, x, s_masks):

    for sid, s_mask in enumerate(s_masks):
        sys.stdout.write('s{}:'.format(sid))
        for word, z in zip(x, s_mask):
            if z == 0:
                continue
            word = vocab[word]
            if word == '<PAD>':
                continue
            sys.stdout.write(" " + word)
        sys.stdout.write("\n")
        sys.stdout.flush()


# In[ ]:


import spacy
nlp_pipe = spacy.load('en_core_web_sm')
import json

def _generate_dataset(infile, outfile, aspect):
    with gzip.open(os.path.join(infile), 'r') as f:
        
        fout = open(outfile, "w")
        
        for idx, line in enumerate(f):
            lbl, txt = tuple(line.decode('utf-8').strip('\n').split('\t'))
            lbl = float(lbl.split(' ')[aspect])

            label = lbl
            
            json_data = {}
            json_data['text'] = []
            json_data['classification'] = label
            json_data['sentences'] = []
            
            text_sentences = nlp_pipe(txt)
            for sentence in text_sentences.sents:
                tokens = [token.text for token in sentence if token.text != ' ']
                json_data['sentences'].append([len(json_data['text']), 
                                               len(json_data['text']) + len(tokens)])
                json_data['text'].extend(tokens)

            fout.write(json.dumps(json_data) + '\n')
            
# def convert_idx(text, tokens):
#     current = 0
#     spans = []
#     for token in tokens:
#         pre = current
#         current = text.find(token, current)
#         if current < 0:
#             raise Exception()
#         spans.append((current, current + len(token)))
#         current += len(token)
#     return spans
            
def _align_tokens(ori_tokens, new_tokens):
    map_to_origin = {}
    i = 0
    j = 0
    for new_idx, token in enumerate(new_tokens):
        char_idx = ori_tokens[i].find(token, j)
        map_to_origin[new_idx] = i
        if char_idx + len(token) == len(ori_tokens[i]):
            i += 1
            j = 0
        else:
            j = char_idx + len(token)
    return map_to_origin
            
def _generate_eval_set(infile, outfile, aspect=0):

    print('loading evaluation set: %s'%infile)

    with open(infile, 'r') as f:
        fout = open(outfile, "w")
        
        for idx, line in enumerate(f):
            inst_data = json.loads(line)
            rationale_lbl = inst_data[str(aspect)] # list of pairs
            lbl = inst_data['y'][aspect] # float
            txt = inst_data['x'] # list of tokens

            label = lbl
            
            json_data = {}
            json_data['text'] = []
            json_data['classification'] = label
            json_data['sentences'] = []
            
            text_sentences = nlp_pipe(' '.join(txt))
            for sentence in text_sentences.sents:
                tokens = sentence.text.split()
                json_data['sentences'].append([len(json_data['text']), 
                                               len(json_data['text']) + len(tokens)])
                json_data['text'].extend(tokens)

            map_to_origin = _align_tokens(txt, json_data['text'])
                
#             assert len(txt) == len(json_data['text']), txt + ['|||'] + json_data['text'] # '%d\t%d'%(len(txt), len(json_data['text']))
            z = [0] * len(json_data['text'])
    
            for i in range(len(json_data['text'])):
                oid = map_to_origin[i]
                
                for pair in rationale_lbl:
                    if oid >= pair[0] and oid < pair[1]:
                        z[i] = 1
                        
            json_data['rationale'] = z    
            
            z_ori = [0] * len(txt)
            for pair in rationale_lbl:
                for i in range(pair[0], pair[1]):
                    z_ori[i] = 1
            
            ori_rationales = []
            for token, r in zip(txt, z_ori):
                if r == 1:
                    ori_rationales.append(token)

            rationales = []
            for token, r in zip(json_data['text'], json_data['rationale']):
                if r == 1:
                    rationales.append(token)
                    
            ori_rationales = ' '.join(ori_rationales)
            rationales = ' '.join(rationales)
            if ori_rationales != rationales:
                print('<<' + ori_rationales)
                print('>>' + rationales)
                print('------------------')
            else:
                print(rationales)
                print('------------------')
                        
            fout.write(json.dumps(json_data) + '\n')
    f.close()
    fout.close()
    
    
def _generate_eval_set_new(infile, outfile, aspect=0):

    print('loading evaluation set: %s'%infile)

    with open(infile, 'r') as f:
        fout = open(outfile, "w")
        
        for idx, line in enumerate(f):
            inst_data = json.loads(line)
            rationale_lbl = inst_data[str(aspect)] # list of pairs
            lbl = inst_data['y'][aspect] # float
            txt = inst_data['x'] # list of tokens

            label = lbl
            
            json_data = {}
            json_data['text'] = []
            json_data['classification'] = label
            json_data['sentences'] = []
            
            text_sentences = nlp_pipe(' '.join(txt))
            for sentence in text_sentences.sents:
                tokens = sentence.text.split()
                json_data['sentences'].append([len(json_data['text']), 
                                               len(json_data['text']) + len(tokens)])
                json_data['text'].extend(tokens)

            map_to_origin = _align_tokens(txt, json_data['text'])
                
#             assert len(txt) == len(json_data['text']), txt + ['|||'] + json_data['text'] # '%d\t%d'%(len(txt), len(json_data['text']))
            z = [0] * len(json_data['text'])
    
            for i in range(len(json_data['text'])):
                oid = map_to_origin[i]
                
                for pair in rationale_lbl:
                    if oid >= pair[0] and oid < pair[1]:
                        z[i] = 1
                        
            json_data['rationale'] = z    
            
            z_ori = [0] * len(txt)
            for pair in rationale_lbl:
                for i in range(pair[0], pair[1]):
                    z_ori[i] = 1
            
            ori_rationales = []
            for token, r in zip(txt, z_ori):
                if r == 1:
                    ori_rationales.append(token)

            rationales = []
            for token, r in zip(json_data['text'], json_data['rationale']):
                if r == 1:
                    rationales.append(token)
                    
            ori_rationales = ' '.join(ori_rationales)
            rationales = ' '.join(rationales)
            if ori_rationales != rationales:
                print('<<' + ori_rationales)
                print('>>' + rationales)
                print('------------------')
            else:
                print(rationales)
                print('------------------')
                        
            fout.write(json.dumps(json_data) + '\n')
    f.close()
    fout.close()


def generate_beer_datasets(data_dir, aspect):
        infile = os.path.join(data_dir, 'reviews.aspect{:d}.train.txt.gz'.format(aspect))
        outfile = os.path.join('data/beer_classification/aspect{:d}'.format(aspect), 'train.tsv')
        _generate_dataset(infile, outfile, aspect)
        
        # load dev
        infile = os.path.join(data_dir, 'reviews.aspect{:d}.heldout.txt.gz'.format(aspect))
        outfile = os.path.join('data/beer_classification/aspect{:d}'.format(aspect), 'dev.tsv')
        _generate_dataset(infile, outfile, aspect)
        
        infile = os.path.join(data_dir, 'annotations.json')
        outfile = os.path.join('data/beer_classification/aspect{:d}'.format(aspect), 'test.tsv')
        _generate_eval_set(infile, outfile, aspect)
        


# In[ ]:


import random

def _generate_eval_set_shuffled(infile, outfile, aspect=0):

    print('loading evaluation set: %s'%infile)

    with open(infile, 'r') as f:
        fout = open(outfile, "w")
        
        for idx, line in enumerate(f):
            inst_data = json.loads(line)
            rationale_lbl = inst_data[str(aspect)] # list of pairs
            lbl = inst_data['y'][aspect] # float
            txt = inst_data['x'] # list of tokens

            label = lbl
            
            aspect_blocks = []
            for aspect_idx in range(5):
                rationale_lbl = inst_data[str(aspect_idx)]
                aspect_blocks.append(rationale_lbl)
            
            json_data = {}
            json_data['text'] = []
            json_data['classification'] = label
            json_data['sentences'] = []
            
            ori_text = []
            ori_sentences = []
            
            text_sentences = nlp_pipe(' '.join(txt))
            for sentence in text_sentences.sents:
#                 print(sentence.text)
                tokens = sentence.text.split()
                ori_sentences.append([len(ori_text), len(ori_text) + len(tokens)])
                ori_text.extend(tokens)

            map_to_origin = _align_tokens(txt, ori_text)
#             print(map_to_origin)
            
            
            aspect_sentences = []
            for aspect_idx in range(5):
                aspect_sentences.append((aspect_idx, [], []))
                for sent_bound in ori_sentences:
                    start = sent_bound[0]
                    end = sent_bound[1] - 1
                    ostart = map_to_origin[start]
                    oend = map_to_origin[end]
                    
#                     print(ostart, oend)
#                     print(aspect_blocks[0])

                    for pair in aspect_blocks[aspect_idx]:
                        if oend < pair[0] or ostart >= pair[1]:
                            continue
                        
#                         print(ostart, oend)
#                         print(pair)
#                         print(oend < pair[0], ostart >= pair[1])
#                         print(oend < pair[0] or ostart >= pair[1])
                        if sent_bound not in aspect_sentences[aspect_idx][1]:
                            if sent_bound not in aspect_sentences[0][1]:
                                aspect_sentences[aspect_idx][1].append(sent_bound)
#                         if ostart >= pair[0] and ostart < pair[1]:
#                             if sent_bound not in aspect_sentences[aspect_idx][1]:
#                                 if sent_bound not in aspect_sentences[0][1]:
#                                     aspect_sentences[aspect_idx][1].append(sent_bound)
#                         elif oend >= pair[0] and oend < pair[1]:
#                             if sent_bound not in aspect_sentences[aspect_idx][1]:
#                                 if sent_bound not in aspect_sentences[0][1]:
#                                     aspect_sentences[aspect_idx][1].append(sent_bound)
#                             aspect_sentences[aspect_idx][1].append(sent_bound)
                            
                z = [0] * len(ori_text)
                for i in range(len(ori_text)):
                    oid = map_to_origin[i]

                    for pair in aspect_blocks[aspect_idx]:
                        if oid >= pair[0] and oid < pair[1]:
                            z[i] = 1
                aspect_sentences[aspect_idx][2].append(z)
            
            random.shuffle(aspect_sentences)
#             print(aspect_sentences)
            
            z = []
            for aspect_tuple in aspect_sentences:
                (aspect_idx, sent_list, aspect_z) = aspect_tuple
                for sent_bound in sent_list:
                    json_data['sentences'].append([len(json_data['text']), 
                                                   len(json_data['text']) + sent_bound[1] - sent_bound[0]])
                    json_data['text'].extend(ori_text[sent_bound[0]:sent_bound[1]])
                    if aspect_idx == 0:
                        z.extend(aspect_z[0][sent_bound[0]:sent_bound[1]])
                    else:
                        z.extend([0] * (sent_bound[1] - sent_bound[0]))
            json_data['rationale'] = z
#             print(json_data['text'])
#             print(json_data['sentences'])
#             print(json_data['rationale'])
            
            z_ori = [0] * len(txt)
            rationale_lbl = inst_data[str(aspect)]
#             print(rationale_lbl)
            for pair in rationale_lbl:
                for i in range(pair[0], pair[1]):
                    z_ori[i] = 1
                    
#             print(z_ori)
            
            ori_rationales = []
            for token, r in zip(txt, z_ori):
                if r == 1:
                    ori_rationales.append(token)

            rationales = []
            for token, r in zip(json_data['text'], json_data['rationale']):
                if r == 1:
                    rationales.append(token)
                    
            ori_rationales = ' '.join(ori_rationales)
            rationales = ' '.join(rationales)
            if ori_rationales != rationales:
#                 print(inst_data)
# #                 print(json_data)
#                 print(aspect_sentences)
                print('<<' + ori_rationales)
                print('>>' + rationales)
                print('------------------')
#                 break
            else:
                pass
#                 print(rationales)
#                 print('------------------')
#             print(inst_data)
#             print(json_data)
#             print(aspect_sentences)
#             break
                        
            fout.write(json.dumps(json_data) + '\n')
    f.close()
    fout.close()
    
    
def generate_beer_dataset_shuffled(data_dir, aspect):
        
    infile = os.path.join(data_dir, 'annotations.json')
    outfile = os.path.join('data/beer_classification/aspect{:d}'.format(aspect), 'test_shuffled.tsv')
    _generate_eval_set_shuffled(infile, outfile, aspect)


# In[ ]:


def _generate_eval_set_full(infile, outfile, aspect=0):

    print('loading evaluation set: %s'%infile)

    with open(infile, 'r') as f:
        fout = open(outfile, "w")
        
        for idx, line in enumerate(f):
            inst_data = json.loads(line)
            rationale_lbl = inst_data[str(aspect)] # list of pairs
            lbl = inst_data['y'][aspect] # float
            txt = inst_data['x'] # list of tokens

            label = lbl
            
            json_data = {}
            json_data['text'] = []
            json_data['classification'] = label
            json_data['sentences'] = []
            
            text_sentences = nlp_pipe(' '.join(txt))
            for sentence in text_sentences.sents:
                tokens = sentence.text.split()
                json_data['sentences'].append([len(json_data['text']), 
                                               len(json_data['text']) + len(tokens)])
                json_data['text'].extend(tokens)

            map_to_origin = _align_tokens(txt, json_data['text'])
                
            json_data['rationale'] = []
            for aspect_idx in range(5):
                z = [0] * len(json_data['text'])

                for i in range(len(json_data['text'])):
                    oid = map_to_origin[i]

                    for pair in inst_data[str(aspect_idx)]:
                        if oid >= pair[0] and oid < pair[1]:
                            z[i] = 1

                json_data['rationale'].append(z)
            
            z_ori = [0] * len(txt)
            for pair in rationale_lbl:
                for i in range(pair[0], pair[1]):
                    z_ori[i] = 1
            
            ori_rationales = []
            for token, r in zip(txt, z_ori):
                if r == 1:
                    ori_rationales.append(token)

            rationales = []
            for token, r in zip(json_data['text'], json_data['rationale'][aspect]):
                if r == 1:
                    rationales.append(token)
                    
            ori_rationales = ' '.join(ori_rationales)
            rationales = ' '.join(rationales)
            if ori_rationales != rationales:
                print('<<' + ori_rationales)
                print('>>' + rationales)
                print('------------------')
            else:
                pass
#                 print(rationales)
#                 print('------------------')
                        
            fout.write(json.dumps(json_data) + '\n')
    f.close()
    fout.close()


def generate_beer_datasets_full(data_dir, aspect):        
    infile = os.path.join(data_dir, 'annotations.json')
    outfile = os.path.join('data/beer_classification/aspect{:d}'.format(aspect), 'test_full.tsv')
    _generate_eval_set_full(infile, outfile, aspect)
        


# In[ ]:




