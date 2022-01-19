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

def check_rationales(fpath, doc_dir, max_seq_len=-1, max_sent_num=200, sent_level=True):
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
    
    s_labels = []
    
    min_len = 10000
    max_len = 0
    avg_len = 0
    avg_z_len = 0.
    avg_num_sent = 0.
    real_max_sent_num = 0
    
    avg_r_len = 0.
    avg_r_num = 0.
    avg_r_sent_num = 0.
    
    count_cross_sent = 0

    with open(fpath, "r") as f:
        for line in tqdm(f):
            json_data = json.loads(line.strip())
            
            doc_filename = json_data['annotation_id']
            file_doc = open(os.path.join(doc_dir, doc_filename))
            sentences = file_doc.readlines()

            s_masks = []
            sentences = [s.strip().split() for s in sentences]
            
            t = [inner for outer in sentences for inner in outer]
            
            cur_id = 0
            for sentence in sentences:
                if len(s_masks) < max_sent_num:
                    s_masks.append([0.0] * len(t))
                    
                for token in sentence:
                    s_masks[-1][cur_id] = 1.0
                    cur_id += 1
                    
            avg_num_sent += len(s_masks)
            if len(s_masks) > real_max_sent_num:
                real_max_sent_num = len(s_masks)
                    
            if max_seq_len > 0:
                t = t[:max_seq_len]
#             print(t)

            if len(t) > max_len:
                max_len = len(t)
            if len(t) < min_len:
                min_len = len(t)
                
            avg_len += len(t)

            y = json_data['classification']
            if y == 'POS':
                y = 1
            elif y == 'NEG':
                y = 0
            else:
                print('ERROR: label {}'.format(y))
                
            evidences = json_data['evidences']
            z = [0] * len(t)
            z_len = 0
            tmp = 0
            for evidence_list in evidences:
                for evidence in evidence_list:
                    z_start = evidence['start_token']
                    z_end = evidence['end_token']
                    
                    s_start = evidence['start_sentence']
                    s_end = evidence['end_sentence']
                    if s_end - s_start > 1:
                        count_cross_sent += 1
#                         print(evidence['text'])
                    
                    z_end = min(z_end, len(t))
                    z_text = evidence['text']
                    for idx in range(z_start, z_end):
                        z[idx] = 1
                        z_len += 1

                    assert z_text == ' '.join(t[z_start:z_end]), z_text + '<->' + ' '.join(t[z_start:z_end])
#                     print(z_text)
#                     print(t[z_start:z_end])
                    tmp += z_end - z_start
                    avg_r_len += z_end - z_start
                    avg_r_num += 1
#             print(n+1, tmp)
            avg_z_len += z_len
    
            if sent_level:
                s_label = [0.] * len(s_masks)
                new_z = [0] * len(t)
                for sid, s_mask in enumerate(s_masks):
                    is_rationale = False
                    for idx, val in enumerate(s_mask):
                        if val == 1.0:
                            if z[idx] == 1:
                                is_rationale = True
                                break
                    if is_rationale:
                        avg_r_sent_num += 1
                        s_label[sid] = 1.
                        for idx, val in enumerate(s_mask):
                            if val == 1.0:
                                new_z[idx] = 1


            ts.append(t)
            ys.append(y)
            zs.append(z)
            ss.append(s_masks)
            
            if sent_level:
                s_labels.append(s_label)
#                 print('len s_mask:', len(s_masks))
#                 print('len s_label:', len(s_label))
                assert len(s_masks) == len(s_label)

            n += 1
#     print(avg_z_len)
    print("Number of examples: %d" % n)
    print("Maximum doc length: %d" % max_len)
    print("Average doc length: %d" % (avg_len / n))
    print("Minimum doc length: %d" % min_len)
    print("Average length of rationales: %.4f" % (avg_z_len / n) )
    print("Average sent number: %d" % real_max_sent_num)
    print("Maximum sent number: %d" % (avg_num_sent/n))
    print("Average rationle-sent number: %d" % (avg_r_num / n))
    
    print("Number of multi-sent rationales: %d" % count_cross_sent)
    
    print(avg_r_len)
    print(avg_r_num)
    print(n)

    if sent_level:
        return ts, ys, zs, ss, s_labels
    
    return ts, ys, zs, ss


# data_dir = "/dccstor/yum-dbqa/Rationale/structured_rationale/invariant_representation/data/movies"
# doc_dir = "/dccstor/yum-dbqa/Rationale/structured_rationale/invariant_representation/data/movies/docs"

# # vocab, D_tr, D_dev, D_te = get_beer_datasets(data_dir, max_seq_len=300)

# print("Train set: ")
# train_outputs = check_rationales(os.path.join(data_dir, "train.jsonl"), doc_dir)
# print("Dev set: ")
# dev_outputs = check_rationales(os.path.join(data_dir, "val.jsonl"), doc_dir)
# print("Test set: ")
# test_outputs = check_rationales(os.path.join(data_dir, "test.jsonl"), doc_dir)
# # t_d, y_d, z_d, s_d = dev_outputs


# In[ ]:


print((2067.0 + 1517.0) / 399)
print((47928.0 + 9953.0) / 399)
print(9953.0 / 200)

print((773* 1600 + 765 * 200 + 799 * 199) / (1600+200+199)) 
print(145/774)


# In[ ]:





# In[ ]:


from tqdm import tqdm

def get_examples(fpath, doc_dir, max_seq_len=-1, max_sent_num=200, sent_level=True):
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
    
    s_labels = []
    
    min_len = 10000
    max_len = 0
    avg_z_len = 0.
    avg_num_sent = 0.
    real_max_sent_num = 0
    
    avg_r_num = 0.

    with open(fpath, "r") as f:
        for line in tqdm(f):
            json_data = json.loads(line.strip())
            
            doc_filename = json_data['annotation_id']
            file_doc = open(os.path.join(doc_dir, doc_filename))
            sentences = file_doc.readlines()

            s_masks = []
            sentences = [s.strip().split() for s in sentences]
            
            t = [inner for outer in sentences for inner in outer]
            
            cur_id = 0
            for sentence in sentences:
                if len(s_masks) < max_sent_num:
                    s_masks.append([0.0] * len(t))
                    
                for token in sentence:
                    s_masks[-1][cur_id] = 1.0
                    cur_id += 1
                    
            avg_num_sent += len(s_masks)
            if len(s_masks) > real_max_sent_num:
                real_max_sent_num = len(s_masks)
                    
            if max_seq_len > 0:
                t = t[:max_seq_len]
#             print(t)

            if len(t) > max_len:
                max_len = len(t)
            if len(t) < min_len:
                min_len = len(t)

            y = json_data['classification']
            if y == 'POS':
                y = 1
            elif y == 'NEG':
                y = 0
            else:
                print('ERROR: label {}'.format(y))
                
            evidences = json_data['evidences']
            z = [0] * len(t)
            z_len = 0
            for evidence_list in evidences:
                for evidence in evidence_list:
                    z_start = evidence['start_token']
                    z_end = evidence['end_token']
                    z_end = min(z_end, len(t))
                    z_text = evidence['text']
                    for idx in range(z_start, z_end):
                        z[idx] = 1
                        z_len += 1

                    if max_seq_len < 0:
                        assert z_text == ' '.join(t[z_start:z_end]), z_text + '<->' + ' '.join(t[z_start:z_end])
                    else:
                        if z_end < max_seq_len:
                            assert z_text == ' '.join(t[z_start:z_end]), z_text + '<->' + ' '.join(t[z_start:z_end])
#                     print(z_text)
#                     print(t[z_start:z_end])
            avg_z_len += z_len
    
            if sent_level:
                s_label = [0.] * len(s_masks)
                new_z = [0] * len(t)
                for sid, s_mask in enumerate(s_masks):
                    is_rationale = False
                    for idx, val in enumerate(s_mask):
                        if idx >= max_seq_len:
                            continue
                        if val == 1.0:
                            if z[idx] == 1:
                                is_rationale = True
                                break
                    if is_rationale:
                        avg_r_num += 1
                        s_label[sid] = 1.
                        for idx, val in enumerate(s_mask):
                            if idx >= max_seq_len:
                                continue
                            if val == 1.0:
                                new_z[idx] = 1
#                 z = new_z
                    
    
#             break
                
#             s_spans = json_data['sentences']
# #             if len(s_spans) > max_sent_num:
# #                 max_sent_num = len(s_spans)
# # #                 print(line)
            
#             s_masks = []
#             for sid, s_span in enumerate(s_spans):
#                 (b, e) = s_span
                
#                 if b >= max_seq_len:
#                     break
                
# #                 print(len(s_masks))
# #                 print(max_sent_num)
#                 if len(s_masks) < max_sent_num:
#                     s_masks.append([0.0] * len(t))
#                 for i in range(b, e):
# #                     print(len(s_masks[-1]), i)
#                     if i >= max_seq_len:
#                         break
#                     s_masks[-1][i] = 1.0

#             if len(s_masks) > real_max_sent_num:
#                 real_max_sent_num = len(s_masks)

            ts.append(t)
            ys.append(y)
            zs.append(z)
            ss.append(s_masks)
            
            if sent_level:
                s_labels.append(s_label)
#                 print('len s_mask:', len(s_masks))
#                 print('len s_label:', len(s_label))
                assert len(s_masks) == len(s_label)

            n += 1
#     print(avg_z_len)
    print("Number of examples: %d" % n)
    print("Maximum doc length: %d" % max_len)
    print("Minimum doc length: %d" % min_len)
    print("Average length of rationales: %.4f" % (avg_z_len / n) )
    print("Average sent number: %d" % (avg_num_sent/n))
    print("Maximum sent number: %d" % real_max_sent_num)
    print("Average rationle-sent number: %d" % (avg_r_num / n))

    if sent_level:
        return ts, ys, zs, ss, s_labels
    
    return ts, ys, zs, ss


# data_dir = "/dccstor/yum-dbqa/Rationale/structured_rationale/invariant_representation/data/movies"
# doc_dir = "/dccstor/yum-dbqa/Rationale/structured_rationale/invariant_representation/data/movies/docs"

# vocab, D_tr, D_dev, D_te = get_beer_datasets(data_dir, max_seq_len=300)


# print("Train set: ")
# train_outputs = get_examples(os.path.join(data_dir, "train.jsonl"), doc_dir)
# print("Dev set: ")
# dev_outputs = get_examples(os.path.join(data_dir, "val.jsonl"), doc_dir)
# print("Test set: ")
# test_outputs = get_examples(os.path.join(data_dir, "test.jsonl"), doc_dir)
# t_d, y_d, z_d, s_d = dev_outputs


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
        
def display_rationale(words, rationales):

    for w, z in zip(words, rationales):
        if z == 0:
            sys.stdout.write(" " + w)
        else:
            sys.stdout.write(" " + "\033[4m" + w + "\033[0m")
    sys.stdout.write("\n")
    sys.stdout.flush()

# t_d, y_d, z_d, s_d = train_outputs
# display_sentences(t_d[9], s_d[9])
# # display_sentences(t_d[10], s_d[10])


# In[ ]:


class MovieDataset(Dataset):
    """Movie dataset from ERASER."""

    def __init__(self, data, stoi, max_seq_len=-1, max_sent_num=-1, transform=None):
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

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        
        texts, ys, zs, ss = self.data
                
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        text = texts[idx]
        sent_mask = ss[idx]
        z = zs[idx]
#         print('num sent ori:', len(sent_mask))
#         print('len of sent ori:', len(text))

        if self.max_seq_len > 0:
            if len(text) > self.max_seq_len:
                text = text[0:self.max_seq_len]
                z = z[0:self.max_seq_len]
                for sid in range(len(sent_mask)):
                    sent_mask[sid] = sent_mask[sid][0:self.max_seq_len]
                    
            for sid in range(len(sent_mask)):
                sent_mask[sid] = sent_mask[sid][0:self.max_seq_len]

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
            z.append(0.)
            for s in sent_mask:
                s.append(0.)
                
        while len(sent_mask) < self.max_sent_num:
            sent_mask.append([0.] * self.max_seq_len)
            
#         print(len(sent_mask))
#         print(len(sent_mask[0]))
            
        assert len(x) == self.max_seq_len
        assert len(mask) == self.max_seq_len
        assert len(z) == self.max_seq_len
        
#         print('num sent:', len(sent_mask))
#         print('len of sent:', len(x))
#         print('len of sent:', len(sent_mask[0]))
#         print('self.max_seq_len:', self.max_seq_len)
        for i, s in enumerate(sent_mask):
            if len(s) != len(sent_mask[0]):
                print('len mismatch {}:'.format(i), len(s), len(sent_mask[0]))
                display_sentences(text, sent_mask)
        
        sample = {"x": np.array(x, dtype=np.int64), 
                  "mask": np.array(mask, dtype=np.float32),
                  "y": int(ys[idx]),
                  "sent_mask": np.array(sent_mask, dtype=np.float32),
                  "z": np.array(z, dtype=np.float32)
                 }

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class MovieDatasetSentEval(Dataset):
    """Movie dataset from ERASER."""

    def __init__(self, data, stoi, max_seq_len=-1, max_sent_num=-1, transform=None):
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
        print('max seq len:', self.max_seq_len)

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        
        texts, ys, zs, ss, sls = self.data
                
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        text = texts[idx]
        sent_mask = ss[idx]
        sent_label = sls[idx]
#         print(sent_label)
        z = zs[idx]
#         print('num sent ori:', len(sent_mask))
#         print('len of sent ori:', len(text))
#         print('len s0 before:', len(sent_mask[0]))

        if self.max_seq_len > 0:
            if len(text) > self.max_seq_len:
                text = text[0:self.max_seq_len]
                z = z[0:self.max_seq_len]
                for sid in range(len(sent_mask)):
                    sent_mask[sid] = sent_mask[sid][0:self.max_seq_len]
                    
            for sid in range(len(sent_mask)):
                sent_mask[sid] = sent_mask[sid][0:self.max_seq_len]
                    
#         print('len s0 after:', len(sent_mask[0]))

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
            z.append(0.)
            for s in sent_mask:
                s.append(0.)
                
#         print('len sent_mask original:', len(sent_mask))
#         print('len sent_label original:', len(sent_label))
                
        while len(sent_mask) < self.max_sent_num:
            sent_mask.append([0.] * self.max_seq_len)
            sent_label.append(0.)
            
#         print(len(sent_mask))
#         print(len(sent_mask[0]))

#         print('len x:', len(x))
#         print('len sent_mask:', len(sent_mask))
#         print('len sent_label:', len(sent_label))
            
        assert len(x) == self.max_seq_len
        assert len(mask) == self.max_seq_len
        assert len(z) == self.max_seq_len
        assert len(sent_label) == self.max_sent_num, '{}\t{}'.format(len(sent_label), self.max_sent_num)
        
#         print('num sent:', len(sent_mask))
#         print('len of sent:', len(x))
#         print('len of sent:', len(sent_mask[0]))
#         print('self.max_seq_len:', self.max_seq_len)
        for i, s in enumerate(sent_mask):
            if len(s) != len(sent_mask[0]):
                print('len mismatch {}:'.format(i), len(s), len(sent_mask[0]))
                display_sentences(text, sent_mask)
        
        sample = {"x": np.array(x, dtype=np.int64), 
                  "mask": np.array(mask, dtype=np.float32),
                  "y": int(ys[idx]),
                  "sent_mask": np.array(sent_mask, dtype=np.float32),
                  "z": np.array(z, dtype=np.float32),
                  "sent_label": np.array(sent_label, dtype=np.float32)
                 }

        if self.transform:
            sample = self.transform(sample)

        return sample


# In[ ]:


def get_movie_datasets(data_dir, doc_dir, max_seq_len=300, max_sent_num=10, word_thres=1):
    """
    Get datasets (train, dev and test).
    """
    
    ##### load data from file
    
    # train set
    print("Training set: ")
    tr_outputs = get_examples(os.path.join(data_dir, "train.jsonl"), doc_dir, max_seq_len=max_seq_len,
                             sent_level=False)
    t_tr, y_tr, z_tr, s_tr = tr_outputs

    # dev set
    print("Dev set: ")
    dev_outputs = get_examples(os.path.join(data_dir, "val.jsonl"), doc_dir, max_seq_len=max_seq_len,
                              sent_level=False)
    t_d, y_d, z_d, s_d = dev_outputs
    
#     display_rationale(t_d[0], z_d[0])

    # test set
    print("Test set: ")
    te_outputs = get_examples(os.path.join(data_dir, "test.jsonl"), doc_dir, max_seq_len=max_seq_len,
                             sent_level=False)
    t_te, y_te, z_te, s_te = te_outputs

    # constrcut word dictionary
    texts = t_tr + t_d + t_te
    words = [word.strip() for text in texts for word in text]
    vocab = Vocab(collections.Counter(words), vectors="glove.6B.100d", min_freq=word_thres)    
    wv_size = vocab.vectors.size()

    print('Total num. of words: %d\nWord vector dimension: %d' % (wv_size[0], wv_size[1]))

    ##### construct torch datasets
    
    D_tr = MovieDataset(tr_outputs, vocab.stoi, max_seq_len, max_sent_num)
    D_dev = MovieDataset(dev_outputs, vocab.stoi, max_seq_len, max_sent_num)
    D_te = MovieDataset(te_outputs, vocab.stoi, max_seq_len, max_sent_num)
    
    return vocab, D_tr, D_dev, D_te


def get_movie_datasets_sent_eval(data_dir, doc_dir, max_seq_len=300, max_sent_num=10, word_thres=1):
    """
    Get datasets (train, dev and test).
    """
    
    ##### load data from file
    
    # train set
    print("Training set: ")
    tr_outputs = get_examples(os.path.join(data_dir, "train.jsonl"), doc_dir, max_seq_len=max_seq_len)
    t_tr, y_tr, z_tr, s_tr, sl_tr = tr_outputs

    # dev set
    print("Dev set: ")
    dev_outputs = get_examples(os.path.join(data_dir, "val.jsonl"), doc_dir, max_seq_len=max_seq_len)
    t_d, y_d, z_d, s_d, sl_d = dev_outputs
    
#     display_rationale(t_d[0], z_d[0])

    # test set
    print("Test set: ")
    te_outputs = get_examples(os.path.join(data_dir, "test.jsonl"), doc_dir, max_seq_len=max_seq_len)
    t_te, y_te, z_te, s_te, sl_te = te_outputs

    # constrcut word dictionary
    texts = t_tr + t_d + t_te
    words = [word.strip() for text in texts for word in text]
    vocab = Vocab(collections.Counter(words), vectors="glove.6B.100d", min_freq=word_thres)    
    wv_size = vocab.vectors.size()

    print('Total num. of words: %d\nWord vector dimension: %d' % (wv_size[0], wv_size[1]))

    ##### construct torch datasets
    
    D_tr = MovieDatasetSentEval(tr_outputs, vocab.stoi, max_seq_len, max_sent_num)
    D_dev = MovieDatasetSentEval(dev_outputs, vocab.stoi, max_seq_len, max_sent_num)
    D_te = MovieDatasetSentEval(te_outputs, vocab.stoi, max_seq_len, max_sent_num)
    
    return vocab, D_tr, D_dev, D_te


# In[ ]:


# data_dir = "/dccstor/yum-dbqa/Rationale/structured_rationale/invariant_representation/data/movies"
# doc_dir = "/dccstor/yum-dbqa/Rationale/structured_rationale/invariant_representation/data/movies/docs"
# vocab, D_tr, D_dev, D_te = get_movie_datasets(data_dir, doc_dir, max_seq_len=2200, max_sent_num=128)

# from torch.utils.data import DataLoader

# D_dev_ = DataLoader(D_dev, batch_size=10, shuffle=False, num_workers=4)
# # D_dev_ = DataLoader(D_tr, batch_size=10, shuffle=False, num_workers=1)

# for i_batch, data in enumerate(D_dev_):
#     print(i_batch)
#     x = data["x"]
#     mask = data["mask"]
#     y = data["y"]
#     z = data['z']
#     sent_mask = data["sent_mask"]
    
#     break


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
        
def display_rationale_with_vocab(vocab, words, rationales):

    for w, z in zip(words, rationales):
        w = vocab[w]
        if w == '<pad>':
            continue
        if z == 0:
            sys.stdout.write(" " + w)
        else:
            sys.stdout.write(" " + "\033[4m" + w + "\033[0m")
    sys.stdout.write("\n")
    sys.stdout.flush()

# print(x.cpu().numpy()[0])
# print(z.cpu().numpy()[0])
# print(torch.sum(z[0]))
# display_rationale_with_vocab(vocab.itos, x.cpu().numpy()[0], z.cpu().numpy()[0])
# display_sentences_with_vocab(vocab.itos, x.cpu().numpy()[0], sent_mask.cpu().numpy()[0])


# {"annotation_id": "negR_805.txt", "classification": "NEG", "evidences": [[{"docid": "negR_805.txt", "end_sentence": 17, "end_token": 257, "start_sentence": 16, "start_token": 250, "text": "just as inane as this film is"}], [{"docid": "negR_805.txt", "end_sentence": 18, "end_token": 279, "start_sentence": 17, "start_token": 274, "text": "just does n't cut it"}], [{"docid": "negR_805.txt", "end_sentence": 12, "end_token": 198, "start_sentence": 11, "start_token": 191, "text": "renders the rest of the film pointless"}], [{"docid": "negR_805.txt", "end_sentence": 4, "end_token": 100, "start_sentence": 3, "start_token": 88, "text": "it 's an overrated horror film that ultimately makes no sense whatsoever"}], [{"docid": "negR_805.txt", "end_sentence": 10, "end_token": 159, "start_sentence": 9, "start_token": 151, "text": "there 's some absurd dialogue and conceptual problems"}]], "query": "What is the sentiment of this review?", "query_type": null}

# In[ ]:


data_dir = "/dccstor/yum-dbqa/Rationale/structured_rationale/invariant_representation/data/movies"
doc_dir = "/dccstor/yum-dbqa/Rationale/structured_rationale/invariant_representation/data/movies/docs"
# vocab, D_tr, D_dev, D_te = get_movie_datasets_sent_eval(data_dir, doc_dir, max_seq_len=2200, max_sent_num=128)

# from torch.utils.data import DataLoader

# D_dev_ = DataLoader(D_dev, batch_size=10, shuffle=False, num_workers=4)
# # D_dev_ = DataLoader(D_tr, batch_size=10, shuffle=False, num_workers=1)

# for i_batch, data in enumerate(D_dev_):
#     print(i_batch)
#     x = data["x"]
#     mask = data["mask"]
#     y = data["y"]
#     z = data['z']
#     sent_mask = data["sent_mask"]
    
#     break


# In[ ]:


# from torch.utils.data import DataLoader

# vocab, D_tr, D_dev, D_te = get_movie_datasets_sent_eval(data_dir, doc_dir, max_seq_len=512, max_sent_num=128)

# D_dev_ = DataLoader(D_dev, batch_size=10, shuffle=False, num_workers=1)
# # D_dev_ = DataLoader(D_tr, batch_size=10, shuffle=False, num_workers=1)

# for i_batch, data in enumerate(D_dev_):
#     print(i_batch)
#     x = data["x"]
#     mask = data["mask"]
#     y = data["y"]
#     z = data['z']
#     sent_mask = data["sent_mask"]
    
# #     break


# In[ ]:





# In[ ]:


def display_sentences_with_vocab_with_rationale(vocab, x, s_masks, s_labels):

    for sid, (s_mask, s_label) in enumerate(zip(s_masks, s_labels)):
        sys.stdout.write('s{}:'.format(sid))
        for word, z in zip(x, s_mask):
            if z == 0:
                continue
            word = vocab[word]
            if word == '<PAD>':
                continue
            if s_label == 1.:
                sys.stdout.write(" " + "\033[4m" + word + "\033[0m")
            else:
                sys.stdout.write(" " + word)
        sys.stdout.write("\n")
        sys.stdout.flush()

# print(data["sent_label"][0])
# display_rationale_with_vocab(vocab.itos, x.cpu().numpy()[0], z.cpu().numpy()[0])
# display_sentences_with_vocab_with_rationale(vocab.itos, x.cpu().numpy()[0], sent_mask.cpu().numpy()[0], data["sent_label"][0])


# In[ ]:




