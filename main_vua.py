from util import *
from util import TextDatasetWithGloveElmoSuffix as TextDataset
from model import RNNSequenceModel, Transformer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import h5py
import ast
import numpy as np
import pandas as pd
import copy
import os
try:
  os.mkdir('./models/')
  os.mkdir('./models/vua/')
  os.mkdir('./graphs/')
  os.mkdir('./graphs/vua/')
  os.mkdir('./predictions/')
except:
  pass
import matplotlib.pyplot as plt

print("PyTorch version:")
print(torch.__version__)
print("GPU Detected:")
print(torch.cuda.is_available())
using_GPU = True
im = 'sc_dis'

"""
1. Data pre-processing
"""
'''
1.1 VUA
get raw dataset as a list:
  Each element is a triple:
    a sentence: string
    a list of labels: 
    a list of pos: 

'''
pos_set = set()

raw_train_vua = []
ma = 0
option = 'vua'
TRAIN_PATH = './data/vua/VUA_corpus_train.csv'
VAL_PATH = './data/vua/VUA_corpus_val.csv'
TEST_PATH = './data/vua/VUA_corpus_test.csv'
ELMO_TRAIN_PATH = './data/vua/elmo_train.pkl'
ELMO_VAL_PATH = './data/vua/elmo_val.pkl'
ELMO_TEST_PATH = './data/vua/elmo_test.pkl'
ALL_POS_TOKENS = './data/vua/all_pos_test_tokens.pkl'
VERB_TOKENS = './data/vua/verb_test_tokens.pkl'

with open(TRAIN_PATH, encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        pos_seq = ast.literal_eval(line[4])
        ma = max(ma, len(pos_seq))
        label_seq = ast.literal_eval(line[3])
        assert (len(pos_seq) == len(label_seq))
        assert (len(line[2].split()) == len(pos_seq))
        raw_train_vua.append([line[2], label_seq, pos_seq, line[0], line[1]])
        pos_set.update(pos_seq)

raw_val_vua = []
with open(VAL_PATH, encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        pos_seq = ast.literal_eval(line[4])
        ma = max(ma, len(pos_seq))
        label_seq = ast.literal_eval(line[3])
        assert (len(pos_seq) == len(label_seq))
        assert (len(line[2].split()) == len(pos_seq))
        raw_val_vua.append([line[2], label_seq, pos_seq, line[0], line[1]])
        pos_set.update(pos_seq)

print('Maximum seq length: ', ma)
# embed the pos tags

pos2idx, idx2pos = get_pos2idx_idx2pos(pos_set)

# print(pos2idx)
for i in range(len(raw_train_vua)):
    raw_train_vua[i][2] = index_sequence(pos2idx, raw_train_vua[i][2])
for i in range(len(raw_val_vua)):
    raw_val_vua[i][2] = index_sequence(pos2idx, raw_val_vua[i][2])
print('size of training set, validation set: ', len(raw_train_vua), len(raw_val_vua))


"""
2. Data preparation
"""
'''
2. 1
get vocabulary and glove embeddings in raw dataset 
'''
# vocab is a set of words
vocab, char_vocab = get_vocab(raw_train_vua)
try:
  with open('/data/vua/char_vocab.pkl', 'rb') as f:
    char_vocab = pickle.load(f)
except:
  with open('/data/vua/char_vocab.pkl', 'wb+') as f:
    pickle.dump(char_vocab, f)
# two dictionaries. <PAD>: 0, <UNK>: 1
word2idx, idx2word = get_word2idx_idx2word(vocab)

c2idx = get_c2idx(char_vocab)

# glove_embeddings a nn.Embeddings
glove_embeddings = get_embedding_matrix(word2idx, idx2word, name='vua', normalization=False)

# elmo_embeddings
with open(ELMO_TRAIN_PATH, 'rb') as f:
  elmos_train_vua = pickle.load(f)
with open(ELMO_VAL_PATH, 'rb') as f:
  elmos_val_vua = pickle.load(f)
with open(ELMO_TEST_PATH, 'rb') as f:
  elmos_test_vua = pickle.load(f)

pos_vocab_size = len(pos2idx)
pos_embeddings = nn.Embedding(pos_vocab_size, 30)
'''
2. 2
embed the datasets
'''
# raw_train_vua: sentence, label_seq, pos_seq
# embedded_train_vua: embedded_sentence, pos, labels
embedded_train_vua = [[embed_indexed_sequence(example[3], example[4], example[0], example[2], word2idx,
                                      glove_embeddings, elmos_train_vua, pos_embeddings),
                       example[2], example[1], get_char_indices(c2idx, example[0])]
                      for example in raw_train_vua]
embedded_val_vua = [[embed_indexed_sequence(example[3], example[4], example[0], example[2], word2idx,
                                    glove_embeddings, elmos_val_vua, pos_embeddings),
                     example[2], example[1], get_char_indices(c2idx, example[0])]
                    for example in raw_val_vua]

'''
2. 3
set up Dataloader for batching
'''
# Separate the input (embedded_sequence) and labels in the indexed train sets.
# embedded_train_vua: embedded_sentence, pos, labels
train_dataset_vua = TextDataset([example[0] for example in embedded_train_vua],
                                [example[1] for example in embedded_train_vua],
                                [example[2] for example in embedded_train_vua],
                                [example[3] for example in embedded_train_vua])
val_dataset_vua = TextDataset([example[0] for example in embedded_val_vua],
                              [example[1] for example in embedded_val_vua],
                              [example[2] for example in embedded_val_vua],
                              [example[3] for example in embedded_val_vua])

# Data-related hyperparameters
batch_size = 4
# Set up a DataLoader for the training, validation, and test dataset
train_dataloader_vua = DataLoader(dataset=train_dataset_vua, batch_size=batch_size, shuffle=True,
                              collate_fn=TextDataset.collate_fn)
val_dataloader_vua = DataLoader(dataset=val_dataset_vua, batch_size=batch_size,
                            collate_fn=TextDataset.collate_fn)

"""
3. Model training
"""
'''
3. 1 
set up model, loss criterion, optimizer
'''
# Instantiate the model
# embedding_dim = glove + elmo + suffix indicator
# dropout1: dropout on input to RNN
# dropout2: dropout in RNN; would be used if num_layers!=1
# dropout3: dropout on hidden state of RNN to linear layer
RNNseq_model = RNNSequenceModel(num_classes=2, embedding_dim=300+1024+250+30, 
                                hidden_size=300, num_layers=1, bidir=True,
                                char_vocab_size=len(c2idx), char_embed_dim=50,
                                dropout1=0.5, dropout2=0, dropout3=0.1)

Transformer_model = Transformer(emb=300+1024+250+30, k=300, heads=1, depth=1, 
                                seq_length=128, num_tokens=13845, num_classes=2,
                                char_vocab_size=len(c2idx), char_embed_dim=50)

transformer_parameters = sum(p.numel() for p in Transformer_model.parameters() if p.requires_grad)
rnn_parameters = sum(p.numel() for p in RNNseq_model.parameters() if p.requires_grad)
total_parameters = transformer_parameters + rnn_parameters
print(f'Number of parameters: {total_parameters}')

# Move the model to the GPU if available
if using_GPU:
    RNNseq_model = RNNseq_model.cuda()
    Transformer_model = Transformer_model.cuda()

# Set up criterion for calculating loss
weight_tensor = torch.Tensor([1.0, 2.0]).cuda()
loss_criterion = nn.NLLLoss(weight=weight_tensor)

rnn_optimizer = optim.Adam(RNNseq_model.parameters(), lr=0.005)
trans_optimizer = optim.Adam(Transformer_model.parameters(), lr=0.0001)

rnn_scheduler = optim.lr_scheduler.MultiStepLR(rnn_optimizer, milestones=[2,5], gamma=0.3)
trans_scheduler = optim.lr_scheduler.MultiStepLR(trans_optimizer, milestones=[2,5], gamma=0.3)

# Number of epochs (passes through the dataset) to train the model for.
num_epochs = 5
'''
3. 2
train model
'''
train_loss = []
val_loss = []
performance_matrix = None
val_f1s = []
train_f1s = []
# A counter for the number of gradient updates
num_iter = 0
comparable = []
bestf1 = 0
best_model_weights = [None, None]
best_optimizer_dict = [None, None]
for epoch in range(num_epochs):

    print("Starting epoch {}".format(epoch + 1))
    tp = 0
    fp = 0
    fn = 0
    total_loss = 0
    t = 0
    for (pos_seqs, example_text, example_lengths, labels, pad_amounts, char_seqs) in train_dataloader_vua:

        example_text = Variable(example_text)
        example_lengths = Variable(example_lengths)
        labels = Variable(labels)
        pad_amounts = Variable(pad_amounts)
        char_seqs = Variable(char_seqs)

        if using_GPU:
            example_text = example_text.cuda()
            example_lengths = example_lengths.cuda()
            labels = labels.cuda()
            pad_amounts = pad_amounts.cuda()
            char_seqs = char_seqs.cuda()

        predicted1, embs1, p1 = RNNseq_model(example_text, example_lengths, char_seqs)
        predicted2, embs2, p2 = Transformer_model(example_text, pad_amounts, char_seqs)

        # combine predictions
        # predicted = (predicted1 + predicted3)/2
        p = (p1 + p2)/2
        predicted = F.log_softmax(p, dim=-1)

        total_batch_loss1 = loss_criterion(predicted1.view(-1, 2), labels.view(-1))
        total_batch_loss2 = loss_criterion(predicted2.view(-1, 2), labels.view(-1))

        total_los = loss_criterion(predicted.view(-1, 2), labels.view(-1))
        # total_los = (total_batch_loss1 + total_batch_loss2)/2

        rnn_optimizer.zero_grad()
        trans_optimizer.zero_grad()

        (total_batch_loss1).backward()
        (total_batch_loss2).backward()

        rnn_optimizer.step()
        trans_optimizer.step()

        num_iter += 1
        total_loss += total_los
        if num_iter % 1000 == 0:
            avg_eval_loss, precision, recall, performance_matrix = evaluate(idx2pos, val_dataloader_vua,
                                                         loss_criterion, using_GPU, RNNseq_model, Transformer_model)
            try:
                val_f1 = 2*precision*recall/(precision+recall)
            except:
                val_f1 = 0
            val_loss.append(avg_eval_loss)
            val_f1s.append(performance_matrix[:, 2])
            print("Iteration {}. Validation Loss {} Val F1 {}.".format(num_iter, avg_eval_loss, val_f1))
            
        _, predicted_labels = torch.max(predicted.data, 2)
        for i in range(len(pos_seqs)):
            indexed_pos_sequence = pos_seqs[i]
            prediction = predicted_labels[i]
            label = labels.data[i]
            for j in range(len(indexed_pos_sequence)):
                indexed_pos = indexed_pos_sequence[j]
                p = prediction[j]
                l = label[j]
                if p == l and p == 1:
                    tp += 1
                elif p == 1 and l == 0:
                    fp += 1
                elif p == 0 and l == 1:
                    fn += 1   
    try:
        precis = tp/(tp + fp)
        recal = tp/(tp + fn)
        f1_score = 2*precis*recal/(precis+recal) 
    except:
        f1_score = 0
    train_f1s.append(f1_score)
    los = (total_loss/train_dataloader_vua.__len__()).item()
    train_loss.append(los)
    rnn_scheduler.step()
    trans_scheduler.step()
    print("Epoch {}. Training Loss {} Train F1 {}.".format(epoch+1, los, f1_score))

    _, precision, recall, _ = evaluate(idx2pos, val_dataloader_vua, 
                                       loss_criterion, using_GPU, 
                                       RNNseq_model, Transformer_model,
                                       )
    val_end_f1 = 2*precision*recall/(precision+recall)
    if val_end_f1 > bestf1:
      bestf1 = val_end_f1
      best_model_weights[0] = copy.deepcopy(RNNseq_model.state_dict())
      best_model_weights[1] = copy.deepcopy(Transformer_model.state_dict())
      best_optimizer_dict[0] = copy.deepcopy(rnn_optimizer.state_dict())
      best_optimizer_dict[1] = copy.deepcopy(trans_optimizer.state_dict())


RNNseq_model.load_state_dict(best_model_weights[0])
Transformer_model.load_state_dict(best_model_weights[1])
rnn_optimizer.load_state_dict(best_optimizer_dict[0])
trans_optimizer.load_state_dict(best_optimizer_dict[1])

torch.save({
            'epoch': num_epochs,
            'model_state_dict': best_model_weights,
            'optimizer_state_dict':best_optimizer_dict,
            }, './models/vua/model.tar')

"""
3.3
plot the training process: losses for validation and training dataset
"""
plt.figure(0)
plt.title('Loss for VUA dataset')
plt.xlabel('iteration (unit:1000)')
plt.ylabel('Loss')
plt.plot(val_loss, 'g')
plt.plot(train_loss, 'b')
plt.legend(['Validation loss', 'Training loss'], loc='upper right')
# plt.show()
plt.savefig(f'./graphs/vua/Loss_{im}.png')


plt.figure(1)
plt.title('Validation F1 for VUA dataset')
plt.xlabel('iteration (unit:1000)')
plt.ylabel('F1')
for i in range(len(idx2pos)):
    plt.plot([x[i] for x in val_f1s])
plt.legend([idx2pos[i] for i in range(len(idx2pos))], loc='upper left')
plt.savefig(f'./graphs/vua/val_f1_{im}.png')

print("**********************************************************")
print("Evalutation on test set (new) (all pos): ")

raw_test_vua = []
with open(TEST_PATH, encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        # txt_id	sen_ix	sentence	label_seq	pos_seq	labeled_sentence	genre
        pos_seq = ast.literal_eval(line[4])
        label_seq = ast.literal_eval(line[3])
        assert(len(pos_seq) == len(label_seq))
        assert(len(line[2].split()) == len(pos_seq))
        raw_test_vua.append([line[2], label_seq, pos_seq, line[0], line[1]])
print('number of examples(sentences) for test_set ', len(raw_test_vua))

for i in range(len(raw_test_vua)):
    raw_test_vua[i][2] = index_sequence(pos2idx, raw_test_vua[i][2])

embedded_test_vua = [[embed_indexed_sequence(example[3], example[4], example[0], example[2], word2idx,
                                      glove_embeddings, elmos_test_vua, pos_embeddings),
                       example[2], example[1], get_char_indices(c2idx, example[0])]
                      for example in raw_test_vua]

ids = [[example[3], example[4], example[0], example[2]] for example in raw_test_vua]

# Separate the input (embedded_sequence) and labels in the indexed train sets.
# embedded_train_vua: embedded_sentence, pos, labels
test_dataset_vua = TextDataset([example[0] for example in embedded_test_vua],
                              [example[1] for example in embedded_test_vua],
                              [example[2] for example in embedded_test_vua],
                              [example[3] for example in embedded_test_vua])

# Set up a DataLoader for the test dataset
test_dataloader_vua = DataLoader(dataset=test_dataset_vua, batch_size=batch_size,
                              collate_fn=TextDataset.collate_fn)

preds, test_pos_seqs = evaluate(idx2pos, test_dataloader_vua, loss_criterion, using_GPU, RNNseq_model, Transformer_model, 'test')

with open(ALL_POS_TOKENS, 'rb') as f:
  test_dic = pickle.load(f)

new_preds = []
new_ids = []

for i, arr in enumerate(ids):
  id1 = arr[0]
  id2 = arr[1]
  try:
    offsets = test_dic[id1][id2]
  except:
    continue
  pred = preds[i]
  new_pred = []
  new_id = []
  for o in offsets:
    new_pred.append(pred[o])
    new_id.append('_'.join([id1, id2, str(o+1)]))

  assert len(new_pred) == len(offsets)
  assert len(new_id) == len(offsets)

  new_preds.extend(new_pred)
  new_ids.extend(new_id)

test_predictions = [[i, p] for i, p in zip(new_ids, new_preds)]
df = pd.DataFrame(test_predictions)
df.to_csv('./predictions/' + option + '_all_pos_pred.csv', index=False, header=False)


print("Evalutation on test set (new) (only verb): ")

with open(VERB_TOKENS, 'rb') as f:
  verb_test_dic = pickle.load(f)

new_preds = []
new_ids = []

for i, arr in enumerate(ids):
  id1 = arr[0]
  id2 = arr[1]
  try:
    offsets = verb_test_dic[id1][id2]
  except:
    continue
  pred = preds[i]
  new_pred = []
  new_id = []
  for o in offsets:
    new_pred.append(pred[o])
    new_id.append('_'.join([id1, id2, str(o+1)]))

  assert len(new_pred) == len(offsets)
  assert len(new_id) == len(offsets)

  new_preds.extend(new_pred)
  new_ids.extend(new_id)

test_predictions = [[i, p] for i, p in zip(new_ids, new_preds)]
df = pd.DataFrame(test_predictions)
df.to_csv('./predictions/' + option + '_verb_pred.csv', index=False, header=False)
