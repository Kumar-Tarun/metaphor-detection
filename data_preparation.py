import numpy as np
import pandas as pd
import nltk
import tensorflow_hub as hub
import tensorflow as tf
import pickle
import logging
from tqdm import tqdm

logging.disable(logging.CRITICAL)

elmo_model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

DATA_PATH = './data/'
VUA_PATH = './data/vua/'
TOEFL_PATH = './data/toefl/'

def prepare_vua_data(data_file, save_file):
  '''
  :param data_file: path to file in data/vua directory from which to read the dataset (csv format)
  :param save_file: path to file in data/vua directory to which to write the formatted dataset
  '''
  data = pd.read_csv(VUA_PATH + data_file)

  sentences = []
  pos_tags = []
  labels = []
  for row in data.values:
    sen = str(row[2]).split(' ')
    sen_len = len(sen)
    cleaned_sen = []
    labels_array = []
    pos_tag = []
    for w in sen:
      if w.startswith('M_'):
        labels_array.append(1)
        w1 = w.replace('M_', '')
      else:
        labels_array.append(0)
        w1 = w
      cleaned_sen.append(w1)
      pos_tag.append(nltk.pos_tag([w1])[0][1])
    cleaned_sen = ' '.join(cleaned_sen).strip()
    sentences.append(cleaned_sen)
    labels.append(labels_array)
    pos_tags.append(pos_tag)

    try:
      assert len(pos_tags) == sen_len
    except:
      print(cleaned_sen)
      print(pos_tags)

  # prepare data according to the format of https://github.com/gao-g/metaphor-in-context
  df = pd.DataFrame({'txt_id': data['txt_id'].values, 'sen_ix': data['sentence_id'].values, 
                     'sentence': sentences, 'label_seq': labels, 'pos_seq': poss, 
                     'labeled_sentence': data['sentence_txt'].values, 'genre': np.empty_like(labels)})

  df.to_csv(VUA_PATH + save_file, index=False)

def prepare_toefl_data(dir_path, save_path):
  '''
  :param dir_path: path to directory in data/toefl directory which contains the essays
  :param save_path: path to file in data/toefl directory to write the formatted dataset
  '''
  essay_ids = []
  sen_ids = []
  sentences = []
  labeled_sentences = []
  label_seq = []
  pos_seq = []
  for essay_file in tqdm(sorted(os.listdir(TOEFL_PATH + dir_path))):
    fp = open(DATA_PATH + dir_path + essay_file, 'r')
    essay_id = essay_file[:-4]
    i = 1
    for line in fp:
      sen = str(line)
      labels = []
      pos_tags = []
      cleaned_sen = []
      essay_ids.append(essay_id)
      sen_ids.append(str(i))
      labeled_sentences.append(str(line))

      for w in sen.split():
        if(w.startswith('M_')):
          w1 = w.replace('M_', '')
          labels.append(1)
        else:
          w1 = w
          labels.append(0)

        pos_tags.append(nltk.pos_tag([w1])[0][1])
        cleaned_sen.append(w1)
      
      assert len(labels) == len(sen.split())
      assert len(pos_tags) == len(labels)
      assert len(cleaned_sen) == len(labels)
      sentences.append(' '.join(cleaned_sen))
      pos_seq.append(pos_tags)
      label_seq.append(labels)
      i += 1
    
  df = pd.DataFrame({'txt_id': essay_ids, 'sen_ix': sen_ids, 'sentence': sentences, 
                     'label_seq': label_seq, 'pos_seq': pos_seq, 
                     'labeled_sentence': labeled_sentences, 'genre': list(np.empty_like(label_seq))})

  df.to_csv(TOEFL_PATH + save_path, index=False)

def train_val_split(read_path):
  '''
  :param read_path: path to formatted train dataset file in data/ directory
  '''
  np.random.seed(101)
  df = pd.read_csv(DATA_PATH + read_path)
  df = df[df['sentence'].notna()]

  # print(len(df))
  r = int(len(df)/10)
  df = df.sample(frac=1, random_state=1001)

  val_df = df[:r]
  train_df = df[r:]

  val_path = read_path[:-4] + '_val.csv'
  train_path = read_path[:-4] + '_train.csv'

  val_df.to_csv(DATA_PATH + val_path, index=False)
  train_df.to_csv(DATA_PATH + train_path, index=False)

def compute_elmo_vectors(read_path, save_path):
  '''
  :param read_path: path to formatted dataset files in data/ directory
  :param save_path: path to pickle file in data/directory to which to write the elmo vectors
  '''
  df = pd.read_csv(DATA_PATH + read_path)

  dic = {}
  txt_ids = df['txt_id'].values
  sen_ids = df['sen_ix'].values
  sentences = df['sentence'].values
  assert len(txt_ids) == len(sentences) 

  batch_sentences = [sentences[i:min(i+64, len(sentences))] for i in range(0, len(sentences), 64)]
  batch_txt_ids = [txt_ids[i:min(i+64, len(txt_ids))] for i in range(0, len(txt_ids), 64)]
  batch_sen_ids = [sen_ids[i:min(i+64, len(sen_ids))] for i in range(0, len(sen_ids), 64)]
  batch_sen_len = [[len(sen.split(' ')) for sen in batch_sen] for batch_sen in batch_sentences]

  assert len(batch_sentences) == len(batch_txt_ids)

  for i in tqdm(range(len(batch_sen_ids))):
    sen = batch_sentences[i]
    txt_id = batch_txt_ids[i]
    sen_id = batch_sen_ids[i]
    sen_len = batch_sen_len[i]
    embed = elmo_model(sen, signature='default', as_dict=True)['elmo']

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      embeddings = sess.run(embed)
      # print(embeddings.shape)

    for t, s, e, l in zip(txt_id, sen_id, embeddings, sen_len):
      if t not in dic:
        dic[t] = {}

      dic[t][str(s)] = e[:l, :]

  sum = 0
  for k in dic:
    sum += len(dic[k].keys())

  assert len(sentences) == sum

  with open(DATA_PATH + save_path, 'wb+') as f:
    pickle.dump(dic, f)

def prepare_tokens_data(label_file, save_tokens_file):
  '''
  :param label_file: path to file in data/ directory which contains the test tokens
  :param save_tokens_file: path to pickle file in data/directory to write the offsets
                           of test tokens corresponding to a sentence
  '''
  labels = pd.read_csv(DATA_PATH + label_file, header=None)

  dic = {}
  for row in labels.values:
    text_id = row[0].split('_')
    txt_id = text_id[0]
    sen_id = text_id[1]
    offset = int(text_id[2]) - 1
    if txt_id not in dic:
      dic[txt_id] = {}
    if sen_id not in dic[txt_id]:
      dic[txt_id][sen_id] = [offset]
    else:
      dic[txt_id][sen_id].append(offset)

  with open(DATA_PATH + save_tokens_file, 'wb+') as  f:
    pickle.dump(dic, f)

def main():
  if sys.argv[1] == 'vua':
    prepare_vua_data('vuamc_corpus_train.csv', 'VUA_corpus.csv')
    prepare_vua_data('vuamc_corpus_test.csv', 'VUA_corpus_test.csv')
    train_val_split('vua/VUA_corpus.csv')
    compute_elmo_vectors('vua/VUA_corpus_train.csv', 'vua/elmo_train.pkl')
    compute_elmo_vectors('vua/VUA_corpus_val.csv', 'vua/elmo_val.pkl')
    compute_elmo_vectors('vua/VUA_corpus_test.csv', 'vua/elmo_test.pkl')
    prepare_tokens_data('vua/all_pos_test_tokens.csv', 'vua/all_pos_test_tokens.pkl')
    prepare_tokens_data('vua/verb_test_tokens.csv', 'vua/verb_test_tokens.pkl')
  elif sys.argv[1] == 'toefl':
    prepare_toefl_data('train_essays/', 'TOEFL_corpus.csv')
    prepare_toefl_data('test_essays/', 'TOEFL_corpus_test.csv')
    train_val_split('toefl/TOEFL_corpus.csv')
    compute_elmo_vectors('toefl/TOEFL_corpus_train.csv', 'toefl/elmo_train.pkl')
    compute_elmo_vectors('toefl/TOEFL_corpus_val.csv', 'toefl/elmo_val.pkl')
    compute_elmo_vectors('toefl/TOEFL_corpus_test.csv', 'toefl/elmo_test.pkl')
    prepare_tokens_data('toefl/all_pos_test_tokens.csv', 'toefl/all_pos_test_tokens.pkl')
    prepare_tokens_data('toefl/verb_test_tokens.csv', 'toefl/verb_test_tokens.pkl')
  else:
    print('Unknown option')

  if __name__ == "__main__":
    main()
