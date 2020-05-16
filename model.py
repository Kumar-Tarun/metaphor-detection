import torch.nn as nn
import torch.nn.functional as F
from allennlp.nn.util import sort_batch_by_length
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RNNSequenceModel(nn.Module):
    # num_classes: The number of classes in the classification problem.
    # embedding_dim: The input dimension
    # hidden_size: The size of the RNN hidden state.
    # num_layers: Number of layers to use in RNN
    # bidir: boolean of wether to use bidirectional or not in RNN
    # dropout1: dropout on input to RNN
    # dropout2: dropout in RNN
    # dropout3: dropout on hidden state of RNN to linear layer
    def __init__(self, num_classes, embedding_dim, hidden_size, num_layers, char_vocab_size, char_embed_dim, bidir=True,
                 dropout1=0.2, dropout2=0.2, dropout3=0.2, name='vua'):
        # Always call the superclass (nn.Module) constructor first
        super(RNNSequenceModel, self).__init__()
        self.char_emb = CharCNN(char_vocab_size, char_embed_dim)
        self.highway = HighWayNetwork(300+250)
        self.name = name

        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                           num_layers=num_layers, dropout=dropout2, batch_first=True, bidirectional=bidir)

        direc = 2 if bidir else 1
        # Set up the final transform to a distribution over classes.

        if name == 'vua':
            self.transform = nn.Sequential(nn.Linear(embedding_dim, hidden_size * direc),
                                          nn.Tanh()
                                          )
            self.features = nn.Sequential(nn.Linear(hidden_size * direc, 50, bias=False),
                                          nn.Tanh()
                                         )
            self.output_projection = nn.Linear(50, num_classes)
        else:
            self.output_projection = nn.Linear(hidden_size * direc, num_classes)

        # Dropout layer
        self.dropout_on_input_to_LSTM = nn.Dropout(dropout1)
        self.dropout_on_input_to_linear_layer = nn.Dropout(dropout3)
        # self.crf = CRF(num_classes, batch_first=True)

    def forward(self, inputs, lengths, char_seqs):

        char_emb_seq = self.char_emb(char_seqs)

        glove_part = inputs[:,:,:300]
        elmo_part = inputs[:,:,300:1324]
        pos_part = inputs[:,:,1324:]

        inputs = torch.cat((glove_part, char_emb_seq), dim=-1)

        inputs = self.highway(inputs)

        inputs = torch.cat([inputs, elmo_part, pos_part], dim=-1)

        embedded_input = self.dropout_on_input_to_LSTM(inputs)
        # Sort the embedded inputs by decreasing order of input length.
        # sorted_input shape: (batch_size, sequence_length, embedding_dim)
        (sorted_input, sorted_lengths, input_unsort_indices, _) = sort_batch_by_length(embedded_input, lengths)
        # Pack the sorted inputs with pack_padded_sequence.
        packed_input = pack_padded_sequence(sorted_input, sorted_lengths.data.tolist(), batch_first=True)
        # Run the input through the RNN.
        packed_sorted_output, _ = self.rnn(packed_input)
        # Unpack (pad) the input with pad_packed_sequence
        # Shape: (batch_size, sequence_length, hidden_size)
        sorted_output, _ = pad_packed_sequence(packed_sorted_output, batch_first=True)
        # Re-sort the packed sequence to restore the initial ordering
        # Shape: (batch_size, sequence_length, hidden_size)
        output = sorted_output[input_unsort_indices]

        input_encoding = self.dropout_on_input_to_linear_layer(output)

        if self.name == 'vua':
            projected_output = self.transform(inputs)
            multiplied_output = projected_output * input_encoding

            features = self.features(multiplied_output)

            unnormalized_output = self.output_projection(features)
        else:
            unnormalized_output = self.output_projection(input_encoding)

        output_distribution = F.log_softmax(unnormalized_output, dim=-1)
        return output_distribution, input_encoding, unnormalized_output

class SelfAttention(nn.Module):
  def __init__(self, emb, k, heads=8):
    super(SelfAttention1, self).__init__()
    self.k, self.heads = k, heads

    # These compute the queries, keys and values for all 
    # heads (as a single concatenated vector)

    self.tokeys    = nn.Linear(emb, k * heads, bias=False)
    self.toqueries = nn.Linear(emb, k * heads, bias=False)
    self.tovalues  = nn.Linear(emb, k * heads, bias=False)

    # This unifies the outputs of the different heads into 
    # a single k-vector
    self.unifyheads = nn.Linear(heads * k, k)

  def forward(self, x, pad_amounts):
    b, t, emb = x.size()

    h = self.heads
    k = self.k

    queries = self.toqueries(x).view(b, t, h, k)
    keys    = self.tokeys(x)   .view(b, t, h, k)
    values  = self.tovalues(x) .view(b, t, h, k)

    keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
    queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
    values = values.transpose(1, 2).contiguous().view(b * h, t, k)

    queries = queries / (k ** (1/4))
    keys    = keys / (k ** (1/4))

    # - get dot product of queries and keys, and scale
    dot = torch.bmm(queries, keys.transpose(1, 2))
    # - dot has size (b*h, t, t) containing raw weights

    # mask out padded tokens
    for i in range(b):
        dot[i, t-pad_amounts[i]:, t-pad_amounts[i]:] = float('-inf')


    dot = F.softmax(dot, dim=2) 
    # - dot now contains row-wise normalized weights

    # apply the self attention to the values
    out = torch.bmm(dot, values).view(b, h, t, k)

    out = out.transpose(1, 2).contiguous().view(b, t, h * k)
    return self.unifyheads(out)

class SelfAttentionNarrow(nn.Module):

    def __init__(self, emb, heads=8, mask=False):
        """
        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads
        self.mask = mask

        s = emb // heads
        # - We will break the embedding into `heads` chunks and feed each to a different attention head

        self.tokeys    = nn.Linear(s, s, bias=False)
        self.toqueries = nn.Linear(s, s, bias=False)
        self.tovalues  = nn.Linear(s, s, bias=False)

        self.unifyheads = nn.Linear(heads * s, emb)

    def forward(self, x, pad_amounts):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        s = e // h
        x = x.view(b, t, h, s)

        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)

        assert keys.size() == (b, t, h, s)
        assert queries.size() == (b, t, h, s)
        assert values.size() == (b, t, h, s)

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        # mask out padded tokens
        for i in range(b):
            dot[i, t-pad_amounts[i]:, t-pad_amounts[i]:] = float('-inf')

        assert dot.size() == (b*h, t, t)

        # if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
        #     mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)

class TransformerBlock(nn.Module):
  def __init__(self, emb, k, heads):
    super(TransformerBlock1, self).__init__()
    self.emb = emb
    self.k = k

    self.attention = SelfAttention(emb, k, heads=heads)
    # self.attention = SelfAttentionNarrow(emb, heads=heads)

    self.norm1 = nn.LayerNorm(k)
    self.norm2 = nn.LayerNorm(k)

    self.ff = nn.Sequential(
      nn.Linear(k, 4 * k),
      nn.ReLU(),
      nn.Linear(4 * k, k))

    self.transform = nn.Linear(emb, k)
    self.do = nn.Dropout(0.2)

  def forward(self, x):
    pad_amounts = x[1]
    x = x[0]

    attended = self.attention(x, pad_amounts)

    if self.emb != self.k:
        y = self.transform(x)
    else:
        y = x

    x = self.norm1(attended + y)
    x = self.do(x)

    fedforward = self.ff(x)
    x = self.norm2(fedforward + x)
    x = self.do(x)

    return {0:x, 1:pad_amounts}

class Transformer(nn.Module):
    def __init__(self, emb, k, heads, depth, num_classes, char_vocab_size, char_embed_dim, name='vua'):
        super(Transformer1, self).__init__()

        self.char_emb = CharCNN(char_vocab_size, char_embed_dim)
        self.name = name

        self.highway = HighWayNetwork(300+250)
        # The sequence of transformer blocks that does all the 
        # heavy lifting
        tblocks = []
        for i in range(depth):
            if(i != 0):
                tblocks.append(TransformerBlock(emb=k, k=k, heads=heads))
            else:
                tblocks.append(TransformerBlock(emb=emb, k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

        if name == 'vua':
            self.transform = nn.Sequential(nn.Linear(emb, k),
                                          nn.Tanh()
                                          )

            self.features = nn.Sequential(nn.Linear(k, 50, bias=False),
                                          nn.Tanh()
                                         )
            self.toprobs = nn.Linear(50, num_classes)

        else:
            self.toprobs = nn.Linear(k, num_classes)

    def forward(self, x, pad_amounts, char_seqs):

        char_emb_seq = self.char_emb(char_seqs)

        glove_part = x[:,:,:300]
        elmo_part = x[:,:,300:1324]
        pos_part = x[:,:,1324:]

        x = torch.cat((glove_part, char_emb_seq), dim=-1)

        x = self.highway(x)

        x = torch.cat([x, elmo_part, pos_part], dim=-1)

        y = self.tblocks({0:x, 1:pad_amounts})
        z = y[0]

        if self.name == 'vua':
            projected_output = self.transform(x)

            multiplied_output = projected_output * z

            features = self.features(multiplied_output)

            x = self.toprobs(features)
        else:
            x = self.toprobs(z)

        return F.log_softmax(x, dim=-1), y[0], x

class CharCNN(nn.Module):
  def __init__(self, vocab_size, embed_dim):
    super(CharCNN, self).__init__()

    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.char_emb = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

    self.conv_1 = nn.Sequential(nn.Conv1d(self.embed_dim, 25, kernel_size=1),
                                nn.Tanh()
                                )
    self.conv_2 = nn.Sequential(nn.Conv1d(self.embed_dim, 50, kernel_size=2),
                                nn.Tanh()
                                )

    self.conv_3 = nn.Sequential(nn.Conv1d(self.embed_dim, 75, kernel_size=3),
                                nn.Tanh()
                                )

    self.conv_4 = nn.Sequential(nn.Conv1d(self.embed_dim, 100, kernel_size=4),
                                nn.Tanh()
                                )

    self.conv = [self.conv_1, self.conv_2, self.conv_3, self.conv_4]

  def forward(self, x):
    chars = self.char_emb(x)
    b, t, w, k = chars.size()
    chars = chars.transpose(2, 3).contiguous().view(b*t, k, w)
    char_embs = []
    for layer in self.conv:
      y = layer(chars)
      y, _ = torch.max(y, -1)
      char_embs.append(y)

    y = torch.cat(char_embs, dim=1)
    y = y.view(b, t, -1)
    return y

class HighWayNetwork(nn.Module):
  def __init__(self, embed_dim):
    super(HighWayNetwork, self).__init__()
    self.embed_dim = embed_dim
    self.t1 = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),
                            nn.ReLU()
                            )
    self.t2 = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),
                            nn.Sigmoid()
                            )
  def forward(self, x):

    f1 = self.t1(x)
    t = self.t2(x)
    z = t * f1 + (1 - t) * x

    return z
