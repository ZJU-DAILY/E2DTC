import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.parameter import Parameter
import os
import constants
from pretrain import pretrain_ae
from cluster import init_cluster


class StackingGRUCell(nn.Module):
    """
    Multi-layer CRU Cell
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(StackingGRUCell, self).__init__()
        self.num_layers = num_layers
        self.grus = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.grus.append(nn.GRUCell(input_size, hidden_size))
        for i in range(1, num_layers):
            self.grus.append(nn.GRUCell(hidden_size, hidden_size))

    def forward(self, input, h0):
        """
        Input:
        input (batch, input_size): input tensor
        h0 (num_layers, batch, hidden_size): initial hidden state
        ---
        Output:
        output (batch, hidden_size): the final layer output tensor
        hn (num_layers, batch, hidden_size): the hidden state of each layer
        """
        hn = []
        output = input
        for i, gru in enumerate(self.grus):
            hn_i = gru(output, h0[i])
            hn.append(hn_i)
            if i != self.num_layers - 1:
                output = self.dropout(hn_i)
            else:
                output = hn_i
        hn = torch.stack(hn)
        return output, hn


class GlobalAttention(nn.Module):
    """
    $$a = \sigma((W_1 q)H)$$
    $$c = \tanh(W_2 [a H, q])$$
    """

    def __init__(self, hidden_size):
        super(GlobalAttention, self).__init__()
        self.L1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.L2 = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, q, H):
        """
        Input:
        q (batch, hidden_size): query
        H (batch, seq_len, hidden_size): context
        ---
        Output:
        c (batch, hidden_size)
        """
        # (batch, hidden_size) => (batch, hidden_size, 1)
        q1 = self.L1(q).unsqueeze(2)
        # (batch, seq_len)
        a = torch.bmm(H, q1).squeeze(2)
        a = self.softmax(a)
        # (batch, seq_len) => (batch, 1, seq_len)
        a = a.unsqueeze(1)
        # (batch, hidden_size)
        c = torch.bmm(a, H).squeeze(1)
        # (batch, hidden_size * 2)
        c = torch.cat([c, q], 1)
        return self.tanh(self.L2(c))


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout,
                 bidirectional, embedding):
        """
        embedding (vocab_size, input_size): pretrained embedding
        """
        super(Encoder, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        assert hidden_size % self.num_directions == 0
        # encoder and decoder have the same hidden size
        self.hidden_size = hidden_size // self.num_directions
        self.num_layers = num_layers

        self.embedding = embedding
        self.rnn = nn.GRU(input_size, self.hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          dropout=dropout)

    def forward(self, input, lengths, h0=None):
        """
        Input:
        input (seq_len, batch): padded sequence tensor
        lengths (1, batch): sequence lengths
        h0 (num_layers*num_directions, batch, hidden_size): initial hidden state
        ---
        Output:
        hn (num_layers*num_directions, batch, hidden_size):
            the hidden state of each layer
        output (seq_len, batch, hidden_size*num_directions): output tensor
        """
        # (seq_len, batch) => (seq_len, batch, input_size)
        embed = self.embedding(input)
        lengths = lengths.data.view(-1).tolist()
        if lengths is not None:
            embed = pack_padded_sequence(embed, lengths)
        output, hn = self.rnn(embed, h0)
        if lengths is not None:
            output = pad_packed_sequence(output)[0]
        return hn, output


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, embedding):
        super(Decoder, self).__init__()
        self.embedding = embedding
        self.rnn = StackingGRUCell(input_size, hidden_size, num_layers,
                                   dropout)
        self.attention = GlobalAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, input, h, H, use_attention=True):
        """
        Input:
        input (seq_len, batch): padded sequence tensor
        h (num_layers, batch, hidden_size): input hidden state
        H (seq_len, batch, hidden_size): the context used in attention mechanism
            which is the output of encoder
        use_attention: If True then we use attention
        ---
        Output:
        output (seq_len, batch, hidden_size)
        h (num_layers, batch, hidden_size): output hidden state,
            h may serve as input hidden state for the next iteration,
            especially when we feed the word one by one (i.e., seq_len=1)
            such as in translation
        """
        assert input.dim() == 2, "The input should be of (seq_len, batch)"
        # (seq_len, batch) => (seq_len, batch, input_size)
        embed = self.embedding(input)
        output = []
        # split along the sequence length dimension
        for e in embed.split(1):
            e = e.squeeze(0)  # (1, batch, input_size) => (batch, input_size)
            o, h = self.rnn(e, h)
            if use_attention:
                o = self.attention(o, H.transpose(0, 1))
            o = self.dropout(o)
            output.append(o)
        output = torch.stack(output)
        return output, h


class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size,
                 hidden_size, num_layers, dropout, bidirectional):
        super(EncoderDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        # the embedding shared by encoder and decoder
        # just look up table from indices to vectors
        self.embedding = nn.Embedding(vocab_size, embedding_size,
                                      padding_idx=constants.PAD)
        self.encoder = Encoder(embedding_size, hidden_size, num_layers,
                               dropout, bidirectional, self.embedding)
        self.decoder = Decoder(embedding_size, hidden_size, num_layers,
                               dropout, self.embedding)
        self.num_layers = num_layers

    def encoder_hn2decoder_h0(self, h):
        """
        Input:
        h (num_layers * num_directions, batch, hidden_size): encoder output hn
        ---
        Output:
        h (num_layers, batch, hidden_size * num_directions): decoder input h0
        """
        if self.encoder.num_directions == 2:
            num_layers, batch, hidden_size = h.size(0)//2, h.size(1), h.size(2)
            return h.view(num_layers, 2, batch, hidden_size)\
                    .transpose(1, 2).contiguous()\
                    .view(num_layers, batch, hidden_size * 2)
        else:
            return h

    def encoder_hn(self, src, lengths):
        """
        Input:
        src (src_seq_len, batch): source tensor
        lengths (1, batch): source sequence lengths
        ---
        Output:
        context (batch, hidden_size * num_directions)
        """
        encoder_hn, _ = self.encoder(src, lengths)
        # (num_layers, batch, hidden_size * num_directions)
        decoder_h0 = self.encoder_hn2decoder_h0(encoder_hn)

        # use the last layer outputs as trajectory representation
        # (batch, hidden_size * num_directions)
        context = decoder_h0[self.num_layers-1]
        return context

    def forward(self, src, lengths, trg):
        """
        Input:
        src (src_seq_len, batch): source tensor
        lengths (1, batch): source sequence lengths
        trg (trg_seq_len, batch): target tensor, the `seq_len` in trg is not
            necessarily the same as that in src
        ---
        Output:
        output (trg_seq_len, batch, hidden_size)
        context (batch, hidden_size * num_directions)
        """
        encoder_hn, H = self.encoder(src, lengths)
        # (num_layers, batch, hidden_size * num_directions)
        decoder_h0 = self.encoder_hn2decoder_h0(encoder_hn)

        # use the last layer outputs as trajectory representation
        # (batch, hidden_size * num_directions)
        context = decoder_h0[self.num_layers-1]

        # for target we feed the range [BOS:EOS-1] into decoder
        output, decoder_hn = self.decoder(trg[:-1], decoder_h0, H)

        return output, context


class clusterLayer(nn.Module):
    def __init__(self, args, alpha=1):
        super(clusterLayer, self).__init__()

        self.clusters = Parameter(torch.Tensor(
            args.n_clusters, args.hidden_size), requires_grad=False)
        self.alpha = alpha

    def forward(self, context):
        # clustering: caculate Student’s t-distribution
        # clusters (n_clusters, hidden_size * num_directions)
        # context (batch, hidden_size * num_directions)
        # q (batch,n_clusters): similarity between embedded point and cluster center
        distance = torch.sum(
            torch.pow(context.unsqueeze(1) - self.clusters, 2), 2)
        q = 1.0 / (1.0 + distance / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


class DTC(nn.Module):
    def __init__(self, args, device, alpha=1):
        super(DTC, self).__init__()
        '''
        autoencoder(device[0]): autoencoder, map input to output.
        rclayer(device[1]): reconstructionlayer, map the output of EncoderDecoder into the vocabulary space and do log transform for KLDIVLoss
        clusterlayer(device[2])
        '''
        self.args = args
        self.device = device
        self.loss_cuda = device[3]
        self.autoencoder = EncoderDecoder(args.vocab_size,
                                          args.embedding_size,
                                          args.hidden_size,
                                          args.num_layers,
                                          args.dropout,
                                          args.bidirectional).to(device[0])
        self.rclayer = nn.Sequential(nn.Linear(args.hidden_size, args.vocab_size),
                                     nn.LogSoftmax(dim=1)).to(device[1])
        self.clusterlayer = clusterLayer(args, alpha).to(device[2])

    def forward(self, src, lengths, trg):
        '''
        Input in cuda0
        Output in cuda1
        output(trg_seq_len, batch, hidden_size)
        context(batch, hidden_size * num_directions)
        q(n_clusters, batch): similarity between embedded point and cluster center
        '''
        output, context = self.autoencoder(src, lengths, trg)
        q = self.clusterlayer(context.to(self.cuda2))

        return output, q

    def pretrain(self):
        pretrain_ae([self.autoencoder, self.rclayer],
                    self.args, self.device[0], self.device[1], self.loss_cuda)
        with torch.no_grad():
            init_cluster([self.autoencoder, self.clusterlayer],
                         self.args, self.device[0], self.device[2])
