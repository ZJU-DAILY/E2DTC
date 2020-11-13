import torch
import torch.nn as nn
import h5py
import os
import constants


def KLDIVloss(output, target, V, D, loss_cuda):
    """
    output (batch, vocab_size)
    target (batch,)
    criterion (nn.KLDIVLoss)
    V (vocab_size, k)
    D (vocab_size, k)
    """
    # (batch, k) index in vocab_size dimension
    # k-nearest neighbors for target
    indices = torch.index_select(V, 0, target)
    # (batch, k) gather along vocab_size dimension
    outputk = torch.gather(output, 1, indices)
    # (batch, k) index in vocab_size dimension
    targetk = torch.index_select(D, 0, target)
    # KLDIVcriterion
    criterion = nn.KLDivLoss(reduction='sum').to(loss_cuda)
    return criterion(outputk, targetk)


def dist2weight(D, dist_decay_speed=0.8):
    '''
    D is a matrix recording distances between each vocab and its k nearest vocabs
    D(k, vocab_size)
    weight: \frac{\exp{-|dis|*scale}}{\sum{\exp{-|dis|*scale}}}
    Divide 100
    '''
    D = D.div(100)
    D = torch.exp(-D * dist_decay_speed)
    s = D.sum(dim=1, keepdim=True)
    D = D / s
    # The PAD should not contribute to the decoding loss
    D[constants.PAD, :] = 0.0
    return D


def load_dis_matrix(args):
    assert os.path.isfile(args.knearestvocabs),\
        "{} does not exist".format(args.knearestvocabs)
    with h5py.File(args.knearestvocabs, 'r') as f:
        V, D = f["V"], f["D"]
        V, D = torch.LongTensor(V), torch.FloatTensor(D)
    D = dist2weight(D, args.dist_decay_speed)
    return V, D


def clusterloss(q, p, loss_cuda):
    '''
    caculate the KL loss for clustering
    '''
    q, p = q.to(loss_cuda), p.to(loss_cuda)
    criterion = nn.KLDivLoss(reduction='sum').to(loss_cuda)
    return criterion(q.log(), p)


def reconstructionLoss(gendata,
                       autoencoder,
                       rclayer,
                       lossF,
                       args,
                       cuda0,
                       cuda1,
                       loss_cuda):
    """
    One batch reconstruction loss
    cuda0 for autoencoder
    cuda1 for rclayer
    loss_cuda for reconstruction loss

    Input:
    gendata: a named tuple contains
        gendata.src (seq_len1, batch): input tensor
        gendata.lengths (1, batch): lengths of source sequences
        gendata.trg (seq_len2, batch): target tensor.
    autoencoder: map input to output.
        log transform.
    lossF: loss function.
    ---
    Output:
    loss
    context (cuda0)
    """
    input, lengths, target = gendata.src, gendata.lengths, gendata.trg
    input = input.to(cuda0)
    lengths = lengths.to(cuda0)
    target = target.to(cuda0)

    # print("input size:", input.size())
    # print("lengths size:", lengths.size())
    # print("target size:", target.size())
    # Encoder & decoder
    # output (trg_seq_len, batch, hidden_size)
    # context (batch, hidden_size * num_directions)
    output, context = autoencoder(input, lengths, target)

    batch = output.size(1)
    loss = 0
    # we want to decode target in range [BOS+1:EOS]
    target = target[1:]
    # generate words from autoencoder output
    for o, t in zip(output.split(args.gen_batch),
                    target.split(args.gen_batch)):
        # (seq_len, gen_batch, hidden_size) =>
        ## (seq_len*gen_batch, hidden_size)
        o = o.view(-1, o.size(2)).to(cuda1)
        # print("o size:", o.size())
        o = rclayer(o)
        # (seq_len*gen_batch,)
        t = t.view(-1)
        o, t = o.to(loss_cuda), t.to(loss_cuda)
        loss += lossF(o, t)

    return loss.div(batch), context


def clusteringLoss(clusterlayer, context, p, cuda2, loss_cuda):
    """
    One batch cluster KL loss

    Input:
    context: (batch, hidden_size * num_directions) last hidden layer from encoder 
    clusterlayer: caculate Student’s t-distribution with clustering center

    p: (batch_size,n_clusters)target distribution

    Output:loss
    """
    batch = context.size(0)
    assert batch == p.size(0)
    q = clusterlayer(context.to(cuda2))
    kl_loss = clusterloss(q, p, loss_cuda)

    return kl_loss.div(batch)


def triLoss(a, p, n, autoencoder, loss_cuda):
    """
    a (named tuple): anchor data
    p (named tuple): positive data
    n (named tuple): negative data
    """
    a_src, a_lengths, a_invp = a.src, a.lengths, a.invp
    p_src, p_lengths, p_invp = p.src, p.lengths, p.invp
    n_src, n_lengths, n_invp = n.src, n.lengths, n.invp

    a_src, a_lengths, a_invp = a_src.to(
        loss_cuda), a_lengths.to(loss_cuda), a_invp.to(loss_cuda)
    p_src, p_lengths, p_invp = p_src.to(
        loss_cuda), p_lengths.to(loss_cuda), p_invp.to(loss_cuda)
    n_src, n_lengths, n_invp = n_src.to(
        loss_cuda), n_lengths.to(loss_cuda), n_invp.to(loss_cuda)

    a_context = autoencoder.encoder_hn(a_src, a_lengths)
    p_context = autoencoder.encoder_hn(p_src, p_lengths)
    n_context = autoencoder.encoder_hn(n_src, n_lengths)

    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2).to(loss_cuda)

    return triplet_loss(a_context[a_invp], p_context[p_invp], n_context[n_invp])
