
import numpy as np
import torch
import constants
import pickle
import random
from collections import namedtuple


def argsort(seq):
    """
    sort by length in reverse order
    ---
    seq (list[array[int32]])
    """
    return [x for x, y in sorted(enumerate(seq),
                                 key=lambda x: len(x[1]),
                                 reverse=True)]


def random_subseq(a, rate):
    """
    Dropping some points between a[3:-2] randomly according to rate.

    Input:
    a (array[int])
    rate (float)
    """
    idx = np.random.rand(len(a)) < rate
    idx[0], idx[-1] = True, True
    return a[idx]


def pad_array(a, max_length, PAD=constants.PAD):
    """
    a (array[int32]), padding with max_length
    """
    return np.concatenate((a, [PAD]*(max_length - len(a))))


def pad_arrays(a):
    '''
    padding each a[i] with max_length, return LongTensor
    '''
    max_length = max(map(len, a))
    a = [pad_array(a[i], max_length) for i in range(len(a))]
    a = np.stack(a).astype(np.int)
    return torch.LongTensor(a)


def pad_arrays_pair(src, trg, keep_invp=False):
    """
    ---
    Input:
    src (list[array[int32]]) (batch,variant_len)
    trg (list[array[int32]]) (batch,variant_len)
    ---
    Output:
    src (seq_len1, batch) with decending length
    trg (seq_len2, batch) with decending length
    lengths (1, batch)
    invp (batch,): inverse permutation, src.t()[invp] gets original order
    e.g. if src length is [4,1,3,2,0], idx = [0,2,3,1,4], invp = [0,3,1,2,4], scr[invp] = [4,1,3,2,0]
    """
    TD = namedtuple('TD', ['src', 'lengths', 'trg', 'invp'])

    assert len(src) == len(
        trg), "source and target should have the same length"
    idx = argsort(src)
    src = list(np.array(src, dtype=object)[idx])
    trg = list(np.array(trg, dtype=object)[idx])

    lengths = list(map(len, src))
    lengths = torch.LongTensor(lengths)
    src = pad_arrays(src)
    trg = pad_arrays(trg)
    if keep_invp == True:
        invp = torch.LongTensor(invpermute(idx))
        # (batch, seq_len) => (seq_len, batch)
        return TD(src=src.t().contiguous(), lengths=lengths.view(1, -1), trg=trg.t().contiguous(), invp=invp)
    else:
        # (batch, seq_len) => (seq_len, batch)
        return TD(src=src.t().contiguous(), lengths=lengths.view(1, -1), trg=trg.t().contiguous(), invp=[])


def invpermute(p):
    """
    inverse permutation
    p:[1,4,0,2,3]
    invp:[2,0,3,4,1]
    """
    p = np.asarray(p)
    invp = np.empty_like(p)
    for i in range(p.size):
        invp[p[i]] = i
    return invp


def pad_arrays_keep_invp(src):
    """
    Pad arrays and return inverse permutation

    Input:
    src (list[array[int32]])
    ---
    Output:
    src (seq_len, batch)
    lengths (1, batch)
    invp (batch,): inverse permutation, src.t()[invp] gets original order
    """
    idx = argsort(src)
    src = list(np.array(src, dtype=object)[idx])
    lengths = list(map(len, src))
    lengths = torch.LongTensor(lengths)
    src = pad_arrays(src)
    invp = torch.LongTensor(invpermute(idx))
    return src.t().contiguous(), lengths.view(1, -1), invp


def load_label(labelpath):
    '''
    Load label data: numpy array (datasize,)
    '''
    f = open(labelpath, 'rb')
    y = pickle.load(f)
    return y


class DataLoader():
    """
    srcfile: source file name(with noise)
    trgfile: target file name(original trajectory)
    mtafile: meta file name(the centroid offset of the trip)
    batch: batch size
    validate: if validate = True return batch orderly otherwise return
        batch randomly
    """

    def __init__(self, srcfile, trgfile, mtafile, batch, validate=False):
        self.srcfile = srcfile
        self.trgfile = trgfile
        self.mtafile = mtafile

        self.batch = batch
        self.validate = validate

    def load(self):
        '''
        load src/target/meta trajectory
        '''
        self.start = 0

        self.srcdata = []
        self.trgdata = []
        self.mtadata = []

        srcstream, trgstream, mtastream = open(self.srcfile, 'r'), open(
            self.trgfile, 'r'), open(self.mtafile, 'r')
        num_line = 0

        for (s, t, m) in zip(srcstream, trgstream, mtastream):
            s = [int(x) for x in s.split()]
            t = [constants.BOS] + [int(x) for x in t.split()] + [constants.EOS]
            m = [float(x) for x in m.split()]

            self.srcdata.append(np.array(s, dtype=np.int32))
            self.trgdata.append(np.array(t, dtype=np.int32))
            self.mtadata.append(np.array(m, dtype=np.float32))

            num_line += 1

        self.srcdata = np.array(self.srcdata, dtype=object)
        self.trgdata = np.array(self.trgdata, dtype=object)
        self.mtadata = np.array(self.mtadata, dtype=object)

        self.size = num_line

        srcstream.close(), trgstream.close(), mtastream.close()
        # print("=> Loaded data size: ", num_line)

    def getbatch_one(self):
        '''
        Get one batch size data
        If training, randomly select, otherwise orderly
        '''
        if self.validate == True:
            src = self.srcdata[self.start:self.start+self.batch]
            trg = self.trgdata[self.start:self.start+self.batch]
            mta = self.mtadata[self.start:self.start+self.batch]

            # update `start` for next batch
            self.start += self.batch
            if self.start >= self.size:
                self.start = 0
            return list(src), list(trg), list(mta)
        else:
            # random select from training datasets
            idx = np.random.choice(self.size, self.batch)
            src = self.srcdata[idx]
            trg = self.trgdata[idx]
            mta = self.mtadata[idx]
            return list(src), list(trg), list(mta)

    def getbatch_generative(self):
        '''
        get batch src / trg data with padding length
        '''
        src, trg, _ = self.getbatch_one()
        # src (seq_len1, batch), lengths (1, batch), trg (seq_len2, batch)
        return pad_arrays_pair(src, trg, keep_invp=False)


class DataOrderScaner():
    def __init__(self, srcfile, batch):
        self.srcfile = srcfile
        self.batch = batch
        self.srcdata = []
        self.trgdata = []
        self.start = 0

    def load(self):
        num_line = 0
        with open(self.srcfile, 'r') as f:
            srcstream = f.readlines()
            for s in srcstream:
                s = [int(x) for x in s.strip().split()]
                t = [constants.BOS] + s + [constants.EOS]
                self.srcdata.append(np.array(s))
                self.trgdata.append(np.array(t))
                num_line += 1
        self.size = num_line
        self.start = 0

    def reload(self):
        self.start = 0

    def getbatch(self, invp=True):
        """
        get batch src / trg data(the same) with padding length
        ---
        Output:
        src (seq_len1, batch) with decending length
        trg (seq_len2, batch) with decending length
        lengths (1, batch)
        invp (batch,): inverse permutation, src.t()[invp] gets original order
        if [start:batch] > left, return left data
        """
        if self.start >= self.size:
            return None
        src = self.srcdata[self.start:self.start+self.batch]
        trg = self.trgdata[self.start:self.start+self.batch]
        # update `start` for next batch
        self.start += self.batch
        return pad_arrays_pair(src, trg, invp)

    def get_random_batch(self):
        return random.sample(self.srcdata, self.batch)

    def getbatch_discriminative(self):
        '''
        Get batch for anchor / negative data
        Positive data get from anchor dropping
        '''
        a_src = self.get_random_batch()
        n_src = self.get_random_batch()

        p_src = []

        for i in range(len(a_src)):
            a = a_src[i]
            if len(a) < 10:
                p_src.append(a)
            else:
                a1, a3, a5 = 0, len(a)//2, len(a)
                a2, a4 = (a1 + a3)//2, (a3 + a5)//2
                rate = np.random.choice([0.5, 0.6, 0.7])
                if np.random.rand() > 0.5:
                    p_src.append(random_subseq(a[a2:a5], rate))
                else:
                    p_src.append(random_subseq(a[a1:a4], rate))

        a = pad_arrays_pair(a_src, a_src, keep_invp=True)
        p = pad_arrays_pair(p_src, p_src, keep_invp=True)
        n = pad_arrays_pair(n_src, n_src, keep_invp=True)
        return a, p, n
