import os
import torch
import time
from sklearn.cluster import KMeans
from data_utils import DataOrderScaner


def init_cluster(model, args, cuda0, cuda2):
    autoencoder, clusterlayer = model
    # load init cluster tensor
    if os.path.isfile(args.cluster_center):
        print("=> Loading cluster center checkpoint '{}'".format(
            args.cluster_center))
        cluster_center = torch.load(args.cluster_center)
        clusters = cluster_center["clusters"]
        n = cluster_center["n_clusters"]
        # load data
        clusterlayer.clusters.data = clusters.to(cuda2)
        return

    autoencoder.eval()
    clusterlayer.eval()
    vecs = []

    print("=> Loading all target data to init clusters...")
    scaner = DataOrderScaner(args.src_file, args.batch)
    scaner.load()  # load trg data

    print("=> Generate representation for all trajectory...")
    while True:
        trjdata = scaner.getbatch()
        if trjdata is None:
            break
        src, lengths, invp = trjdata.src, trjdata.lengths, trjdata.invp
        src, lengths = src.to(cuda0), lengths.to(cuda0)
        # (batch, hidden_size * num_directions)
        context = autoencoder.encoder_hn(src, lengths)
        context = context[invp]
        vecs.append(context.cpu().data)
    vecs = torch.cat(vecs)

    print("=> init cluster center with KMeans...")
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=100,
                    random_state=58).fit(vecs.numpy())
    clusterlayer.clusters.data = torch.Tensor(
        kmeans.cluster_centers_).to(cuda2)

    autoencoder.train()
    clusterlayer.train()

    torch.save({
        "clusters": clusterlayer.clusters.data.cpu(),
        "n_clusters": args.n_clusters
    }, args.cluster_center)
    print("-" * 7 + "Initiated cluster center" + "-" * 7)


def target_distribution(q):
    # clustering target distribution for self-training
    # q (batch,n_clusters): similarity between embedded point and cluster center
    # p (batch,n_clusters): target distribution
    weight = q**2 / q.sum(0)
    p = (weight.t() / weight.sum(1)).t()
    return p


def update_cluster(dtc, args, cuda0, cuda2):
    q = []
    vecs = []
    dtc.eval()
    autoencoder, clusterlayer = dtc.autoencoder, dtc.clusterlayer

    scaner = DataOrderScaner(args.src_file, args.batch)
    scaner.load()  # load trg data
    while True:
        trjdata = scaner.getbatch()
        if trjdata is None:
            break
        src, lengths, invp = trjdata.src, trjdata.lengths, trjdata.invp
        src, lengths = src.to(cuda0), lengths.to(cuda0)

        # get trajectory represention
        # (batch, hidden_size * num_directions)
        context = autoencoder.encoder_hn(src, lengths)
        context = context[invp]

        # q_i (batch,n_clusters)
        q_i = clusterlayer(context)
        q.append(q_i.cpu().data)
        vecs.append(context.cpu().data)

    # (datasize,n_clusters)
    q = torch.cat(q)
    vecs = torch.cat(vecs)

    dtc.train()
    return vecs, q, target_distribution(q)
