import argparse
from train import train
import torch

root_path = "."

parser = argparse.ArgumentParser(description="e2dtc.py")

parser.add_argument("-data", default=root_path + "/data",
                    help="Path to training and validating data")

parser.add_argument("-src_file", default=root_path + "/data/trj_vocab.h5",
                    help="source trajectory file to cluster")

parser.add_argument("-label_file", default=root_path + "/data/labels.pkl",
                    help="cluster label")

parser.add_argument("-model", default=root_path + "/model",
                    help="Path to store model state")

parser.add_argument("-pretrain_checkpoint", default=root_path +
                    "/model/pretrain_checkpoint.pt", help="The saved pretrain checkpoint")

parser.add_argument("-cluster_center", default=root_path +
                    "/model/cluster_center.pt", help="The saved cluster center checkpoint")

parser.add_argument("-checkpoint", default=root_path + "/model/checkpoint.pt",
                    help="The saved checkpoint")

parser.add_argument("-n_clusters", type=int, default=20,
                    help="Number of luster")

parser.add_argument("-num_layers", type=int, default=4,
                    help="Number of layers in the RNN cell")

parser.add_argument("-bidirectional", type=bool, default=True,
                    help="True if use bidirectional rnn in encoder")

parser.add_argument("-hidden_size", type=int, default=256,
                    help="The hidden state size in the RNN cell")

parser.add_argument("-embedding_size", type=int, default=256,
                    help="The word (cell) embedding size")

parser.add_argument("-dropout", type=float, default=0.2,
                    help="The dropout probability")

parser.add_argument("-max_grad_norm", type=float, default=5.0,
                    help="The maximum gradient norm")

parser.add_argument("-learning_rate", type=float, default=0.0001)

parser.add_argument("-gamma", type=float, default=0.1,
                    help="'coefficient of clustering loss')")

parser.add_argument("-beta", type=float, default=0.1,
                    help="coefficient of triplet loss")

parser.add_argument("-cuda", type=int, default=3,
                    help="which GPU is in use")

parser.add_argument("-gen_batch", type=int, default=32,
                    help="The generate representation batch size")

parser.add_argument("-batch", type=int, default=256,
                    help="The batch size")

parser.add_argument("-pretrain_epoch", type=float, default=10,
                    help="The pretrain epoch")

parser.add_argument("-epoch", type=int, default=20,
                    help="The training epoch")

parser.add_argument('-tolerance', type=float, default=1e-04)

parser.add_argument('-update_interval', default=1, type=int,
                    help="The interval iteration to update clustering")

parser.add_argument("-print_freq", type=int, default=100,
                    help="Print frequency")

parser.add_argument("-save_freq", type=int, default=1000,
                    help="Save frequency")

parser.add_argument("-knearestvocabs", default=None,
                    help="""The file of k nearest cells and distances used in KLDIVLoss,
    produced by preprocessing, necessary if KLDIVLoss is used""")

parser.add_argument("-dist_decay_speed", type=float, default=0.8,
                    help="""How fast the distance decays in dist2weight, a small value will
    give high weights for cells far away""")

parser.add_argument("-max_length", default=200,
                    help="The maximum length of the target sequence")

parser.add_argument("-vocab_size", type=int, default=0,
                    help="Vocabulary Size")

if __name__ == "__main__":
    args = parser.parse_args()
    for k, v in args._get_kwargs():
        print("{0} =  {1}".format(k, v))

    print("-"*7 + " start training " + "-"*7)
    train(args)
