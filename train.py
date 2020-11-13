import losses
import constants

from data_utils import DataOrderScaner, load_label
from cluster import update_cluster
from metrics import nmi_score, ami_score, ari_score, fms_score, cluster_acc, cluster_purity
from models import DTC

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
import os
import shutil


def save_checkpoint(state, args):
    torch.save(state, args.checkpoint)
    shutil.copyfile(args.checkpoint, os.path.join(args.model, 'best_model.pt'))


def init_parameters(model):
    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    devices = [torch.device("cuda:" + str(i)) for i in range(4)]
    for i in range(len(devices)):
        devices[i] = devices[args.cuda]
    loss_cuda = devices[args.cuda]

    # define criterion, model, optimizer
    dtc = DTC(args, devices)
    init_parameters(dtc)
    dtc.pretrain()

    optimizer = torch.optim.Adam(
        dtc.parameters(), lr=args.learning_rate)

    V, D = losses.load_dis_matrix(args)
    V, D = V.to(loss_cuda), D.to(loss_cuda)

    def rclossF(o, t):
        return losses.KLDIVloss(o, t, V, D, loss_cuda)

    start_epoch = 0
    iteration = 0
    # load model state and optmizer state
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        iteration = checkpoint["iteration"] + 1
        dtc.load_state_dict(checkpoint["dtc"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        print("=> No checkpoint found at '{}'".format(args.checkpoint))

    # Training
    print("=> Reading trajecoty data...")
    scaner = DataOrderScaner(args.src_file, args.batch)
    scaner.load()  # load trg data
    y = load_label(args.label_file)
    y_pred_last = np.zeros_like(y)

    print("=> Epoch starts at {} "
          "and will end at {}".format(start_epoch, args.epoch-1))

    best_loss = [-1, -1, -1, 0]

    for epoch in range(start_epoch, args.epoch):

        # update target distribution p
        if epoch % args.update_interval == 0:
            with torch.no_grad():
                # q (datasize,n_clusters)
                vecs, tmp_q, p = update_cluster(
                    dtc, args, devices[0], devices[2])

            # evaluate clustering performance
            y_pred = tmp_q.numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred

            acc = cluster_acc(y, y_pred)
            nmi = nmi_score(y, y_pred)
            ari = ari_score(y, y_pred)

            if best_loss[0] < acc:
                best_loss[0] = acc
                best_loss[1] = nmi
                best_loss[2] = ari
                best_loss[3] = epoch
            else:
                if epoch - best_loss[3] > 5:
                    break

            if epoch > 0 and delta_label < args.tolerance:
                print('Delta_label {:.4f} < tolerance {:.4f}'.format(
                    delta_label, args.tolerance))
                print('=> Reached tolerance threshold. Stopping training.')
                break
            else:
                print('Epoch {0}\tAcc: {1:.4f}\tnmi: {2:.4f}\tari: {3:.4f}'.format(
                    epoch, acc, nmi, ari))

        scaner.reload()
        while True:
            optimizer.zero_grad()
            gendata = scaner.getbatch(invp=False)
            if gendata is None:
                break

            reconstr_loss, context = losses.reconstructionLoss(
                gendata, dtc.autoencoder, dtc.rclayer, rclossF, args, devices[0], devices[1], loss_cuda)

            # (batch_size,n_clusters)
            p_select = p[scaner.start - args.batch:scaner.start]
            kl_loss = losses.clusteringLoss(
                dtc.clusterlayer, context, p_select, devices[2], loss_cuda)

            anchor, positive, negative = scaner.getbatch_discriminative()
            tri_loss = losses.triLoss(
                anchor, positive, negative, dtc.autoencoder, loss_cuda)

            loss = reconstr_loss + args.gamma * kl_loss + args.beta * tri_loss

            # compute the gradients
            loss.backward()
            # clip the gradients
            clip_grad_norm_(dtc.parameters(), args.max_grad_norm)
            # one step optimization
            optimizer.step()

            # average loss for one word
            if iteration % args.print_freq == 0:
                print("Iteration: {0:}\tLoss: {1:.3f}\t"
                      "Rc Loss: {2:.3f}\tKL Loss: {3:.3f}\tTriplet Loss: {4:.4f}"
                      .format(iteration, loss, reconstr_loss, kl_loss, tri_loss))

            if iteration % args.save_freq == 0 and iteration > 0:
                # print("Saving the model at iteration {}  loss {}"
                #       .format(iteration, loss))
                save_checkpoint({
                    "iteration": iteration,
                    "best_loss": loss,
                    "dtc": dtc.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }, args)

            iteration += 1

    print("=================")
    print('Best Epoch {0}\tAcc: {1:.4f}\tnmi: {2:.4f}\tari: {3:.4f}'.format(
        best_loss[3], best_loss[0], best_loss[1], best_loss[2]))
