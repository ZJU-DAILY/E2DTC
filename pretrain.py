import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from data_utils import DataLoader
import losses
import constants
import time
import os
import h5py


def init_parameters(model):
    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)


def save_pretrain_checkpoint(state, args):
    torch.save(state, args.pretrain_checkpoint)


def validate(valData, model, lossF, args, cuda0, cuda1, loss_cuda):
    """
    ValData (DataLoader)
    """
    autoencoder, rclayer = model
    # switch to evaluation mode
    autoencoder.eval()
    rclayer.eval()

    num_iteration = valData.size // args.batch
    if valData.size % args.batch > 0:
        num_iteration += 1

    total_genloss = 0
    for iteration in range(num_iteration):
        gendata = valData.getbatch_generative()
        with torch.no_grad():
            loss, _ = losses.reconstructionLoss(
                gendata, autoencoder, rclayer, lossF, args, cuda0, cuda1, loss_cuda)
            # gendata.trg.size(1) is the batch size
            total_genloss += loss.item() * gendata.trg.size(1)
    # switch back to training mode
    autoencoder.train()
    rclayer.train()
    return total_genloss / valData.size


def pretrain_ae(model, args, cuda0, cuda1, loss_cuda):
    '''
    Pretrain autoencoder
    cuda0 for autoencoder
    cuda1 for relayer 
    loss_cuda for reconstruction loss
    '''
    # define criterion, model, optimizer
    autoencoder, rclayer = model

    ae_optimizer = torch.optim.Adam(
        autoencoder.parameters(), lr=args.learning_rate)
    rc_optimizer = torch.optim.Adam(
        rclayer.parameters(), lr=args.learning_rate)

    V, D = losses.load_dis_matrix(args)
    V, D = V.to(loss_cuda), D.to(loss_cuda)

    def rclossF(o, t):
        return losses.KLDIVloss(o, t, V, D, loss_cuda)

    start_iteration = 0
    is_pretrain = False
    # load model state and optmizer state
    if os.path.isfile(args.pretrain_checkpoint):
        print("=> loading pretrain checkpoint '{}'".format(
            args.pretrain_checkpoint))
        pretrain_checkpoint = torch.load(args.pretrain_checkpoint)
        start_iteration = pretrain_checkpoint["iteration"] + 1
        is_pretrain = pretrain_checkpoint["pretrain"]
        autoencoder.load_state_dict(pretrain_checkpoint["autoencoder"])
        rclayer.load_state_dict(pretrain_checkpoint["rclayer"])
        ae_optimizer.load_state_dict(pretrain_checkpoint["ae_optimizer"])
        rc_optimizer.load_state_dict(pretrain_checkpoint["rc_optimizer"])
    else:
        print("=> No checkpoint found at '{}'".format(
            args.pretrain_checkpoint))

    if is_pretrain:
        print("-"*7 + " Loaded pretrain model" + "-"*7)
        return

    # load data for pretrain
    print("=> Reading training data...")
    trainsrc = os.path.join(args.data, "train.src")
    traintrg = os.path.join(args.data, "train.trg")
    trainmta = os.path.join(args.data, "train.mta")
    trainData = DataLoader(trainsrc, traintrg, trainmta,
                           args.batch)
    trainData.load()
    print("Loaded data,training data size ", trainData.size)

    valsrc = os.path.join(args.data, "val.src")
    valtrg = os.path.join(args.data, "val.trg")
    valmta = os.path.join(args.data, "val.mta")
    validation = True
    if os.path.isfile(valsrc) and os.path.isfile(valtrg):
        valData = DataLoader(valsrc, valtrg, valmta,
                             args.batch, True)
        valData.load()
        assert valData.size > 0, "Validation data size must be greater than 0"
        print("=> Loaded validation data size {}".format(valData.size))
    else:
        print("No validation data found, training without validating...")
        validation = False

    num_iteration = int(trainData.size / args.batch * args.pretrain_epoch)
    best_prec_loss = float('inf')

    print("=> Iteration starts at {} "
          "and will end at {}".format(start_iteration, num_iteration-1))
    # training
    for iteration in range(start_iteration, num_iteration):
        ae_optimizer.zero_grad()
        rc_optimizer.zero_grad()
        # reconstruction loss
        gendata = trainData.getbatch_generative()
        loss, _ = losses.reconstructionLoss(
            gendata, autoencoder, rclayer, rclossF, args, cuda0, cuda1, loss_cuda)

        # compute the gradients
        loss.backward()
        # clip the gradients
        clip_grad_norm_(autoencoder.parameters(), args.max_grad_norm)
        clip_grad_norm_(rclayer.parameters(),
                        args.max_grad_norm)
        # one step optimization
        ae_optimizer.step()
        rc_optimizer.step()
        if iteration % args.print_freq == 0:
            print("Iteration: {0:}\tReconstruction genLoss: {1:.3f}\t".format(
                iteration, loss))

        if (iteration % args.save_freq == 0 or iteration == num_iteration - 1) and validation:
            prec_loss = validate(
                valData, (autoencoder, rclayer), rclossF, args, cuda0, cuda1, loss_cuda)
            if prec_loss < best_prec_loss:
                best_prec_loss = prec_loss
                print("Saving the model at iteration {} validation loss {}"
                      .format(iteration, prec_loss))
                save_pretrain_checkpoint({
                    "iteration": iteration,
                    "autoencoder": autoencoder.state_dict(),
                    "rclayer": rclayer.state_dict(),
                    "ae_optimizer": ae_optimizer.state_dict(),
                    "rc_optimizer": rc_optimizer.state_dict(),
                    "pretrain": False
                }, args)
    print("-"*7 + " Pretrain model finished " + "-"*7)
