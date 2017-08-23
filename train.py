# -*- coding: utf-8 -*-
#!/usr/bin/env python
#python train.py
#python train.py -g 0
#ハマるエラー３種
#shape
#ndim
#dtype
#https://docs.chainer.org/en/stable/tutorial/type_check.html
from __future__ import print_function
import argparse
import os
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import training
from chainer.training import extensions
from chainer import serializers
from chainer.utils import force_array
from chainer import optimizers, cuda, serializers
from chainer import Variable
from models import EncoderDecoder, Discriminator
from data import get_training_set, get_test_set

parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
parser.add_argument('--batchsize', '-b', type=int, default=1, help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=200, help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--dataset', '-i', default='facades', help='Directory of image files.')
parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--snapshot_interval', type=int, default=1000, help='Interval of snapshot')
parser.add_argument('--display_interval', type=int, default=100, help='Interval of displaying log to console')

parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
args = parser.parse_args()

print('GPU: {}'.format(args.gpu))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))

print('===> Loading datasets')
root_path = "dataset/"
train_set = get_training_set(root_path + args.dataset)
test_set = get_test_set(root_path + args.dataset)

# for iteration, batch in enumerate(train_set, 1):
#     print("iteration", iteration)
#     print(batch[0].shape)
#     print(batch[1].shape)
#     break

print('===> Building model')
encoderdecoder_model = EncoderDecoder(args.input_nc, args.output_nc, args.ngf)
discriminator_model = Discriminator(args.input_nc, args.output_nc, args.ngf)

if args.gpu >= 0:
    print("use gpu")
    chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
    encoderdecoder_model.to_gpu()
    discriminator_model.to_gpu()

optimizer_encoderdecoder = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_encoderdecoder.setup(encoderdecoder_model)
optimizer_discriminator = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_discriminator.setup(discriminator_model)

if args.gpu >= 0:
    xp = cuda.cupy
    label = xp.random.randn(args.batchsize)
    real_label = Variable(xp.ones((1,1,30,30), dtype=xp.float32))
    fake_label = Variable(xp.zeros((1,1,30,30), dtype=xp.float32))
else:
    label = np.random.randn(args.batchsize)
    real_label = Variable(np.ones((1,1,30,30), dtype=np.float32))
    fake_label = Variable(np.zeros((1,1,30,30), dtype=np.float32))

def train(epoch):
    for iteration, batch in enumerate(train_set, 1):
        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################
        # train with real
        discriminator_model.zerograds()
        
        if args.gpu >= 0:
            real_A, real_B = xp.asarray(batch[0], dtype=xp.float32) / 255.0, xp.asarray(batch[1], dtype=xp.float32) / 255.0
        else:
            real_A, real_B = np.asarray(batch[0], dtype=np.float32) / 255.0, np.asarray(batch[1], dtype=np.float32) / 255.0
        real_A = real_A.transpose(2, 0, 1)
        real_B = real_B.transpose(2, 0, 1)
        real_A = real_A.reshape(1,3,256,256)
        real_B = real_B.reshape(1,3,256,256)

        real_A = Variable(real_A)
        real_B = Variable(real_B)
        output = discriminator_model(F.concat((real_A, real_B), axis=1))

        label = (real_label)
        err_d_real = loss_dis(output, label)
        err_d_real.backward()
        fake_b = encoderdecoder_model(real_A)
        output = discriminator_model(F.concat((real_A, fake_b), axis=1))
        label = (fake_label)
        err_d_fake = loss_dis(output, label)
        err_d_fake.backward()
        err_d = (err_d_real + err_d_fake) / 2.0
        optimizer_discriminator.update()
        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ###########################
        output = discriminator_model(F.concat((real_A, fake_b), axis=1))
        label = (real_label)
        err_g = loss_criterion(output, label) + args.lamb * loss_criterion_l1(fake_b, real_B)
        err_g.backward()
        optimizer_encoderdecoder.update()

        print("===> Epoch[{}]({}/{}): Loss_D: {} Loss_G: {} ".format(
            epoch, iteration, len(train_set), err_d.data, err_g.data))

        if args.snapshot_interval % epoch == 0:
            serializers.save_npz("encoderdecoder_model_"+epoch, encoderdecoder_model)
            serializers.save_npz("discriminator_model_"+epoch, discriminator_model)

def loss_criterion(output, label, lam1=100, lam2=1):
    loss = lam1*(F.mean_absolute_error(output, label))
    return loss

def loss_criterion_l1(y_out, t_out, lam1=100, lam2=1):
    batchsize,_,w,h = list(y_out.data.shape)
    loss_rec = lam1*(F.mean_absolute_error(y_out, t_out))
    loss_adv = lam2*F.sum(F.softplus(-y_out)) / batchsize / w / h
    loss = loss_rec + loss_adv
    return loss

def loss_dis(y_in, y_out):
    batchsize,_,w,h = y_in.data.shape    
    L1 = F.sum(F.softplus(-y_in)) / batchsize / w / h
    L2 = F.sum(F.softplus(y_out)) / batchsize / w / h
    loss = L1 + L2
    return loss

for epoch in range(1, args.epoch + 1):
    train(epoch)
    # test()
    # if epoch % 50 == 0:
    #     checkpoint(epoch)




