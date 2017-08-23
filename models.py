#!/usr/bin/env python
from __future__ import print_function
import numpy
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L

# For input size input_nc x 256 x 256
class EncoderDecoder(chainer.Chain):
    def __init__(self, input_nc, output_nc, ngf):
        super(EncoderDecoder, self).__init__()
        #input_nc=3, output_nc=3
        #ngf=64
        #Convolution2D(self, in_channels, out_channels, ksize=None, stride=1, pad=0, nobias=False, initialW=None, initial_bias=None)
        self.conv1 = L.Convolution2D(input_nc, ngf, 4, 2, 1)
        self.conv2 = L.Convolution2D(ngf, ngf * 2, 4, 2, 1)
        self.conv3 = L.Convolution2D(ngf * 2, ngf * 4, 4, 2, 1)
        self.conv4 = L.Convolution2D(ngf * 4, ngf * 8, 4, 2, 1)
        self.conv5 = L.Convolution2D(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv6 = L.Convolution2D(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv7 = L.Convolution2D(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv8 = L.Convolution2D(ngf * 8, ngf * 8, 4, 2, 1)
        self.batch_norm = L.BatchNormalization(ngf)
        self.batch_norm2 = L.BatchNormalization(ngf * 2)
        self.batch_norm4 = L.BatchNormalization(ngf * 4)
        self.batch_norm8 = L.BatchNormalization(ngf * 8)
        self.dconv1 = L.Deconvolution2D(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv2 = L.Deconvolution2D(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv3 = L.Deconvolution2D(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv4 = L.Deconvolution2D(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv5 = L.Deconvolution2D(ngf * 8 * 2, ngf * 4, 4, 2, 1)
        self.dconv6 = L.Deconvolution2D(ngf * 4 * 2, ngf * 2, 4, 2, 1)
        self.dconv7 = L.Deconvolution2D(ngf * 2 * 2, ngf, 4, 2, 1)
        self.dconv8 = L.Deconvolution2D(ngf * 2, output_nc, 4, 2, 1)

    def __call__(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 256 x 256
        e1 = self.conv1(input)
        # state size is (ngf) x 128 x 128
        e2 = self.batch_norm2(self.conv2(F.leaky_relu(e1)))
        # state size is (ngf x 2) x 64 x 64
        e3 = self.batch_norm4(self.conv3(F.leaky_relu(e2)))
        # state size is (ngf x 4) x 32 x 32
        e4 = self.batch_norm8(self.conv4(F.leaky_relu(e3)))
        # state size is (ngf x 8) x 16 x 16
        e5 = self.batch_norm8(self.conv5(F.leaky_relu(e4)))
        # state size is (ngf x 8) x 8 x 8
        e6 = self.batch_norm8(self.conv6(F.leaky_relu(e5)))
        # state size is (ngf x 8) x 4 x 4
        e7 = self.batch_norm8(self.conv7(F.leaky_relu(e6)))
        # state size is (ngf x 8) x 2 x 2
        # No batch norm on output of Encoder
        e8 = self.conv8(F.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (ngf x 8) x 1 x 1
        d1_ = F.dropout(self.batch_norm8(self.dconv1(F.relu(e8))))
        # state size is (ngf x 8) x 2 x 2
        d1 = F.concat((d1_, e7), axis=1)
        d2_ = F.dropout(self.batch_norm8(self.dconv2(F.relu(d1))))
        # state size is (ngf x 8) x 4 x 4
        d2 = F.concat((d2_, e6), axis=1)
        d3_ = F.dropout(self.batch_norm8(self.dconv3(F.relu(d2))))
        # state size is (ngf x 8) x 8 x 8
        d3 = F.concat((d3_, e5), axis=1)
        d4_ = self.batch_norm8(self.dconv4(F.relu(d3)))
        # state size is (ngf x 8) x 16 x 16
        d4 = F.concat((d4_, e4), axis=1)
        d5_ = self.batch_norm4(self.dconv5(F.relu(d4)))
        # state size is (ngf x 4) x 32 x 32
        d5 = F.concat((d5_, e3), axis=1)
        d6_ = self.batch_norm2(self.dconv6(F.relu(d5)))
        # state size is (ngf x 2) x 64 x 64
        d6 = F.concat((d6_, e2), axis=1)
        d7_ = self.batch_norm(self.dconv7(F.relu(d6)))
        # state size is (ngf) x 128 x 128
        d7 = F.concat((d7_, e1), axis=1)
        d8 = self.dconv8(F.relu(d7))
        # state size is (nc) x 256 x 256
        output = F.tanh(d8)
        return output


class Discriminator(chainer.Chain):
    def __init__(self, input_nc, output_nc, ngf):
        super(Discriminator, self).__init__()
        #input_nc=3, output_nc=3
        #ngf=64
        #Convolution2D(self, in_channels, out_channels, ksize=None, stride=1, pad=0, nobias=False, initialW=None, initial_bias=None)
        self.disconv1 = L.Convolution2D(input_nc + output_nc, ngf, 4, 2, 1)
        self.disconv2 = L.Convolution2D(ngf, ngf * 2, 4, 2, 1)
        self.disconv3 = L.Convolution2D(ngf * 2, ngf * 4, 4, 2, 1)
        self.disconv4 = L.Convolution2D(ngf * 4, ngf * 8, 4, 1, 1)
        self.disconv5 = L.Convolution2D(ngf * 8, 1, 4, 1, 1)
        self.batch_norm2 = L.BatchNormalization(ngf * 2)
        self.batch_norm4 = L.BatchNormalization(ngf * 4)
        self.batch_norm8 = L.BatchNormalization(ngf * 8)

    def __call__(self, input):
        #input_nc=3, output_nc=3
        #ngf=64
        #Convolution2D(self, in_channels, out_channels, ksize=None, stride=1, pad=0, nobias=False, initialW=None, initial_bias=None)

        # input is (nc x 2) x 256 x 256
        h1 = self.disconv1(input)
        # state size is (ndf) x 128 x 128
        h2 = self.batch_norm2(self.disconv2(F.leaky_relu(h1)))
        # state size is (ndf x 2) x 64 x 64
        h3 = self.batch_norm4(self.disconv3(F.leaky_relu(h2)))
        # state size is (ndf x 4) x 32 x 32
        h4 = self.batch_norm8(self.disconv4(F.leaky_relu(h3)))
        # state size is (ndf x 8) x 31 x 31
        h5 = self.disconv5(F.leaky_relu(h4))
        # state size is (ndf) x 30 x 30, corresponds to 70 x 70 receptive
        output = F.sigmoid(h5)
        return output




