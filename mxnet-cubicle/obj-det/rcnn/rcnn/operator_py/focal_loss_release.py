#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2017/12/15 @Northrend
#
# Reproduction Focal Loss Layer
# Paper Reference: 
# https://arxiv.org/abs/1708.02002
# Code Reference: 
# https://github.com/unsky/RetinaNet/blob/master/retinanet/operator_py/focal_loss.py
# 

from __future__ import print_function
import mxnet as mx
import numpy as np

class FocalLossOperator(mx.operator.CustomOp):
    def __init__(self, gamma, alpha, use_ignore, ignore_label, normalize=True):
        super(FocalLossOperator, self).__init__()
        self.gamma = gamma
        self.alpha = alpha 
        self.use_ignore = use_ignore
        self.ignore_label = ignore_label
        self.normalize = normalize
        self.eps = 1e-14
        # print('Focalloss args:\ngamma: {}\nalpha: {}\nuse_ignore: {}\nignore_label: {}\nnormalize: {}'.format(
        #                 self.gamma, self.alpha, self.use_ignore, self.ignore_label, self.normalize))

    def forward(self, is_train, req, in_data, out_data, aux):
        # pass softmax activation outputs forward
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        num_class = in_data[0].shape[1] # class number 
        p_t = in_data[0]    # probablity matrix 
        label = in_data[1]  # label

        # compute gradients
        a_t = (label > 0) * self.alpha + (label == 0) * (1 - self.alpha)  # rescale foreground & background
        a_t = mx.nd.broadcast_axis(mx.nd.reshape(a_t,(-1,1)), axis=1, size=p_t.shape[1])
        label_mask = mx.nd.one_hot(label, num_class, on_value=1, off_value=0)   # convert one-hot label
        dp_t = (-a_t) * label_mask * mx.nd.power(1 - p_t, self.gamma - 1.0) * \
                   (1 - p_t - (self.gamma * p_t * mx.nd.log(mx.nd.maximum(p_t, self.eps)))) / p_t # dFL/dpt
        dp_t /= mx.nd.sum(label > 0).asscalar() if self.normalize else 1.0  # norm gradients
            
        self.assign(in_grad[0], req[0], dp_t)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('FocalLoss')
class FocalLossProp(mx.operator.CustomOpProp):
    def __init__(self, gamma, alpha, use_ignore, ignore_label, normalize=True):
        super(FocalLossProp, self).__init__(need_top_grad=False)
        self.use_ignore = False if use_ignore == 'False' else True  # seems to be trivial
        self.ignore_label = int(ignore_label)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.normalize = normalize

    def list_arguments(self):
        return ['data', 'labels']

    def list_outputs(self):
        return ['focal_loss']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        labels_shape = in_shape[1]
        out_shape = data_shape
        return  [data_shape, labels_shape], [out_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return FocalLossOperator(self.gamma, self.alpha, self.use_ignore, self.ignore_label, self.normalize)

