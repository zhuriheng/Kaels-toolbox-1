# --------------------------------------------------------
# Focal loss
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by unsky https://github.com/unsky/
# --------------------------------------------------------

"""
Focal loss 
"""

from __future__ import print_function
import mxnet as mx
import numpy as np

class FocalLossOperator(mx.operator.CustomOp):
    def __init__(self,  gamma, alpha,use_ignore,ignore_label):
        super(FocalLossOperator, self).__init__()
        self._gamma = gamma
        self._alpha = alpha 
        self.use_ignore = use_ignore
        self.ignore_label = ignore_label
        # self.normalize = normalize
        self.normalize = True
        self.eps=1e-14
        # print('Focalloss params: ',self._gamma,self._alpha,self.use_ignore,self.ignore_label)

    def forward(self, is_train, req, in_data, out_data, aux):
        # print('----forward----')
        # for index,data in enumerate(in_data):
        #     print('in_data[{}].shape:{}'.format(index,data.shape))
        # self.label = in_data[1]
        
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # print('----backward----')
        # for index,data in enumerate(in_data):
        #     print('in_data[{}].shape:{}'.format(index,data.shape))
        # label = mx.nd.reshape(in_data[1], (0, 1, -1))
        # label = mx.nd.reshape(in_data[1], (-1, 1))
        label = in_data[1]
        # print('label:',label)
        # print('label.shape:',label.shape)
        # p = mx.nd.pick(in_data[0], label, axis=1, keepdims=True)
        # print('p.shape',p.shape)

        n_class = in_data[0].shape[1]
        # print('n_class:',n_class)

        p_t = in_data[0] 
        # print('p_t.shape:',p_t.shape)

        a_t = (label > 0) * self._alpha + (label == 0) * (1 - self._alpha)  # rescale background & foreground
        a_t = mx.nd.broadcast_axis(mx.nd.reshape(a_t,(-1,1)), axis=1, size=p_t.shape[1])
        # print('a_t[0]:',a_t[0])
        # print('a_t.shape:',a_t.shape)
        label_mask = mx.nd.one_hot(label, n_class, on_value=1, off_value=0)
        # print('label_mask.shape:',label_mask.shape)
        # p_t = p_t * label_mask
        # p_t += self.eps
        # print('p_t[0]:',p_t[0])
        dp_t = (-a_t) * label_mask * mx.nd.power(1-p_t, self._gamma-1.0) * (1-p_t-(self._gamma*p_t*mx.nd.log(mx.nd.maximum(p_t, self.eps)))) / p_t
        # print('dp_t.shape:',dp_t.shape)


        # u = 1 - p - (self._gamma * p * mx.nd.log(mx.nd.maximum(p, self.eps)))
        # v = 1 - p if self._gamma == 2.0 else mx.nd.power(1 - p, self._gamma - 1.0)
        # a = (label > 0) * self._alpha + (label == 0) * (1 - self._alpha)  # rescale background & foreground
        # print('u.shape:',u.shape,'\nv.shape:',v.shape,'\na.shape:',a.shape)
        # gf = v * u * a.reshape((0,1))
        # print('gf.shape:',gf.shape)
        
        # label_ = mx.nd.reshape(label, (0, -1))
        # print('label.shape:',label.shape)

        # label_mask = mx.nd.one_hot(label, n_class, on_value=1, off_value=0)
        # print('label_mask:',label_mask)
        # label_mask = mx.nd.transpose(label_mask, (0, 2, 1))
        # label_mask = mx.nd.reshape(label_mask,(0, -1))
        # gf = mx.nd.reshape(gf,(0,))

        # print('label_mask.shape:',label_mask.shape)
        # print('gf.shape:',gf.shape)

        # g = (in_data[0] - label_mask) * gf
        # g = gf * (in_data[0] - label_mask)
        # g *= (label >= 0)  #

        if self.normalize:
            # g /= mx.nd.sum(label > 0).asscalar()
            # print('dp_t[0] before norm:',dp_t[0])
            # dp_t /= mx.nd.sum(label > 0).asscalar()
            dp_t /= in_data[1].shape[0]		# 128 
            # print('norm scale:',mx.nd.sum(label > 0).asscalar())
            # print('dp_t[0] after norm:',dp_t[0])
            

        self.assign(in_grad[0], req[0], dp_t)
        self.assign(in_grad[1], req[1], 0)
        # self.assign(in_grad[2], req[2], 0) 

        # assert False, 'Debugging'

@mx.operator.register('FocalLoss')
class FocalLossProp(mx.operator.CustomOpProp):
    def __init__(self, gamma,alpha,use_ignore,ignore_label):
        super(FocalLossProp, self).__init__(need_top_grad=False)
        # self.use_ignore = bool(use_ignore)
        self.use_ignore = False if use_ignore == 'False' else True
        print('use_ignore: ',self.use_ignore)
        self.ignore_label = int(ignore_label)

        self._gamma = float(gamma)
        self._alpha = float(alpha)

    def list_arguments(self):
        return ['data', 'labels']

    def list_outputs(self):
        return ['focal_loss']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        labels_shape = in_shape[1]
        out_shape = data_shape

        return  [data_shape, labels_shape],[out_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return FocalLossOperator(self._gamma,self._alpha,self.use_ignore,self.ignore_label)

    # def declare_backward_dependency(self, out_grad, in_data, out_data):
    #     return []
