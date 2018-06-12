#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import logging
import mxnet as mx
import cv2
import numpy as np
from config import cfg


def check_dir(path):
    '''
    '''
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        logging.info("{} does not exist, and it is now created.".format(dirname))
        os.mkdir(dirname)


def inst_iterators(data_train, data_dev, batch_size=1, data_shape=(3,224,224), resize=(-1,-1), resize_scale=(1,1), use_svm_label=False):
    '''
    Instantiate specified training and developing data iterators
    :params:
    data_train      training iterator
    data_dev        developing iterator
    batch_size      mini batch size, sum of all device
    data_shape      input shape
    resize          resize shorter edge of (train,dev) data, -1 means no resize
    resize_scale    resize train-data into (width*s, height*s), with s randomly chosen from this range 
    use_svm_label   set as True if classifier needs svm label name
    :return:
    train, dev      tuple of 2 iterators
    '''
    # initialization
    assert data_train and data_dev, logging.error("Please input training or developing data") 
    mean, std = cfg.TRAIN.MEAN_RGB, cfg.TRAIN.STD_RGB 
    assert len(mean)==3 and len(std)==3, logging.error("Mean or Std should be a list of 3 items")
    mean_r, mean_g, mean_b, std_r, std_g, std_b = mean[:] + std[:] 
    max_random_scale, min_random_scale = resize_scale 
    logging.info('Input normalization : Mean-RGB {}, Std-RGB {}'.format([mean_r, mean_g, mean_b],[std_r, std_g, std_b]))
    resize_train, resize_dev = resize
    label_name = 'softmax_label' if not use_svm_label else 'svm_label'

    # build iterators
    train = mx.io.ImageRecordIter(
        dtype=cfg.TRAIN.DATA_TYPE,
        path_imgrec=data_train,
        preprocess_threads=cfg.TRAIN.PROCESS_THREAD,
        data_name='data',
        label_name=label_name,
        label_width=cfg.TRAIN.LABEL_WIDTH,
        data_shape=data_shape,
        batch_size=batch_size,
        resize=resize_train,
        max_random_scale=max_random_scale,
        min_random_scale=min_random_scale,
        shuffle=cfg.TRAIN.SHUFFLE,
        rand_crop=cfg.TRAIN.RAND_CROP,
        rand_mirror=cfg.TRAIN.RAND_MIRROR,
        max_rotate_angle=cfg.TRAIN.MAX_ROTATE_ANGLE,
        max_aspect_ratio=cfg.TRAIN.MAX_ASPECT_RATIO,
        max_shear_ratio=cfg.TRAIN.MAX_SHEAR_RATIO,
        mean_r=mean_r,
        mean_g=mean_g,
        mean_b=mean_b,
        std_r=std_r,
        std_g=std_g,
        std_b=std_b
        )
    val = mx.io.ImageRecordIter(
        dtype=cfg.TRAIN.DATA_TYPE,
        path_imgrec=data_dev,
        preprocess_threads=cfg.TRAIN.PROCESS_THREAD,
        data_name='data',
        label_name=label_name,
        label_width=cfg.TRAIN.LABEL_WIDTH,
        batch_size=batch_size,
        data_shape=data_shape,
        resize=resize_dev,
        shuffle=False,
        rand_crop=False,
        rand_mirror=False,
        mean_r=mean_r,
        mean_g=mean_g,
        mean_b=mean_b,
        std_r=std_r,
        std_g=std_g,
        std_b=std_b
        )
    logging.info("Data iters created successfully")
    return train, val


def load_model(model_prefix, load_epoch, gluon_style=False):
    '''
    Load existing model
    :params:
    model_prefix        prefix of model with path
    load_epoch          which epoch to load
    gluon_style         set True to load model saved by gluon
    :return:
    sym, arg, aux       symbol, arg_params, aux_params of this model
                        aux_params will be an empty dict in gluon style
    '''
    assert model_prefix and load_epoch is not None, logging.error('Missing valid pretrained model prefix')
    assert load_epoch is not None, logging.error('Missing epoch of pretrained model to load')
    if not gluon_style:
        sym, arg, aux = mx.model.load_checkpoint(model_prefix, load_epoch)
    else:
        sym = mx.sym.load(model_prefix+'-symbol.json')
        save_dict = mx.nd.load('%s-%04d.params'%(model_prefix, load_epoch))
        arg, aux = dict(), dict()
        for k, v in save_dict.items():
            arg[k] = v
    logging.info('Loaded model: {}-{:0>4}.params'.format(model_prefix, load_epoch))

    return sym, arg, aux


def load_model_gluon(symbol, arg_params, aux_params, ctx, layer_name=None):
    '''
    Use to load net and params with gluon after load_model()
    '''
    def _init_gluon_style_params(raw_params, net_params, ctx):
        '''
        '''
        for param in raw_params:
            if param in net_params:
                net_params[param]._load_init(raw_params[param], ctx=ctx)
        return net_params

    if layer_name:
        net = symbol.get_internals()[layer_name + '_output']
    else:
        net = symbol
    net_hybrid = mx.gluon.nn.SymbolBlock(outputs=net, inputs=mx.sym.var('data'))
    net_params = net_hybrid.collect_params()
    net_params = _init_gluon_style_params(arg_params,net_params,ctx)
    net_params = _init_gluon_style_params(aux_params,net_params,ctx)

    return net_hybrid


def save_model(model_prefix, rank=0):
    '''
    '''
    assert model_prefix, logging.error('Model-prefix is needed to save model')
    dst_dir = os.path.dirname(model_prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return mx.callback.do_checkpoint(model_prefix if rank == 0 else "%s-%d"%(model_prefix, rank))


def save_model_gluon(net, model_prefix, rank=0):
    '''
    '''
    net.collect_params().save("{}-{:0>4}.params".format(model_prefix, rank))
    net(mx.sym.Variable('data')).save("{}-symbol.json".format(model_prefix))
    logging.info('Saved checkpoint to \"{}-{:0>4}.params\"'.format(model_prefix, rank))
    return 0


def np_img_preprocessing(img, as_float=True, keep_aspect_ratio=False, **kwargs):
    assert isinstance(img, np.ndarray), logging.error("Input images should be type of numpy.ndarray")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if as_float:
        img = img.astype(float)
    # reshape
    if 'resize_w_h' in kwargs and not keep_aspect_ratio:
        img = cv2.resize(img, (kwargs['resize_w_h'][0], kwargs['resize_w_h'][1]))
    if 'resize_min_max' in kwargs and keep_aspect_ratio: 
        ratio = float(max(img.shape[:2]))/min(img.shape[:2])
        min_len, max_len = kwargs['resize_min_max']
        if min_len*ratio <= max_len:    # resize by min
            if img.shape[0] > img.shape[1]:     # h > w
                img = cv2.resize(img, (min_len, int(min_len*ratio)))
            elif img.shape[0] <= img.shape[1]:   # h <= w
                img = cv2.resize(img, (int(min_len*ratio), min_len)) 
        elif min_len*ratio > max_len:   # resize by max
            if img.shape[0] > img.shape[1]:     # h > w
                img = cv2.resize(img, (int(max_len/ratio), max_len))
            elif img.shape[0] <= img.shape[1]:   # h <= w
                img = cv2.resize(img, (max_len, int(max_len/ratio))) 
    # normalization
    if 'mean_rgb' in kwargs:
        img -= kwargs['mean_rgb'][:]
    if 'std_rgb' in kwargs:
        img /= kwargs['std_rgb'][:] 
    # (h,w,c) => (c,h,w)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    return img
