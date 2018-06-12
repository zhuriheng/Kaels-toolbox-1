#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2017/05/05 @Northrend
#
# Training script
# for mxnet image classification
#

from __future__ import print_function
import sys
import os
import time
import math
import re
import logging
import docopt
import pprint
import mxnet as mx
from importlib import import_module
from operator_py import svm_metric 

cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_path,'../lib'))
from io_hybrid import *
from net_util import *
from train_util import *
from cam_util import *
from config import merge_cfg_from_file
from config import cfg as _
cfg = _.TRAIN

# init global logger
log_format = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger()  # sync with mxnet internal logger
fhandler = None     # log to file


def _init_():
    '''
    Training script for image-classification task on mxnet
    Update: 2018-06-07 16:43:27
    Author: @Northrend
    Contributor:

    Changelog:
    2018/06/07  v4.0                code-refactoring 
    2018/03/12  v3.2                support freeze feature layer weights 
    2018/02/28  v3.1                support svm classifier 
    2018/02/11  v3.0                support customized finetune layer name
    2018/02/10  v2.9                support resize dev data separately
    2018/02/03  v2.8                support random resize scale
    2018/01/29  v2.7                fix resume training job bugs
    2017/12/27  v2.6                support se-inception-v4
    2017/12/04  v2.5                support change shorter edge size
    2017/11/21  v2.4                support input nomalization
    2017/09/22  v2.3                support resume training job 
    2017/09/04  v2.2                support modify cardinality for resnext
    2017/08/15  v2.1                support modify dropout argument
    2017/08/09  v2.0                apply logging module
    2017/08/08  v1.5                support scratch-training and more networks
    2017/07/24  v1.4                support customized input shape
    2017/07/13	v1.3	            fix lr descend bug
    2017/05/19  v1.2                support multi-metrics during training
                                    support threshold and save-json options
                                    modify script name from mxnet_train_finetune to mxnet_train
    2017/05/10  v1.1                separate _init_ as a function
    2017/05/08  v1.0                finetune task tested
                                    resume training mode unsupported

    Usage:
        mxnet_train.py              <input-cfg>
        mxnet_train.py              -v | --version
        mxnet_train.py              -h | --help

    Arguments:
        <input-cfg>                 path to customized config file

    Options:
        -h --help                   show this help screen
        -v --version                show current version
    
    '''
    # merge configuration
    merge_cfg_from_file(args["<input-cfg>"])

    # config logger
    logger.setLevel(eval('logging.' + cfg.LOG_LEVEL))
    assert cfg.LOG_PATH, logger.error('Missing LOG_PATH!')
    fhandler = logging.FileHandler(cfg.LOG_PATH, mode=cfg.LOG_MODE)
    logger.addHandler(fhandler)

    # print arguments
    logger.info('=' * 80 + '\nCalled with arguments:')
    for key in sorted(args.keys()):
        logger.info('{:<20}= {}'.format(key.replace('--', ''), args[key]))
    logger.info('=' * 80)

    # reset logger format
    fhandler.setFormatter(logging.Formatter(log_format))


def test_io(rec, batch_size, log_interval):
    tic = time.time()
    for i, batch in enumerate(rec):
        for j in batch.data:
            j.wait_to_read()
        if (i + 1) % log_interval == 0:
            logger.info('Batch [%d]\tSpeed: %.2f samples/sec' % (i+1, float(log_interval)*batch_size / (time.time()-tic)))
            tic = time.time()
    return 0


def main():
    logger.info('Configuration:')
    logger.info(pprint.pformat(_))
    logger.info('MXNet version: {}'.format(mx.__version__))
    
    # initialization
    num_classes = cfg.NUM_CLASSES 
    num_samples = cfg.NUM_SAMPLES
    batch_per_gpu = cfg.BATCH_SIZE
    resize = cfg.RESIZE_SHAPE 
    num_gpus = len(cfg.GPU_IDX) 
    batch_size = batch_per_gpu * num_gpus
    log_interval = cfg.LOG_INTERVAL
    train, dev = inst_iterators(cfg.TRAIN_REC, cfg.DEV_REC, batch_size=batch_size, data_shape=cfg.INPUT_SHAPE, resize=cfg.RESIZE_SHAPE, resize_scale=cfg.RESIZE_SCALE, use_svm_label=cfg.USE_SVM_LABEL)
    
    # io testing mode
    if cfg.TEST_IO_MODE:
        test_io(train, batch_size, log_interval)
        return 0

    # init network
    if cfg.FINETUNE:
        begin_epoch = cfg.FT.PRETRAINED_MODEL_EPOCH 
        layer_name = cfg.FT.FINETUNE_LAYER 
        logger.info("Finetune layer:".format(layer_name))
        symbol, arg_params, aux_params = load_model(cfg.FT.PRETRAINED_MODEL_PREFIX, begin_epoch, gluon_style=False)
        svm = cfg.SVM_LOSS if cfg.USE_SVM else None
        reg_coeff = cfg.SVM_REG_COEFF if cfg.USE_SVM else None
        symbol, arg_params = general_finetune_model(symbol, arg_params, num_classes, layer_name=layer_name, use_svm=svm, reg_coeff=reg_coeff, gluon_style=False)
    elif cfg.RESUME:
        begin_epoch = cfg.RES.MODEL_EPOCH 
        symbol, arg_params, aux_params = load_model(cfg.RES.MODEL_PREFIX, begin_epoch, gluon_style=False)
    elif cfg.SCRATCH:
        begin_epoch = 0
        network = import_module('symbols.'+cfg.SCR.NETWORK)
        symbol = network.get_symbol(num_classes, cfg.SCR.NUM_LAYERS, cfg.INPUT_SHAPE, **kwargs) 
        arg_params, aux_params = None, None
    else:
        logger.error("Please at least choose one training mode: scratch, finetune or resume")
        return 0
    if cfg.LOG_NET_PARAMS:
        dummy_data_shape = (batch_size, cfg.INPUT_SHAPE[0], cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[2])
        for idx,buff in enumerate(zip(symbol.list_arguments(), symbol.infer_shape(data=dummy_data_shape)[0])):
            logger.info("Param [{}] {}".format(idx, buff))
    
    # training
    socre_dev = generic_train(train, dev, symbol, arg_params, aux_params, num_samples, batch_size, begin_epoch)
    logger.info("Final evaluation on dev-set:")
    for tup in socre_dev:
        logger.info("Validation-{}={:.6f}".format(tup[0],tup[1]))


if __name__ == '__main__':
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(
        _init_.__doc__, version='General mxnet training script {}'.format(version))
    _init_()
    logger.info('Start training job...')
    main()
    logger.info('...Done')

