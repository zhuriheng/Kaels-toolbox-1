#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import os
import json
import re
import logging
import pprint
import docopt
import mxnet as mx

cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_path,'../lib'))
from io_hybrid import *
from net_util import *
from train_util import *
from config import merge_cfg_from_file
from config import cfg as _
cfg = _.TRAIN

# init global logger
log_format = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger()  # sync with internal logger
fhandler = None     # log to file


def _init_():
    '''
    Alternately training multi-image-classification networks on mxnet
    Update: 2018/05/24
    Author: @Northrend
    Contributor:

    Changelog:
    2018/05/24      v1.0            basic functions 

    Usage:
        altet_train.py              <input-cfg> 
        altet_train.py              -v | --version
        altet_train.py              -h | --help

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


def _gluon_net_add(net, block):
    with net.name_scope():
            net.add(block)
    return net
    

def main():
    logger.info('Configuration:')
    logger.info(pprint.pformat(_))
    assert cfg.ALTERNATE, logger.error("Set ALTER_TRAIN as True to train multi-nets alternately")
    
    # initialize arguments 
    kv_store = mx.kvstore.create(cfg.KV_STORE)
    ctx = [mx.gpu(i) for i in cfg.GPU_IDX]
    num_gpu = len(cfg.GPU_IDX)
    batch_size = num_gpu * cfg.BATCH_SIZE
    data_shape = cfg.INPUT_SHAPE
    num_nets = cfg.ALTERNATE_NUM_NETS
    num_classes = cfg.NUM_CLASSES
    assert len(num_classes)==num_nets, logger.error("Number of classes should be match with number of nets")
    num_samples = cfg.NUM_SAMPLES
    nets = list()

    # instantiate metrics
    eval_metrics = inst_eval_metrics(cfg.METRICS) 

    if cfg.ALTERNATE_PHASE == 1:
        logger.info("Alternately training phase 1:")
        # load master net
        sym, arg_params, aux_params = load_model(cfg.PRETRAINED_MODEL_PREFIX, cfg.PRETRAINED_MODEL_EPOCH, gluon_style=False)    # load mxnet style model
        net_master = mx.gluon.nn.Sequential()
        net_master_pretrained = load_model_gluon(sym, arg_params, aux_params, ctx, layer_name='flatten')
        net_master = _gluon_net_add(net_master, net_master_pretrained)
        nets.append(net_master)
        
        # load sub nets
        for i in range(num_nets):
            net_branch = gluon_init_classifer(num_classes[i], ctx)
            nets.append(net_branch)

    elif cfg.ALTERNATE_PHASE == 2:
        logger.info("Alternately training phase 2:")
        # load master net
        sym, arg_params, aux_params = load_model(cfg.PRETRAINED_MODEL_PREFIX+'-master-phase-1', cfg.PRETRAINED_MODEL_EPOCH, gluon_style=True) 
        net_master = mx.gluon.nn.Sequential()
        net_master_pretrained = load_model_gluon(sym, arg_params, aux_params, ctx)
        net_master = _gluon_net_add(net_master, net_master_pretrained)
        nets.append(net_master)

        # load sub nets
        for i in range(num_nets):
            sym, arg_params, aux_params = load_model(cfg.PRETRAINED_MODEL_PREFIX+'-branch-{}-phase-1'.format(i+1), cfg.PRETRAINED_MODEL_EPOCH, gluon_style=True)
            net_branch = mx.gluon.nn.Sequential()
            net_branch_pretrained = load_model_gluon(sym, arg_params, aux_params, ctx)
            # exec("net_branch_{} = mx.gluon.nn.Sequential()".format(i+1))
            # exec("net_branch_{} = _gluon_net_add(net_branch, net_branch_pretrained)".format(i+1))
            net_branch = _gluon_net_add(net_branch, net_branch_pretrained)
            nets.append(net_branch)

    # display net blocks
    if cfg.LOG_NET_PARAMS:
        logger.info("-" * 80 + "\nNets Params:")
        for net in nets:
            logger.info(net.collect_params())
        logger.info("-" * 80)

    # instantiate data iters
    assert len(cfg.TRAIN_REC) == len(cfg.DEV_REC) == cfg.ALTERNATE_NUM_NETS, logger.error("Number of datasets and nets does not match")
    train_iters = ['dummy' for x in range(num_nets)]
    dev_iters = ['dummy' for x in range(num_nets)]
    for i in range(cfg.ALTERNATE_NUM_NETS):
        train_iters[i], dev_iters[i] = inst_iterators(cfg.TRAIN_REC[i], cfg.DEV_REC[i], batch_size=batch_size, data_shape=data_shape)

    # set learning rates
    lrs, lr_schedulers = ['dummy' for x in range(num_nets+1)], ['dummy' for x in range(num_nets+1)]
    lrs[0], lr_schedulers[0] = inst_lr_scheduler(sum(num_samples), batch_size, kv_store, begin_epoch=cfg.PRETRAINED_MODEL_EPOCH, base_lr=cfg.BASE_LR, lr_factor=cfg.LR_FACTOR, step_epochs=cfg.STEP_EPOCHS) 
    for i in range(num_nets):
        lrs[i+1], lr_schedulers[i+1] = inst_lr_scheduler(num_samples[i], batch_size, kv_store, begin_epoch=cfg.PRETRAINED_MODEL_EPOCH, base_lr=cfg.BASE_LR, lr_factor=cfg.LR_FACTOR, step_epochs=cfg.STEP_EPOCHS) 
    
    # alternately training
    alternate_train_gluon(cfg.ALTERNATE_PHASE, nets, train_iters, dev_iters, num_samples, lrs, lr_schedulers, batch_size, eval_metrics, ctx, epochs=cfg.MAX_EPOCHS, weight_decays=cfg.WEIGHT_DECAY, momentums=cfg.MOMENTUM)
    

if __name__ == '__main__':
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(_init_.__doc__, version='alternate gluon training script {}'.format(version))
    _init_()
    logger.info('Start training job...')
    main()
    logger.info('...Done')
