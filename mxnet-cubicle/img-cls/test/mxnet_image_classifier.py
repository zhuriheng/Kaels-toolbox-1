#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2017/06/20 @Northrend
#
# Universal image classifier
# On MXNet
#

from __future__ import print_function
import os
import sys
import time
import json
import cv2
import re
import csv
import docopt
import pprint
import logging
import mxnet as mx
import numpy as np
from AvaLib import _time_it

cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_path,'../lib'))
from io_hybrid import load_model, load_image_list, load_category_list 
from net_util import init_forward_net 
from test_util import multi_gpu_test 
from config import merge_cfg_from_file
from config import cfg as _
cfg = _.TEST

# init global logger
log_format = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger()  # sync with mxnet internal logger
fhandler = None     # log to file


def _init_():
    '''
    Inference script for image-classification task on mxnet
    Update: 2018-06-11 16:18:43 
    Author: @Northrend
    Contributor: 

    Change log:
    2018/06/11  v3.0                code-refactoring 
    2018/05/31  v2.6                support log file name with parent path 
    2018/04/18  v2.5                support print foward fps
    2017/12/29  v2.4                fix numpy truth value err bug
    2017/12/11  v2.3                fix center crop bug
    2017/12/07  v2.2                convert img-data to float before resizing
    2017/11/29  v2.1                support center crop
    2017/11/17  v2.0                support mean and std
    2017/09/25  v1.3                support alternative gpu
    2017/09/21  v1.2                support batch-inference & test mode
    2017/07/31  v1.1                support different label file
    2017/06/20  v1.0                basic functions

    Usage:
        mxnet_image_classifier.py   <input-cfg>
        mxnet_image_classifier.py   -v | --version
        mxnet_image_classifier.py   -h | --help

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


@_time_it.time_it
def main():
    logger.info('Configuration:')
    logger.info(pprint.pformat(_))

    # init
    devices = [mx.gpu(x) for x in cfg.GPU_IDX]
    batch_size_per_gpu = cfg.MULTI_CROP_NUM if cfg.MULTI_CROP else cfg.BATCH_SIZE
    batch_size = len(devices)*batch_size_per_gpu
    input_shape = cfg.INPUT_SHAPE
    center_crop = cfg.CENTER_CROP
    multi_crop_num = cfg.MULTI_CROP_NUM if cfg.MULTI_CROP else None 
    h_flip = cfg.HORIZENTAL_FLIP
    image_list, _label_list = load_image_list(cfg.INPUT_IMG_LST)
    kwargs = dict()
    if cfg.RESIZE_KEEP_ASPECT_RATIO:
        kwargs['resize_min_max'] = cfg.RESIZE_MIN_MAX
    else:
        kwargs['resize_w_h'] = cfg.RESIZE_WH
    kwargs['mean_rgb'] = cfg.MEAN_RGB
    kwargs['std_rgb'] = cfg.STD_RGB
    categories = load_category_list(cfg.INPUT_CAT_FILE, name_position=cfg.CAT_NAME_POS, split=cfg.CAT_FILE_SPLIT)
    symbol, arg_params, aux_params = load_model(cfg.MODEL_PREFIX, cfg.MODEL_EPOCH)
    model = init_forward_net(symbol, arg_params, aux_params, batch_size, input_shape, ctx=devices, redefine_output_group=None, allow_missing=True, allow_extra=False)
    # test
    result = multi_gpu_test(model, image_list, categories, batch_size, input_shape, kwargs, center_crop=center_crop, multi_crop=multi_crop_num, h_flip=h_flip, img_prefix=None, base_name=True)
    # write json file
    with open(cfg.OUTPUT_JSON_PATH,'w') as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(
        _init_.__doc__, version='Mxnet image classifer {}'.format(version))
    _init_()
    logger.info('MXNet version: ' + str(mx.__version__))
    logger.info('Start predicting image label...')
    main()
    logger.info('...done')
