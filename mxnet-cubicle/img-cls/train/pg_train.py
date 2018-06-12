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
import cv2
from collections import namedtuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import time

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
logger = logging.getLogger()  # sync with internal logger
fhandler = None     # log to file

def _init_():
    '''
    Precision-guided training for image classification task on mxnet
    Update: 2018/05/30
    Author: @Northrend
    Contributor:

    Changelog:
    2018/05/30      v1.0            basic functions

    Usage:
        pg_train.py                 <input-cfg>
        pg_train.py                 -v | --version
        pg_train.py                 -h | --help

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


def generate_guider(raw_img, gt_label, target_shape, guiding_model, fc_weights, preproc_kwargs, log=False):
        '''
        '''
        img = np_img_preprocessing(raw_img, keep_aspect_ratio=True, **preproc_kwargs)

        Batch = namedtuple('Batch', ['data'])
        img_batch = mx.nd.array(img[np.newaxis, :])
        guiding_model.forward(Batch([img_batch])) 
        outputs = guiding_model.get_outputs() 
        conv_feature_map = outputs[0].asnumpy()[0]
        score = outputs[1].asnumpy()[0]

        det_map = gen_det_map(conv_feature_map, fc_weights)
        det_map = det_map[gt_label].astype(np.float32)
        # assert round(float(raw_img.shape[0])/det_map.shape[0],2) == round(float(raw_img.shape[1])/det_map.shape[1],2), logger.error("!!!")
        sample_rate = [max(float(raw_img.shape[0])/det_map.shape[0], float(raw_img.shape[1])/det_map.shape[1]) for x in range(2)]   # get 1:1 down-sampling rate
        try:
            target_idx = target_search(det_map, target_shape, top_k=0)
            target_crop = recover_coordinates(raw_img, sample_rate, target_shape, target_idx)
            if log:
                logger.info("raw_image_shape={}, detection_map_shape={}, guider_indices={}, guider_shape={}".format(raw_img.shape, det_map.shape, target_idx, target_crop.shape))
            return target_crop 
        except:
            return None 

    
def main():
    logger.info('Configuration:')
    logger.info(pprint.pformat(_)) 
    
    # initialization
    img_lst = cfg.PG.INPUT_IMG_LST
    pg_img_lst = cfg.PG.IMG_CACHE_LST
    pg_img_save_path = cfg.PG.IMG_CACHE_PATH
    check_dir(pg_img_save_path)

    batch_size = 1  #   single image for now 
    target_shape = cfg.PG.TARGET_SHAPE 
    kwargs = dict()
    # kwargs['resize_w_h'] = cfg.INPUT_SHAPE[1:]
    kwargs['resize_min_max'] = cfg.RESIZE_RANGE
    kwargs['mean_rgb'] = cfg.MEAN_RGB
    kwargs['std_rgb'] = cfg.STD_RGB
    input_shape = cfg.INPUT_SHAPE
    output_group = cfg.PG.OUTPUT_GROUP
    gpu_index = cfg.GPU_IDX[0]  # single gpu for now
    sym, arg_params, aux_params = load_model(cfg.PG.GIUDING_MODEL_PREFIX, cfg.PG.GIUDING_MODEL_EPOCH, gluon_style=False)
    fc_weights = arg_params[cfg.PG.CLASSIFIER_WEIGHTS].asnumpy()
    model = init_forward_net(sym, arg_params, aux_params, batch_size, input_shape, ctx=mx.gpu(gpu_index), redefine_output_group=cfg.PG.OUTPUT_GROUP, allow_missing=True, allow_extra=True)

    with open(img_lst,'r') as fin, open(pg_img_lst,'w') as fout:
        for i, item in enumerate(fin.readlines()): 
            assert len(item.strip().split())<=2, logger.error("Invalid input file syntax")
            img_path, gt_label = item.strip().split()
            gt_label = int(gt_label)
            raw_img = cv2.imread(img_path)
            if cfg.PG.LOG_GEN_GUIDER:
                logger.info("Batch[{}]: {}".format(i, os.path.basename(img_path)))
            if np.shape(raw_img) == tuple():
                logger.error("Image file error")
                continue 
            guider = generate_guider(raw_img, gt_label, target_shape, model, fc_weights, kwargs, log=cfg.PG.LOG_GEN_GUIDER)
            if np.shape(guider):
                recover_img_path = os.path.join(pg_img_save_path, os.path.splitext(cfg.PG.IMG_CACHE_PREFIX+os.path.basename(img_path))[0]) + cfg.PG.IMG_CACHE_EXT 
                cv2.imwrite(recover_img_path, guider)
                # os.rename(recover_img_path, os.path.splitext(recover_img_path)[0])
                # fout.write("{} {}\n".format(os.path.splitext(recover_img_path)[0], gt_label))
                fout.write("{} {}\n".format(recover_img_path, gt_label))
            else:
                logger.info("Failed generating guider")
                

            # preprocess img
            # img = np_img_preprocessing(test_img, **kwargs)

            # det_map = generate_guiding_data(conv_feature_map, fc_weights)
            # print('det_map.shape:',det_map.shape)
            # # print('det_map:',det_map)
            # 
            # heat_map = cv2.resize(det_map[0].astype(np.float32), (test_img.shape[1], test_img.shape[0]))
            # # # heat_map /= np.absolute(heat_map).max()
            # # heat_map /= heat_map.max()
            # # print('heat_map.shape:',heat_map.shape)
            # # print('heat_map:',heat_map)

            # # heat_map = np.sum(conv_feature_map,axis=0) 
            # # heat_map = np.sum(det_map,axis=0) 
            # # heat_map = cv2.resize(heat_map.astype(np.float32), (test_img.shape[1], test_img.shape[0]))
            # # heat_map /= np.absolute(heat_map).max()
            # # print('heat_map.shape:',heat_map.shape)
            # # print('heat_map:',heat_map)
            # # assert False, 'debugging'
            # heat_map = (heat_map-heat_map.mean())/(np.linalg.norm(heat_map))
            # heat_map /= np.absolute(heat_map).max()
            # # heat_map /= heat_map.max()
            # # print('heat_map.shape:',heat_map.shape)
            # # print('heat_map:',heat_map)

            # plt.figure(figsize=(18, 18))
            # # plt.imshow(test_img)
            # im_show = test_img.astype(np.float32)/255 * 0.3 + plt.cm.jet(heat_map)[:,:,:3] * 0.7
            # plt.imshow(im_show)
            # plt.savefig('tmp/{:0>4}.png'.format(i))
            # plt.close()


if __name__ == '__main__':
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(_init_.__doc__, version='precision-guided training script {}'.format(version))
    _init_()
    logger.info('Start training job...')
    main()
    logger.info('...Done')
