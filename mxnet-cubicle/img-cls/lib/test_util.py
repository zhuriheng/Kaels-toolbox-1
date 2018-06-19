#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import math
import time
import logging
import cv2
import mxnet as mx
import numpy as np
from collections import namedtuple
from io_hybrid import np_img_preprocessing,np_img_center_crop,np_img_multi_crop
from config import cfg


def _get_filename_with_parents(filepath, level=1):
    common = filepath
    for i in range(level + 1):
        common = os.path.dirname(common)
    return os.path.relpath(filepath, common)


def infer_one_batch(model, categories, data_batch, img_list, base_name=True):
    '''
    '''
    Batch = namedtuple('Batch', ['data'])
    results_one_batch = list() 
    k = cfg.TEST.TOP_K
    level = cfg.TEST.FNAME_PARENT_LEVEL 
    model.forward(Batch([data_batch]))
    output_prob_batch = model.get_outputs()[0].asnumpy()
    for idx, img_name in enumerate(img_list):
        output_prob = output_prob_batch[idx]
        
        # sort index-list and create sorted rate-list
        index_list = output_prob.argsort()
        rate_list = output_prob[index_list]
        _index_list = index_list.tolist()[-k:][::-1]
        _rate_list = rate_list.tolist()[-k:][::-1]

        # write result dictionary
        result = dict()
        if base_name:
            result['File Name'] = os.path.basename(img_name)
        else:
            result['File Name'] = _get_filename_with_parents(img_name, level=level)
        # result['File Name'] = img_name 

        # get top-k indices and revert to top-1 at first
        result['Top-{} Index'.format(k)] = _index_list
        result['Top-{} Class'.format(k)] = [categories[int(x)] for x in _index_list]

        # use str to avoid JSON serializable error
        result['Confidence'] = [str(x) for x in list(output_prob)] if cfg.TEST.LOG_ALL_CONFIDENCE else [str(x) for x in _rate_list]
        results_one_batch.append(result)
    return results_one_batch
    

def multi_gpu_test(model, img_list, categories, batch_size, input_shape, img_preproc_kwargs, center_crop=False, multi_crop=None, h_flip=False, img_prefix=None, base_name=True):
    level = cfg.TEST.FNAME_PARENT_LEVEL 
    result = dict()
    count = 0 
    err_num = 0
    while(img_list):
        count += 1
        # list of one batch data
        buff_list = list()
        error_list = list()
        for i in range(batch_size):
            if not img_list:
                logging.debug("current list empty")
                break
            elif img_prefix:
                buff_list.append(img_prefix + img_list.pop(0))
            else:
                buff_list.append(img_list.pop(0))
        # process one data batch
        img_batch = mx.nd.array(np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))) 
        for idx,img in enumerate(buff_list):
            try:
                img_read = cv2.imread(img)
                if np.shape(img_read) == tuple():
                    raise empty_image
            except:
                img_read = np.zeros((input_shape[1], input_shape[2], input_shape[0]), dtype=np.uint8)
                if base_name:
                    error_list.append(os.path.basename(img))
                else:
                    error_list.append(_get_filename_with_parents(img, level=level))
                logging.error('Image error: {}, result will be deprecated!'.format(img))
            img_tmp = np_img_preprocessing(img_read, **img_preproc_kwargs)
            if center_crop:
                img_ccr = np_img_center_crop(img_tmp, inpt_shape[1]) 
                img_batch[idx] = mx.nd.array(img_ccr[np.newaxis, :])
            elif multi_crop:
                img_crs = np_img_multi_crop(img_tmp, inpt_shape[1], crop_number=multi_crop)
                for idx_crop,crop in enumerate(img_crs):
                    img_batch[idx_crop] = mx.nd.array(crop[np.newaxis, :])
            else:
                img_batch[idx] = mx.nd.array(img_tmp[np.newaxis, :])
        buff_result = infer_one_batch(model, categories, img_batch, buff_list, base_name=True)
        for buff in buff_result:
            result[buff["File Name"]] = buff
        for img in error_list:
            del result[img]
        err_num += len(error_list)
        logging.info("Batch [{}]: gpu_number={}\tbatch_size={}\terror_number={}".format(count, len(cfg.TEST.GPU_IDX), len(buff_result), len(error_list)))
    logging.info("Tocal error image number:", err_num)
    return result

