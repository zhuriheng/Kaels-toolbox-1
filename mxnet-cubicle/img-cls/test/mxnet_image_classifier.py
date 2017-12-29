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
import json
import cv2
import mxnet as mx
import numpy as np
import re
import csv
import docopt
from collections import namedtuple
from AvaLib import _time_it


def _init_():
    '''
    Inference script for image-classification task on mxnet
    Update: 2017/12/29
    Author: @Northrend
    Contributor: 

    Change log:
    2017/12/29  v2.4    fix numpy truth value err bug
    2017/12/11  v2.3    fix center crop bug
    2017/12/07  v2.2    convert img-data to float before resizing
    2017/11/29  v2.1    support center crop
    2017/11/17  v2.0    support mean and std
    2017/09/25  v1.3    support alternative gpu
    2017/09/21  v1.2    support batch-inference & test mode
    2017/07/31  v1.1	support different label file
    2017/06/20  v1.0    basic functions

    Usage:
        mxnet_image_classifier.py       <in-list> <out-log> [-c|--confidence] [-t|--test] [--center-crop]
                                        (--label=str --model-prefix=str --model-epoch=int)
                                        [--batch-size=int --img-width=int --data-prefix=str]
                                        [--top-k=int --label-position=int --gpu=int]
                                        [--pre-crop-width=int --mean=lst --std=lst]
        mxnet_image_classifier.py       -v | --version
        mxnet_image_classifier.py       -h | --help

    Arguments:
        <in-list>       test samples path list
        <out-log>       output log

    Options:
        -h --help                   show this help screen
        -v --version                show current version
        -----------------------------------------------------------------------------
        -c --confidence             set to output confidence of each class
        -t --test                   single image test mode
        --center-crop               set to use center crop
        --gpu=int                   choose one gpu to run network [default: 0]
        --label=str                 text file which maps label index to concrete word
        --label-position=int        classname position in label file [default: 1]
        --model-prefix=str          prefix of .params file and .json file
        --model-epoch=int           epoch number of model to load [default: 0]
        --batch-size=int            number of samples per forward [default: 1]
        --img-width=int             image width of model input [default: 224]
        --data-prefix=str           prefix of image path, if needed
        --top-k=int                 output top k classes prediction [default: 1]  
        --mean=lst                  list of rgb mean value [default: 123.68,116.779,103.939]
        --std=lst                   list of rgb std value [default: 58.395,57.12,57.375]
        --pre-crop-width=int        set image resize width before cropping if center crop is true [default: 256]
    '''
    print('=' * 80 + '\nArguments submitted:')
    for key in sorted(args.keys()):
        print('{:<20}= {}'.format(key.replace('--', ''), args[key]))
    print('=' * 80)


# golbal error file list
ERROR_LIST = list()


class empty_image(Exception):
    '''
    catch empty image error
    '''
    pass


def _read_list(image_list_file):
    '''
    read image path file
    file syntax, label is trivial:
    /path/to/image1.jpg (label)
    /path/to/image2.jpg (label)
    '''
    image_list = list()
    f_image_list = open(image_list_file, 'r')
    for buff in f_image_list:
        image_list.append(buff.split()[0])
    return image_list


def _index_to_classname(label_file, classname_position=1):
    '''
    load label file and get category list.
    set classname_position=0 to read file which has classname before index.
    file syntax:
    0 classname
    1 classname
    '''
    f_label = open(label_file, 'r')
    label_list = list()
    for buff in f_label:
        label_list.append(buff.strip().split()[classname_position])
    return label_list


def center_crop(img, crop_width):
    _, height, width = img.shape
    assert (height > crop_width and width > crop_width), 'crop size should be larger than image size!'
    top = int(float(height) / 2 - float(crop_width) / 2)
    left = int(float(width) / 2 - float(crop_width) / 2)
    crop = img[:, top:(top + crop_width), left:(left + crop_width)]
    return crop


def net_init():
    '''
    initialize mxnet model
    '''
    batch_size = int(args['--batch-size'])
    image_width = int(args['--img-width'])

    # get compute graph
    sym, arg_params, aux_params = mx.model.load_checkpoint(args['--model-prefix'], int(args['--model-epoch']))

    # ---- debugging ----
    # internals = sym.get_internals()
    # conv0 = internals['conv0_output']
    # group = mx.symbol.Group([sym, conv0])
    # ---- debugging ----

    # bind module with graph
    # ---- debugging ----
    # model = mx.mod.Module(symbol=group, context=mx.gpu(int(args['--gpu'])), label_names=None)
    # ---- debugging ----
    model = mx.mod.Module(symbol=sym, context=mx.gpu(int(args['--gpu'])), label_names=None)
    model.bind(for_training=False, data_shapes=[
               ('data', (batch_size, 3, image_width, image_width))], label_shapes=model._label_shapes)

    # load model parameters
    model.set_params(arg_params, aux_params, allow_missing=True)
    return model


def net_single_infer(model, list_image_path):
    '''
    predict label of one single image.
    '''
    global ERROR_LIST
    Batch = namedtuple('Batch', ['data'])
    batch_size = int(args['--batch-size'])
    image_width = int(args['--img-width'])
    resize_width = int(args['--pre-crop-width']) if args['--center-crop'] else image_width
    k = int(args['--top-k'])
    mean_r, mean_g, mean_b = float(args['--mean'].split(',')[0]
                                   ), float(args['--mean'].split(',')[1]), float(args['--mean'].split(',')[2])
    std_r, std_g, std_b = float(args['--std'].split(',')[0]), float(args['--std'].split(',')
                                                                    [1]), float(args['--std'].split(',')[2])

    img_batch = mx.nd.array(np.zeros((batch_size, 3, image_width, image_width)))
    for index, image_path in enumerate(list_image_path):
        # image preprocessing
        try:
            img_read = cv2.imread(image_path)
            if np.shape(img_read) == tuple():
                raise empty_image
        except:
            img_read = np.zeros((resize_width, resize_width, 3), dtype=np.uint8)
            ERROR_LIST.append(os.path.basename(image_path))
            print('image error: ', image_path, ', inference result will be deprecated!')
        img = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
        img = img.astype(float)
        img = cv2.resize(img, (resize_width, resize_width))
        # img[:,:,0] -= mean_r
        # img[:,:,0] /= std_r
        # img[:,:,1] -= mean_g
        # img[:,:,1] /= std_g
        # img[:,:,2] -= mean_b
        # img[:,:,2] /= std_b
        img -= [mean_r, mean_g, mean_b]
        img /= [std_r, std_g, std_b]
        # (h,w,c) => (b,c,h,w)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        # img = img[np.newaxis, :]
        if args['--center-crop']:
            img = center_crop(img, image_width)
        # img_batch[index] = mx.nd.array(img)[0]
        img_batch[index] = mx.nd.array(img)
    # print(mx.nd.array(img).shape)
    # print(img_batch.asnumpy())

    # forward propagation
    # print(Batch([mx.nd.array(img)]))
    # model.forward(Batch([mx.nd.array(img)]))
    model.forward(Batch([img_batch]))
    output_prob_batch = model.get_outputs()[0].asnumpy()

    # ---- debugging ----
    # conv_w = model.get_params()[0]['conv0_weight']
    # print(model.get_outputs()[1].asnumpy()[0,0,:3,:3])
    # ---- debugging ----

    # get label list
    label_list = _index_to_classname(
        args['--label'], int(args['--label-position']))

    # get prediction result
    list_result_dict = list()
    for index in xrange(len(list_image_path)):
        output_prob = output_prob_batch[index]

        # sort index-list and create sorted rate-list
        index_list = output_prob.argsort()
        rate_list = output_prob[index_list]

        # write result dictionary
        result_dict = dict()
        result_dict['File Name'] = os.path.basename(list_image_path[index])
        # get top-k indices and revert to top-1 at first
        result_dict['Top-{} Index'.format(k)] = index_list.tolist()[-k:][::-1]
        result_dict['Top-{} Class'.format(k)] = [label_list[int(i)]
                                                 for i in index_list.tolist()[-k:][::-1]]
        # use str to avoid JSON serializable error
        result_dict['Confidence'] = [str(i) for i in list(
            output_prob)] if args['--confidence'] else [str(i) for i in rate_list.tolist()[-k:][::-1]]
        list_result_dict.append(result_dict)
    return list_result_dict


def net_list_infer(model, image_list):
    '''
    process list of images
    '''
    global ERROR_LIST
    dict_result = dict()
    batch_size = int(args['--batch-size'])
    count = 0
    while(image_list):
        count += 1
        print('Processing {}th batch...'.format(count))
        buffer_image_list = list()
        for _ in xrange(batch_size):
            if not image_list:      # list is empty
                break
            elif args['--data-prefix']:
                buffer_image_list.append(
                    str(args['--data-prefix']) + image_list.pop(0))
            else:
                buffer_image_list.append(image_list.pop(0))
        temp_list_result = net_single_infer(model, buffer_image_list)
        for item in temp_list_result:
            dict_result[item['File Name']] = item
    for image in ERROR_LIST:        # deprecate error img results
        del dict_result[image]
    return dict_result


@_time_it.time_it
def test_2():
    '''
    test code for one batch
    '''
    image_list = ['test1.jpg', 'test2.jpg', 'test1.jpg', 'test2.jpg', 'test1.jpg',
                  'test2.jpg', 'test1.jpg', 'test2.jpg', 'test1.jpg', 'test2.jpg']
    model = net_init()
    print(net_list_infer(model, image_list))


@_time_it.time_it
def main():
    '''
    image classification job for list of images
    '''
    image_list = _read_list(args['<in-list>'])
    model = net_init()
    result = net_list_infer(model, image_list)
    log_result = open(args['<out-log>'], 'w')
    json.dump(result, log_result, indent=4)
    log_result.close()


@_time_it.time_it
def test():
    '''
    test code for one image
    '''
    image = [args['<in-list>']]
    model = net_init()
    print(net_single_infer(model, image))


if __name__ == "__main__":
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(
        _init_.__doc__, version='Mxnet image classifer {}'.format(version))
    _init_()
    print('MXNet version: ' + str(mx.__version__))
    print('Start predicting image label...')
    if args['--test']:
        test()
    else:
        main()
    print('...done')
