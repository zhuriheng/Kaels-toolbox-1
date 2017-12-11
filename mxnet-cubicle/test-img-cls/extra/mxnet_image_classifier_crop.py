#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2017/06/20 @Northrend
#
# Universal image classifier
# On MXNet
#

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
    Update: 2017/11/28
    Author: @Northrend
    Contributor: 

    Change log:
    2017/11/28  v2.0    support mean and std
    2017/07/31  v1.1	support different label file
    2017/06/20  v1.0    basic functions

    Usage:
        mxnet_image_classifier.py       <in-list> <out-log> [-c|--confidence]
                                        (--label=str --model-prefix=str --model-epoch=int)
                                        [--batch-size=int --img-width=int --data-prefix=str]
                                        [--top-k=int --label-position=int --gpu=int]
                                        [--mean=lst --std=lst]

        mxnet_image_classifier.py       -v | --version
        mxnet_image_classifier.py       -h | --help

    Arguments:
        <in-list>                   test samples path list
        <out-log>                   output log

    Options:
        -h --help                   show this help screen
        -v --version                show current version
        -----------------------------------------------------------------------------
        -c --confidence             set to output confidence of each class
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
    '''
    print '-' * 80 + '\nArguments submitted:'
    for key in sorted(args.keys()):
        print '{:<20}= {}'.format(key.replace('==', ''), args[key])
    print '-' * 80


def _read_list(image_list_file):
    '''
    read image path file
    file syntax, label is trivial:
    /path/to/image1.jpg label
    /path/to/image1.jpg label
    '''
    image_list = []
    f_image_list = open(image_list_file, 'r')
    for buff in f_image_list:
        image_list.append(buff.split()[0])
    return image_list


# def _make_mean(meanfile):
#     print('generating mean file...')
#     mean_blob = caffe_pb2.BlobProto()
#     mean_blob.ParseFromString(open(meanfile, 'rb').read())
#     mean_npy = blobproto_to_array(mean_blob)
#     mean_npy_shape = mean_npy.shape
#     mean_npy = mean_npy.reshape(
#         mean_npy_shape[1], mean_npy_shape[2], mean_npy_shape[3])
#     print "done."
#     return mean_npy


def _index_to_classname(label_file, classname_position=1):
    '''
    load label file and get category list
    set classname_position=0 to read file which has classname before index
    file syntax:
    0 classname
    1 classname
    '''
    f_label = open(label_file, 'r')
    label_list = []
    for buff in f_label:
        label_list.append(buff.strip().split()[classname_position])
    return label_list


def net_init():
    '''
    initialize mxnet model
    '''
    batch_size = int(args['--batch-size'])
    image_width = int(args['--img-width'])       # matches model input

    # get compute graph
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        args['--model-prefix'], int(args['--model-epoch']))

    # bind module with graph
    model = mx.mod.Module(symbol=sym, context=mx.gpu(1), label_names=None)
    model.bind(for_training=False, data_shapes=[
               ('data', (batch_size, 3, image_width, image_width))], label_shapes=model._label_shapes)

    # load model parameters
    model.set_params(arg_params, aux_params, allow_missing=True)
    return model


def net_single_infer(model, image_path):
    '''
    predict label of one single image
    batch size should be 6
    '''
    def _short_edge_resize(img,short_edge):
        '''
        resize image (height,width,channel), to fixed short edge size
        '''
        height, width, _  = img.shape
        if height > width:
            img_new = cv2.resize(img,(short_edge,int(float(height)*short_edge/width)))
        elif height <= width:
            img_new = cv2.resize(img,((int(float(width)*short_edge/height)),short_edge))
        # print 'image size: {}, resized to: {}'.format(img.shape,img_new.shape)
        return img_new                                

    def _weighted_prob(softmax_output):
        # print softmax_output.asnumpy()
        weight_full = 0.5
        weight_crop = 0.1
        weighted_prob = mx.nd.array(np.zeros((6,3)))
        crop_prob = softmax_output[1]
        for i in xrange(2,6):
            crop_prob += softmax_output[i] 
        weighted_prob = weight_full * softmax_output[0] + weight_crop * crop_prob
        return weighted_prob

    Batch = namedtuple('Batch', ['data']) 
    image_width = int(args['--img-width'])
    k = int(args['--top-k'])
    mean_r, mean_g, mean_b = float(args['--mean'].split(',')[0]), float(args['--mean'].split(',')[1]), float(args['--mean'].split(',')[2])
    std_r, std_g, std_b = float(args['--std'].split(',')[0]), float(args['--std'].split(',')[1]), float(args['--std'].split(',')[2])

    img_batch = mx.nd.array(np.zeros((int(args['--batch-size']),3,image_width,image_width)))

    # image preprocessing
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img = img.astype(float)
    img = _short_edge_resize(img,int(image_width/0.875))
    img -= [mean_r,mean_g,mean_b]
    img /= [std_r,std_g,std_b]
    
    img0 = cv2.resize(img, (image_width, image_width)).astype(float) # full image
    img0 -= [mean_r,mean_g,mean_b]
    img0 /= [std_r,std_g,std_b]
    img0 = np.swapaxes(img0, 0, 2)
    img0 = np.swapaxes(img0, 1, 2)
    # img_batch[0] = img0[np.newaxis, :]
    img_batch[0] = img0
    
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    # img = img[np.newaxis, :]
    _, height, width = img.shape
    centercrop_top = int(height/2)-int(image_width/2)
    centercrop_left = int(width/2)-int(image_width/2)
    img_batch[1] = img[:,0:image_width,0:image_width] # top left
    img_batch[2] = img[:,0:image_width,width-image_width:] # top right
    img_batch[3] = img[:,height-image_width:,0:image_width] # bottom left
    img_batch[4] = img[:,height-image_width:,width-image_width:] # bottom right
    img_batch[5] = img[:,centercrop_top:centercrop_top+image_width,centercrop_left:centercrop_left+image_width] # center crop

    # forward propagation
    # model.forward(Batch([mx.nd.array(img4)]))
    model.forward(Batch([img_batch]))
    # print model.get_outputs()[0].asnumpy()
    output_prob = _weighted_prob(model.get_outputs()[0]).asnumpy()
    
    # output_prob = model.get_outputs()[0].asnumpy()[0]

    # sort index-list and create sorted rate-list
    index_list = output_prob.argsort()
    rate_list = output_prob[index_list]

    # get label list
    label_list = _index_to_classname(args['--label'], int(args['--label-position']))

    # write result dictionary
    result_dict = dict()
    result_dict['File Name'] = os.path.basename(image_path)
    # get top-k indices and revert to top-1 at first
    result_dict['Top-{} Index'.format(k)] = index_list.tolist()[-k:][::-1]
    result_dict['Top-{} Class'.format(k)] = [label_list[int(i)]
                                             for i in index_list.tolist()[-k:][::-1]]
    # avoid JSON serializable error
    result_dict['Confidence'] = [str(i) for i in list(
        output_prob)] if args['--confidence'] else [str(i) for i in rate_list.tolist()[-k:][::-1]]

    return result_dict


def net_list_infer(model, image_list):
    '''
    process list of images
    '''
    dict_result = {}
    i = 0
    for image_path in image_list:
        i += 1
        image_path = str(args['--data-prefix']) + image_path \
            if args['--data-prefix'] else image_path
        dict_result[os.path.basename(image_path)] = net_single_infer(
            model, image_path)
        print 'Processing {}th image:{}'.format(i, image_path)
    return dict_result


@_time_it.time_it
def test():
    '''
    test code for one image
    '''
    image = args['<in-list>']
    model = net_init()
    print net_single_infer(model, image)


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


if __name__ == "__main__":
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(
        _init_.__doc__, version = 'Mxnet image classifer {}'.format(version))
    _init_()
    print 'MXNet version: ' + str(mx.__version__)
    print 'Start predicting image label...'
    main()
    # test()
    print '...done'

