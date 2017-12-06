#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2017/05/10 @Northrend
#
# Universal image classifier
# On caffe
#

from __future__ import print_function
import os
import sys
import json
import cv2
import numpy as np
import re
import csv
import docopt
import Queue
import threading
import time
from AvaLib import _time_it


PYCAFFE_ROOT = '/opt/caffe/python/'
sys.path.append(PYCAFFE_ROOT)
import caffe
import skimage.io
from caffe.proto import caffe_pb2
from caffe.io import blobproto_to_array

# MODEL_ROOT = '/disk2/Northrend/warden/'

GLOBAL_LOCK = threading.Lock()
GLOBAL_FLAG = False
ERROR_IMG = list()


def _init_():
    '''
    Inference script for image-classification task on caffe
    Update: 2017/07/19
    Author: @Northrend
    Contributor: 

    Change log:
    2017/07/18  v1.4    support batch inference
    2017/07/18  v1.3    support single image test
    2017/05/11  v1.2    support data prefix
    2017/05/11  v1.1    fix doc mislead
    2017/05/10  v1.0    basic functions

    Usage:
        caffe_image_classifier.py   <in-list> <out-log> [-t | --test]
                                    (--arch=str --weights=str --mean=str --label=str)
                                    [--model-root=str --batch-size=int --img-width=int]
                                    [--data-prefix=str --interval=int]
        caffe_image_classifier.py   -v | --version
        caffe_image_classifier.py   -h | --help

    Arguments:
        <in-list>       test samples path list
        <out-log>       output log

    Options:
        -h --help                   show this help screen
        -v --version                show current version
        -----------------------------------------------------------------------------
        -t --test                   single image test mode
        --arch=str                  network architecture (.prototxt)
        --weights=str               model file (.caffemodel)
        --mean=str                  mean file (.binaryproto)
        --label=str                 text file which maps label index to concrete word
        --model-root=str            path prefix of model & net-arch &... if needed
        --batch-size=int            number of samples per forward [default: 1]
        --img-width=int             image width of model input [default: 224]
        --data-prefix=str           prefix of image path, if needed
        --interval=int              interval seconds between each batch [default: 3]
    '''
    print('=' * 80 + '\nArguments submitted:')
    for key in sorted(args.keys()):
        print('{:<20}= {}'.format(key.replace('--', ''), args[key]))
    print('=' * 80)


class ImageQueue(threading.Thread):
    def __init__(self, thread_name, queue, src_lst):
        '''
        thread_name: thread name
        queue: process image queue
        src_lst: source image list
        '''
        threading.Thread.__init__(self, name=thread_name)
        self.queue = queue
        self.src_lst = src_lst

    def run(self):
        global GLOBAL_FLAG
        for image in self.src_lst:
            self.queue.put(image)
            GLOBAL_LOCK.acquire()
            # print image
            GLOBAL_LOCK.release()
        GLOBAL_FLAG = True


def _read_list(image_list_file):
    image_list = []
    f_image_list = open(image_list_file, 'r')
    for buff in f_image_list:
        image_list.append(buff.split()[0])
    return image_list


def _make_mean(meanfile):
    print('generating mean file...')
    mean_blob = caffe_pb2.BlobProto()
    mean_blob.ParseFromString(open(meanfile, 'rb').read())
    mean_npy = blobproto_to_array(mean_blob)
    mean_npy_shape = mean_npy.shape
    mean_npy = mean_npy.reshape(
        mean_npy_shape[1], mean_npy_shape[2], mean_npy_shape[3])
    print "done."
    return mean_npy


def _index_to_classname(label_file):
    f_label = open(label_file, 'r')
    label_list = []
    for buff in f_label:
        label_list.append(buff.split()[1])
    return label_list


def net_init():
    '''
    initialize network
    '''
    arguments = {}
    inference_architecture = args['--model-root'] + \
        args['--arch'] if args['--model-root'] else args['--arch']
    inference_weights = args['--model-root'] + \
        args['--weights'] if args['--model-root'] else args['--weights']
    mean_file = args['--model-root'] + \
        args['--mean'] if args['--model-root'] else args['--mean']
    label_file = args['--model-root'] + \
        args['--label'] if args['--model-root'] else args['--label']
    batch_size = int(args['--batch-size'])
    image_width = int(args['--img-width'])       # matches model input

    label_list = _index_to_classname(label_file)
    net = caffe.Net(str(inference_architecture),        # defines the structure of the model which contains the trained weights
                    str(inference_weights),
                    caffe.TEST)                         # use test mode (e.g., don't perform dropout)
    meanf = _make_mean(mean_file)
    meannpy = meanf.mean(1).mean(1)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # move image channels to outermost dimension
    transformer.set_transpose('data', (2, 0, 1))
    # subtract the dataset-mean value in each channel
    transformer.set_mean('data', meannpy)
    # rescale from [0, 1] to [0, 255]
    transformer.set_raw_scale('data', 255)
    # swap channels from RGB to BGR
    transformer.set_channel_swap('data', (2, 1, 0))
    net.blobs['data'].reshape(batch_size,        # batch size
                              3,         # 3-channel (BGR) images
                              image_width, image_width)  # image size is 224*224
    caffe.set_mode_gpu()

    model = {}
    model['net'] = net
    model['transformer'] = transformer
    model['label_list'] = label_list

    return model


def net_single_infer(model, lst_image_path):
    '''
    predict single batch of input
    '''
    global ERROR_IMG
    output_dict = {}
    net = model['net']
    transformer = model['transformer']
    label_list = model['label_list']
    for index, image_path in enumerate(lst_image_path):
        try:
            img = caffe.io.load_image(image_path)
            transformed_image = transformer.preprocess('data', img)
        except Exception, e:
            print Exception, ":", e
            ERROR_IMG.append(os.path.basename(image_path))
            continue
        net.blobs['data'].data[index] = transformed_image
    # print net.blobs['data'].data.shape
    # print transformed_image.shape
    # print type(net.blobs['data'].data)
    # print net.blobs['data'].data

    tic = time.time()
    output = net.forward()
    toc = time.time()
    print "forward time: " + str(toc - tic) + "s"
    lst_result = list()
    for index, output_prob in enumerate(output['prob']):
        # current batch can not satisfit batch-size argument
        if index > (len(lst_image_path) - 1):
            continue
        output_prob = output['prob'][index]
        # print output
        # print output_prob

        # sort index list & create sorted rate list
        index_list = output_prob.argsort()
        rate_list = output_prob[index_list]

        result_dict = dict()
        result_dict['File Name'] = os.path.basename(lst_image_path[index])
        result_dict['Top-1 Index'] = index_list[-1]
        result_dict['Top-1 Class'] = label_list[index_list[-1]]
        # avoid JSON serializable error
        result_dict['Confidence'] = [str(i) for i in list(output_prob)]
        lst_result.append(result_dict)
    return lst_result


def net_list_infer(model, image_list):
    '''
    recurrently predict list of inputs 
    '''
    dict_result = {}
    src_lst = list()
    img_queue = Queue.Queue(maxsize=int(args['--batch-size']))
    for image_path in image_list:
        if args['--data-prefix']:
            image_path = str(args['--data-prefix']) + image_path
        src_lst.append(image_path)
    thread_queue = ImageQueue('ImgQueue', img_queue, src_lst)
    thread_queue.start()
    i = 0
    while(True):
        if img_queue.full() or (GLOBAL_FLAG and not img_queue.empty()):
            lst_image = list()
            i += 1
            print('Processing {}th batch...'.format(i))
            GLOBAL_LOCK.acquire()
            for count in xrange(img_queue.qsize()):
                lst_image.append(img_queue.get())
            # print len(lst_image)
            dict_result_tmp = net_single_infer(model, lst_image)
            # print dict_result_tmp
            for item in dict_result_tmp:
                dict_result[os.path.basename(item['File Name'])] = item
            GLOBAL_LOCK.release()
            time.sleep(int(args['--interval']))
        elif GLOBAL_FLAG and img_queue.empty():
            break
    return dict_result


@_time_it.time_it
def main():
    image_list = _read_list(args['<in-list>'])
    model = net_init()
    result = net_list_infer(model, image_list)
    # delete error image result
    for img in ERROR_IMG:
        print('error image: ' + img)
        del result[img]
    print('error image number: ' + str(len(ERROR_IMG)))
    log_result = open(args['<out-log>'], 'w')
    json.dump(result, log_result, indent=4)
    log_result.close()


def test():
    '''
    predict single image
    '''
    image_path = [str(args['<in-list>'])]
    model = net_init()
    print net_single_infer(model, image_path)


def test_code():
    img_queue = Queue.Queue(maxsize=int(args['--batch-size']))
    src_lst = ['test1.jpg', 'test2.jpg', 'test1.jpg', 'test2.jpg', 'test1.jpg',
               'test2.jpg', 'test1.jpg', 'test2.jpg', 'test1.jpg', 'test2.jpg']
    model = net_init()
    thread_queue = ImageQueue('ImgQueue', img_queue, src_lst)
    thread_queue.start()
    while(True):
        if img_queue.full() or (GLOBAL_FLAG and not img_queue.empty()):
            lst_image = list()
            GLOBAL_LOCK.acquire()
            for count in xrange(img_queue.qsize()):
                lst_image.append(img_queue.get())
            print('batch loaded')
            print net_single_infer(model, lst_image)
            GLOBAL_LOCK.release()
            time.sleep(int(args['--interval']))
        elif GLOBAL_FLAG and img_queue.empty():
            break
    print('All done')


if __name__ == "__main__":
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(
        _init_.__doc__, version='Caffe image classifer {}'.format(version))
    _init_()
    print('Start predicting image label...')
    if args['--test']:
        test()
    else:
        main()
    print('...done')
