# -*- coding: utf-8 -*-
# created 2017/05/16 @Northrend
# updated 2017/05/16 @Northrend
#
# Compute and display heatmap
# On mxnet
#

from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import cv2
import re
import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt
import docopt


def _init_():
    '''
    Display heatmap of input image
    Update: 2017/05/16
    Author: @Northrend
    Contributor:

    Change log:
    2017/05/16  v1.0                basic functions

    Usage:
        mxnet_cam.py                <in-img> <out-img> [-d|--display]
                                    (--network=str --weights=str --label=str)
                                    [--gpu=int --top-k=int]
        mxnet_cam.py                -v | --version
        mxnet_cam.py                -h | --help

    Arguments:
        <in-img>                    test image
        <out-img>                   result image

    Options:
        -h --help                   show this help screen
        -v --version                show current version
        ---------------------------------------------------------------------------------
        -d --display                display mode
        --network=str               network architecture, *.json
        --weights=str               model file, *.params
        --label=str                 maps label index to concrete word
        --gpu=int                   choose gpu, cpu will be used by default [default: -1]
        --top-k=int                 top k classes to compute CAM [default: 5]
    '''
    print('=' * 80 + '\nArguments submitted:')
    for key in sorted(args.keys()):
        print('{:<20}={}'.format(key.replace('--', ''), args[key]))
    print('=' * 80)


def get_cam(conv_feat_map, weight_fc):
    assert len(weight_fc.shape) == 2
    if len(conv_feat_map.shape) == 3:
        C, H, W = conv_feat_map.shape
        assert weight_fc.shape[1] == C
        detection_map = weight_fc.dot(conv_feat_map.reshape(C, H * W))
        detection_map = detection_map.reshape(-1, H, W)
    elif len(conv_feat_map.shape) == 4:
        N, C, H, W = conv_feat_map.shape
        assert weight_fc.shape[1] == C
        M = weight_fc.shape[0]
        detection_map = np.zeros((N, M, H, W))
        for i in xrange(N):
            tmp_detection_map = weight_fc.dot(
                conv_feat_map[i].reshape(C, H * W))
            detection_map[i, :, :, :] = tmp_detection_map.reshape(-1, H, W)
    return detection_map


def draw_cam(rgb, width, height, top_k, conv_fm, weight_fc, category, score, display):
    '''
    draw class active map
    '''
    score_sort = -np.sort(-score)[:top_k]
    inds_sort = np.argsort(-score)[:top_k]
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 1 + top_k, 1)
    plt.imshow(rgb)
    cam = get_cam(conv_fm, weight_fc[inds_sort, :])
    for k in xrange(top_k):
        detection_map = np.squeeze(cam.astype(np.float32)[k, :, :])
        heat_map = cv2.resize(detection_map, (width, height))
        max_response = detection_map.mean()
        heat_map /= heat_map.max()

        im_show = rgb.astype(np.float32) / 255 * 0.3 +
            plt.cm.jet(heat_map / heat_map.max())[:, :, :3] * 0.7
        plt.subplot(1, 1 + top_k, k + 2)
        plt.imshow(im_show)
        print('Top {}: {}({.6f}), max_response={.4f}'.format(
            k + 1, category[inds_sort[k]], score_sort[k], max_response))

    if display:
        plt.show()
    plt.savefig(args['<out-img>'])
    plt.close()
    return


def im2blob(img, width, height, mean=None, input_scale=1.0, raw_scale=1.0, swap_channel=True):
    '''
    return top k classes
    '''
    blob = cv2.resize(img, (height, width)).astype(np.float32)
    blob = blob.reshape((1, height, width, 3))

    #  n,h,w,c -> n,c,h,w
    blob = np.swapaxes(blob, 2, 3)
    blob = np.swapaxes(blob, 1, 2)

    if swap_channel:
        blob[:, [0, 2], :, :] = blob[:, [2, 0], :, :]

    if raw_scale != 1.0:
        blob *= raw_scale

    if isinstance(mean, np.ndarray):
        blob -= mean

    elif isinstance(mean, tuple) or isinstance(mean, list):
        blob[:, 0, :, :] -= mean[0]
        blob[:, 1, :, :] -= mean[1]
        blob[:, 2, :, :] -= mean[2]

    elif mean is None:
        pass
    else:
        raise TypeError, 'mean should be either a tuple or a np.ndarray'

    if input_scale != 1.0:
        blob *= input_scale

    return blob


def forward(net_json, params, ctx):
    '''
    process network forward propagation
    '''
    conv_layer = 'ch_concat_mixed_10_chconcat_output'
    prob_layer = 'softmax_output'
    symbol = mx.sym.load(net_json)
    internals = symbol.get_internals()
    symbol = mx.sym.Group([internals[prob_layer], internals[conv_layer]])
    save_dict = mx.nd.load(params)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        l2_tp, name = k.split(':', 1)
        if l2_tp == 'arg':
            arg_params[name] = v
        if l2_tp == 'aux':
            aux_params[name] = v
    mod = mx.model.FeedForward(symbol,
                               arg_params=arg_params,
                               aux_params=aux_params,
                               ctx=ctx,
                               allow_extra_params=False,
                               numpy_batch_size=1)

    return arg_params, mod


def main():
    arg_fc = 'fc1'
    mean = (128, 128, 128)
    raw_scale = 1.0
    input_scale = 1.0 / 128
    width = 299
    height = 299
    resize_size = 340
    ctx = mx.cpu(1) if int(args['--gpu']) == -1 else mx.gpu(int(args['--gpu']))
    category = [l.strip() for l in open(args['--label']).readlines()]

    arg_params, mod = forward(args['--network'], args['--weights'], ctx)
    weight_fc = arg_params[arg_fc + '_weight'].asnumpy()
    # bias_fc = arg_params[arg_fc+'_bias'].asnumpy()

    img = cv2.imread(args['<in-img>'])
    rgb = cv2.cvtColor(cv2.resize(img, (width, height)), cv2.COLOR_BGR2RGB)
    blob = im2blob(img, width, height, mean=mean, swap_channel=True,
                   raw_scale=raw_scale, input_scale=input_scale)
    outputs = mod.predict(blob)
    score = outputs[0][0]
    conv_fm = outputs[1][0]
    draw_cam(rgb, width, height, int(args['--top-k']), conv_fm,
             weight_fc, category, score, args['--display'])


if __name__ == "__main__":
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(
        _init_.__doc__, version='CAM on mxnet {}'.format(version))
    _init_()
    print('start cam job...')
    main()
    print('...done')
