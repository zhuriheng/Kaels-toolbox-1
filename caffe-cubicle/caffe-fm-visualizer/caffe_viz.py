#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2016/8/22 16:11 @Northrend
#
# Visualize internal filters
# of Convolutional Neural Networks


import ConfigParser
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

conf = ConfigParser.ConfigParser()
conf.read("./caffe_viz.conf")
print 'configuration file loaded successfully.'

# this file should be run from {caffe_root}/examples (otherwise change
# this line)
caffe_root = conf.get('path', 'caffe_root')
sys.path.insert(0, caffe_root + 'python')
import caffe

# display plots in this notebook
# %matplotlib inline
# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)   # large images
# don't interpolate: show square pixels
plt.rcParams['image.interpolation'] = 'nearest'
# use grayscale output rather than a (potentially misleading) color heatmap
plt.rcParams['image.cmap'] = 'gray'


# Convert a .protobinary file into a npy array
def protobinary2npy(protobinaryFilename):
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(protobinaryFilename, 'rb').read()
    blob.ParseFromString(data)
    npy = np.array(caffe.io.blobproto_to_array(blob))[0]
    return npy


def vis_square(data):
    """
    Take an array of shape (n, height, width) or (n, height, width, 3)
    and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
    """

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant',
                  constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape(
        (n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape(
        (n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    # plt.show()
    plt.axis('off')


if __name__ == '__main__':
    caffe.set_mode_gpu()
    model_def = caffe_root + conf.get('path', 'model_deploy')
    model_weights = caffe_root + conf.get('path', 'model_weights')
    net = caffe.Net(model_def,  # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = protobinary2npy(
        caffe_root + conf.get('path', 'mean_binary'))
    # average over pixels to obtain the mean (BGR) pixel values
    mu = mu.mean(1).mean(1)

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # move image channels to outermost dimension
    transformer.set_transpose('data', (2, 0, 1))
    # subtract the dataset-mean value in each channel
    transformer.set_mean('data', mu)
    # rescale from [0, 1] to [0, 255]
    transformer.set_raw_scale('data', 255)
    # swap channels from RGB to BGR
    transformer.set_channel_swap('data', (2, 1, 0))

    # set the size of the input (we can skip this if we're happy
    # with the default; we can also change it later, e.g., for different batch
    # sizes)
    net.blobs['data'].reshape(1,        # batch size
                              3,         # 3-channel (BGR) images
                              conf.getint('para', 'net_imgsize'),
                              conf.getint('para', 'net_imgsize'))  # image size X*X
    image = caffe.io.load_image(conf.get('path', 'test_pic'))
    transformed_image = transformer.preprocess('data', image)
    # plt.imshow(image)
    # plt.show()

    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image

    # perform classification
    output = net.forward()

    # the output probability vector for the first image in the batch
    output_prob = output['prob'][0]

    # load ImageNet labels
    labels_file = caffe_root + conf.get('path', 'label_synset')
    # if not os.path.exists(labels_file):
    # !.. / data / ilsvrc12 / get_ilsvrc_aux.sh

    labels = np.loadtxt(labels_file, str, delimiter='\t')

    print 'output label:', labels[output_prob.argmax()]

    # sort top five predictions from softmax output
    # reverse sort and take five largest items
    top_inds = output_prob.argsort()[::-1][:5]

    # print 'probabilities and labels:'
    # zip(output_prob[top_inds], labels[top_inds])

    # for each layer, show the output shape
    print 'blob shape:'
    for layer_name, blob in net.blobs.iteritems():
        print layer_name + '\t' + str(blob.data.shape)
    print 'param shape:'
    for layer_name, param in net.params.iteritems():
        # , str(param[1].data.shape)
        print layer_name + '\t' + str(param[0].data.shape)

    # visualize filters
    # the parameters are a list of [weights, biases]
    filters = net.params[conf.get('para', 'plt_param')][0].data
    # transpose matrix to match input dimensions
    filters = filters.transpose(0, 2, 3, 1)
    vis_square(filters)
    # conv2 filters
    # filters = net.params[conf.get('para', 'plt_param')][0].data
    # vis_square(filters[:8].reshape(8*48, 5, 5))
    # plt.savefig(conf.get('path', 'plt_save') +
    #             conf.get('para', 'plt_param') + '_filters.png')

    # visualize feats
    feat = net.blobs[conf.get('para', 'plt_feature')].data[0, conf.getint(
        'para', 'plt_feature_start'):conf.getint('para', 'plt_feature_end')]
    vis_square(feat)
    plt.savefig(conf.get('path', 'plt_save') +
                conf.get('para', 'plt_feature') + '_featmap.png')

    # feat = net.blobs[conf.get('para', 'plt_hist')].data[0]
    # plt.subplot(2, 1, 1)
    # plt.plot(feat.flat)
    # plt.subplot(2, 1, 2)
    # _ = plt.hist(feat.flat[feat.flat > 0], bins=100)
    # plt.savefig(conf.get('path', 'plt_save') +
    #             conf.get('para', 'plt_hist') + '_hist.png')

    # feat = net.blobs['prob'].data[0]
    # plt.figure(figsize=(15, 3))
    # plt.plot(feat.flat)

    print layer_name + '\t' + str(blob.data.shape)
    print 'predicted class is:', output_prob.argmax()
    print 'mean-subtracted values:', zip('BGR', mu)

    print 'Done.'
