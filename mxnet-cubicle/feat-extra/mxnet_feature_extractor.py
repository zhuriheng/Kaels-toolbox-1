# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import os
import time
import cv2
import mxnet as mx
import pprint
from collections import namedtuple
import numpy as np
import docopt


def net_init(model_prefix,model_epoch,gpu=0,output_layer='flatten0_output',batch_size=1,image_width=224):
    '''
    initialize mxnet model
    '''
    # get compute graph
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, model_epoch)   # load original model
    output_layer = sym.get_internals()['flatten0_output']
    # bind module with graph
    model = mx.mod.Module(symbol=output_layer, context=mx.gpu(gpu), label_names=None)
    model.bind(for_training=False, data_shapes=[('data', (batch_size, 3, image_width, image_width))], label_shapes=model._label_shapes)

    # load model parameters
    model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)
    return model


def extra_feature(model, image_path):
    resize_width = 224
    mean_r, mean_g, mean_b = 123.68, 116.779, 103.939
    std_r, std_g, std_b = 58.395, 57.12, 57.375
    Batch = namedtuple('Batch', ['data'])
    img_read = cv2.imread(image_path)
    if np.shape(img_read) == tuple():
        return None
    img = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
    img = img.astype(float)
    img = cv2.resize(img, (resize_width, resize_width))
    img -= [mean_r, mean_g, mean_b]
    img /= [std_r, std_g, std_b]
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img_batch = mx.nd.array(np.zeros((1,3,224,224)))
    img_batch[0] = mx.nd.array(img)
    
    model.forward(Batch([img_batch]))
    output = model.get_outputs()[0].asnumpy()
    # print('output.shape:',output.shape)
    return output

def main():
    '''
    :inputs: /path/to/model/prefix epoch /path/to/images.lst /path/to/result.npy /path/to/image/prefix
    '''
    # root_path = './test-images/'
    root_path = sys.argv[5] 
    with open(sys.argv[3],'r') as f:
        images = [os.path.join(root_path,x.strip()) for x in f.readlines()]
    image_number = len(images)
    feature = np.zeros((image_number,2048))
    model = net_init(sys.argv[1],int(sys.argv[2]),gpu=7)
    tic_0 = time.time()
    for i in xrange(image_number):
	tic = time.time()
        output = extra_feature(model, images[i])
        print('Batch [{}]: {:.4f}s'.format(i, time.time()-tic))
        if np.shape(output) != tuple():
            feature[i] = output
    print('Total time: {:.4f}s'.format(time.time()-tic_0))
    try:
        np.save(sys.argv[4], feature)
    except:
        np.save('./tmp.npy', feature)
        print('Saving failed, result file saved to ./tmp.npy')
    print('...done')
    # print('==> Original params:')
    # pprint.pprint(zip(sym.list_arguments(), sym.infer_shape(data=(1, 3, 224, 224))[0]))
    # sym_new, arg_new = modify_net(sym, arg_params)
    # mx.model.save_checkpoint(sys.argv[1] + '-modified', 0, sym_new, arg_new, aux_params)
    # print('New model saved at: {}'.format(sys.argv[1] + '-modified'))


if __name__ == '__main__':
    main()
