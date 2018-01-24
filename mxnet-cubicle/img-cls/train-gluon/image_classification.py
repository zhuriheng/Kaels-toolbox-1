# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import division

import argparse, time
import logging
logging.basicConfig(level=logging.INFO)

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
from mxnet import autograd as ag
# from mxnet.test_utils import get_mnist_iterator

from data import *
from importlib import import_module 

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='dataset to use. options are mnist, cifar10, and dummy.')
parser.add_argument('--train-data', type=str, default='',
                    help='training record file to use, required for imagenet.')
parser.add_argument('--val-data', type=str, default='',
                    help='validation record file to use, required for imagenet.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--num-gpus', type=int, default=0,
                    help='number of gpus to use.')
parser.add_argument('--epochs', type=int, default=3,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate. default is 0.01.')
parser.add_argument('-momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed to use. Default=123.')
parser.add_argument('--mode', type=str,
                    help='mode in which to train the model. options are symbolic, imperative, hybrid')
parser.add_argument('--model', type=str, required=True,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--use_thumbnail', action='store_true',
                    help='use thumbnail or not in resnet. default is false.')
parser.add_argument('--batch-norm', action='store_true',
                    help='enable batch normalization or not in vgg. default is false.')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--kvstore', type=str, default='device',
                    help='kvstore to use for trainer/module.')
parser.add_argument('--log-interval', type=int, default=50, help='Number of batches to wait before logging.')
parser.add_argument('--profile', action='store_true',
                    help='Option to turn on memory profiling for front-end, '\
                         'and prints out the memory usage by python function at the end.')
opt = parser.parse_args()

logging.info(opt)

mx.random.seed(opt.seed)

dataset_classes = {'mnist': 10, 'cifar10': 10, 'imagenet': 1000, 'dummy': 3}

batch_size, dataset, classes = opt.batch_size, opt.dataset, dataset_classes[opt.dataset]

num_gpus = opt.num_gpus

batch_size *= max(1, num_gpus)
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

model_name = opt.model

kwargs = {'ctx': context, 'pretrained': opt.use_pretrained, 'classes': classes}
if model_name.startswith('resnet'):
    kwargs['thumbnail'] = opt.use_thumbnail
elif model_name.startswith('vgg'):
    kwargs['batch_norm'] = opt.batch_norm

network = import_module('symbols.mobilenet-v2')
net = network.MobilenetV2(2,1) 
# net.save('/disk2/Northrend/blued/models/blued_taster_v4/mobilenet-v2')
# net = models.get_model(opt.model, **kwargs)

def _get_lr_scheduler(lr):
    # if '--lr-factor' not in args or float(args['--lr-factor']) >= 1:
    #     return (args['--lr'], None)
    num_samples = 308000.0
    batch_size = 256.0
    epoch_size = int(num_samples/batch_size)
    begin_epoch = 0 
    step_epochs = [30,60,90] 
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= float(args['--lr-factor'])
    # if lr != float(args['--lr']):
    #     logging.info('Adjust learning rate to %e for epoch %d' %
    #                  (lr, begin_epoch))
    steps = [epoch_size * (x - begin_epoch)
             for x in step_epochs if x - begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=0.1)

def _get_iterators(data_train, data_dev, batch_size, data_shape=(3, 224, 224)):
    '''
    define the function which returns the data iterators
    '''
    # [mean_r, mean_g, mean_b] = [float(item) for item in args['--mean'].split(',')]
    # [std_r, std_g, std_b] = [float(item) for item in args['--std'].split(',')]
    mean_r, mean_g, mean_b = 123.68,116.779,103.939
    std_r, std_g, std_b = 58.395,57.12,57.375 
    logging.info('Input normalization params: mean_rgb {}, std_rgb {}'.format([mean_r, mean_g, mean_b],[std_r, std_g, std_b]))
    train = mx.io.ImageRecordIter(
        path_imgrec=data_train,
    # preprocess_threads=32,
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        # resize=int(args['--resize']),
        shuffle=True,
        rand_crop=True,
        rand_mirror=True,
        mean_r=mean_r,
        mean_g=mean_g,
        mean_b=mean_b,
        std_r=std_r,
        std_g=std_g,
        std_b=std_b
        )
    val = mx.io.ImageRecordIter(
        path_imgrec=data_dev,
    # preprocess_threads=32,
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        # resize=int(args['--resize']),
        shuffle=False,
        rand_crop=False,
        rand_mirror=False,
        mean_r=mean_r,
        mean_g=mean_g,
        mean_b=mean_b,
        std_r=std_r,
        std_g=std_g,
        std_b=std_b
        )
    logging.info("Data iters created successfully")
    return train, val

def get_data_iters(dataset, batch_size, num_workers=1, rank=0):
    # # get dataset iterators
    # if dataset == 'mnist':
    #     train_data, val_data = get_mnist_iterator(batch_size, (1, 28, 28),
    #                                               num_parts=num_workers, part_index=rank)
    # elif dataset == 'cifar10':
    #     train_data, val_data = get_cifar10_iterator(batch_size, (3, 32, 32),
    #                                                 num_parts=num_workers, part_index=rank)
    # elif dataset == 'imagenet':
    #     if model_name == 'inceptionv3':
    #         train_data, val_data = get_imagenet_iterator(opt.train_data, opt.val_data,
    #                                                      batch_size, (3, 299, 299),
    #                                                      num_parts=num_workers, part_index=rank)
    #     else:
    #         train_data, val_data = get_imagenet_iterator(opt.train_data, opt.val_data,
    #                                                      batch_size, (3, 224, 224),
    #                                                      num_parts=num_workers, part_index=rank)
    # elif dataset == 'dummy':
    if dataset == 'dummy':
        if model_name == 'inceptionv3':
            train_data, val_data = dummy_iterator(batch_size, (3, 299, 299))
        else:
            # train_data, val_data = dummy_iterator(batch_size, (3, 224, 224))
            train_data, val_data = _get_iterators(opt.train_data, opt.val_data, batch_size, (3, 224, 224)) 
    return train_data, val_data


def test(ctx, val_data):
    metric = mx.metric.Accuracy()
    val_data.reset()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
    return metric.get()


def train(epochs, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
    kv = mx.kv.create(opt.kvstore)
    train_data, val_data = get_data_iters(dataset, batch_size, kv.num_workers, kv.rank)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum, 'lr_scheduler': _get_lr_scheduler(opt.lr)},
                            kvstore = kv)
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    for epoch in range(epochs):
        tic = time.time()
        train_data.reset()
        metric.reset()
        btic = time.time()
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            Ls = []
            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)
                    L = loss(z, y)
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    outputs.append(z)
                for L in Ls:
                    L.backward()
            trainer.step(batch.data[0].shape[0])
            metric.update(label, outputs)
            if i!=0 and opt.log_interval and not (i)%opt.log_interval:
                name, acc = metric.get()
                logging.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f'%(
                               epoch, i, batch_size/(time.time()-btic), name, acc))
            btic = time.time()

        name, acc = metric.get()
        logging.info('[Epoch %d] training: %s=%f'%(epoch, name, acc))
        logging.info('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
        name, val_acc = test(ctx, val_data)
        logging.info('[Epoch %d] validation: %s=%f'%(epoch, name, val_acc))
        net.collect_params().save('/disk2/Northrend/blued/models/blued_taster_v4/{}-{:0>4}.params'.format(opt.model, (epoch+1)))

def main():
    if opt.mode == 'symbolic':
        data = mx.sym.var('data')
        out = net(data)
        softmax = mx.sym.SoftmaxOutput(out, name='softmax')
        mod = mx.mod.Module(softmax, context=[mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()])
        kv = mx.kv.create(opt.kvstore)
        train_data, val_data = get_data_iters(dataset, batch_size, kv.num_workers, kv.rank)
        mod.fit(train_data,
                eval_data = val_data,
                num_epoch=opt.epochs,
                kvstore=kv,
                batch_end_callback = mx.callback.Speedometer(batch_size, max(1, opt.log_interval)),
                epoch_end_callback = mx.callback.do_checkpoint('image-classifier-%s'% opt.model),
                optimizer = 'sgd',
                optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum},
                initializer = mx.init.Xavier(magnitude=2))
        mod.save_params('image-classifier-%s-%d-final.params'%(opt.model, opt.epochs))
    else:
        if opt.mode == 'hybrid':
            net.hybridize()
        train(opt.epochs, context)

if __name__ == '__main__':
    if opt.profile:
        import hotshot, hotshot.stats
        prof = hotshot.Profile('image-classifier-%s-%s.prof'%(opt.model, opt.mode))
        prof.runcall(main)
        prof.close()
        stats = hotshot.stats.load('image-classifier-%s-%s.prof'%(opt.model, opt.mode))
        stats.strip_dirs()
        stats.sort_stats('cumtime', 'calls')
        stats.print_stats()
    else:
        main()

