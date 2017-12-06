#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2017/05/25 @Northrend
#
# log visualizer script
# for mxnet
#

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
import docopt
import os
import logging

# init global logger
log_format = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger()


COLOR_LST = ['k', 'y', 'm', 'c', 'g', 'b', 'r']     # Preset Color List

# Preset Regular Expressions
VA_TACC = re.compile('.*?]\sTrain-accuracy=([\w\.]+)')
VA_TACC_K = re.compile('.*?]\sTrain-top_k_accuracy_\d=([\w\.]+)')
VA_VACC = re.compile('.*?]\sValidation-accuracy=([\w\.]+)')
VA_VACC_K = re.compile('.*?]\sValidation-top_k_accuracy_\d=([\w\.]+)')
TR_ACC = re.compile('.*?\saccuracy=([\d\.]+)')
TR_ACC_K = re.compile('.*?\stop_k_accuracy_5=([\d\.]+)')
TR_ACC_CE = re.compile('.*?\scross-entropy=([\d\.]+)')
SPEED = re.compile('.*?\sSpeed: ([\d\.]+)')
BATCH = re.compile('.*?Epoch\[([\d]+)\] Batch \[([\d]+)\]\sSpeed')


def _init_():
    '''
    Draw curves of mxnet training log 
    Update: 2017/08/14
    Author: @Northrend
    Contributor: 

    Change log:
    2017/08/14  v2.0    apply logging module
    2017/07/10  v1.4    add save name prefix of result pics
    2017/07/06  v1.3    support cross entropy curve
    2017/06/21  v1.2    fix no top-k log bug
    2017/05/25  v1.1    fix BATCH regular expression
    2017/05/23  v1.0    basic functions

    Usage:
        mxnet_viz.py                <log> <output-dir> [--log-lv=str]
                                    [-dtvsm --top-k --ce --max-batch=int --logs=list]
                                    [--log-prefix=str]
        mxnet_viz.py                --version
        mxnet_viz.py                -h | --help

    Arguments:
        <log>                       input mxnet log
        <output-dir>                output directory

    Options:
        -h --help                   show this help screen
        --version                   show current version
        ---------------------------------------------------------------------------------------------------
        -d                          display mode
        -t                          draw training curve
        -v                          draw validation curve
        -s                          draw speed curve
        -m                          draw multi-curves
        --log-lv=str                logging level, one of INFO DEBUG WARNING ERROR CRITICAL [default: INFO]
        --logs=list                 resume input mxnet logs，use ',' to split
        --log-prefix=str            logs path prefix, if needed [default: '']
        --ce                        set to read cross entropy log
        --top-k                     set to read top-k acc log
        --max-batch=int             max logged iteration of each epoch
    '''
    logger.setLevel(eval('logging.{}'.format(args['--log-lv'])))
    logger.info('=' * 80 + '\nArguments submitted:')
    for key in sorted(args.keys()):
        logger.info('{:<20}= {}'.format(key.replace('--', ''), args[key]))
    logger.info('=' * 80)


def draw_curve_by_epoch(epoch, dict_y, label_y, output='./tmp.png', fsize=(16, 10)):
    color_lst = COLOR_LST[:]
    (max_y, min_y) = ('flag', 'flag')
    plt.figure(figsize=fsize)
    plt.xlabel("Epoch")
    plt.ylabel(label_y)

    for key in dict_y.keys():
        plt.plot(epoch, dict_y[key], 'o', linestyle='-', color=color_lst.pop(),
                 linewidth=0.8, label=key)
        if (max_y, min_y) == ('flag', 'flag'):
            max_y, min_y = max(dict_y[key]), min(dict_y[key])
        else:
            max_y, min_y = max(max_y, max(dict_y[key])), min(
                min_y, min(dict_y[key]))

    # config
    plt.grid(color='grey', linestyle=':')
    plt.xticks(np.arange(0, max(epoch) * 1.1, max(epoch) * 0.1))
    plt.xlim([-max(epoch) * 0.05, max(epoch) * 1.05])
    plt.yticks(np.arange(min_y, 1.1 * max_y -
                         0.1 * min_y, (max_y - min_y) * 0.1))
    plt.yticks(np.arange(min_y, 1.1 * max_y -
                         0.1 * min_y, (max_y - min_y) * 0.1))
    plt.ylim([1.05 * min_y - 0.05 * max_y, 1.05 * max_y - 0.05 * min_y])
    plt.legend(loc="best")
    if args['-d']:
        plt.show()
    plt.savefig(output)
    plt.close()


def draw_curve_by_batch(batch, batch_size, dict_y, label_y, output='./tmp.png', fsize=(16, 10)):
    color_lst = COLOR_LST[:]
    (max_y, min_y) = ('flag', 'flag')
    plt.figure(figsize=fsize)
    plt.xlabel("Batch")
    plt.ylabel(label_y)

    # get batches with epoch accumulated
    batch_lst = []
    for e, b in batch:
        batch_lst.append(int(e) * int(batch_size) + int(b))

    # draw curves
    for key in dict_y.keys():
        plt.plot(batch_lst, dict_y[key], linestyle='-', color=color_lst.pop(),
                 linewidth=0.8, label=key)
        if (max_y, min_y) == ('flag', 'flag'):
            max_y, min_y = max(dict_y[key]), min(dict_y[key])
        else:
            max_y, min_y = max(max_y, max(dict_y[key])), min(
                min_y, min(dict_y[key]))

    # config
    plt.grid(color='grey', linestyle=':')
    plt.xticks(np.arange(0, max(batch_lst) * 1.1, max(batch_lst) * 0.1))
    plt.xlim([-max(batch_lst) * 0.05, max(batch_lst) * 1.05])
    plt.yticks(np.arange(min_y, 1.1 * max_y -
                         0.1 * min_y, (max_y - min_y) * 0.1))
    plt.ylim([1.05 * min_y - 0.05 * max_y, 1.05 * max_y - 0.05 * min_y])
    plt.legend(loc="best")
    if args['-d']:
        plt.show()
    plt.savefig(output)
    plt.close()


def draw_multi_curves_by_batch(batch, batch_size, lst_dict_y, label_y, output='./tmp.png', fsize=(16, 10), lst_legend=list()):
    color_lst = COLOR_LST[:]
    (max_y, min_y) = ('flag', 'flag')
    plt.figure(figsize=fsize)
    plt.xlabel("Batch")
    plt.ylabel(label_y)

    # get batches with epoch accumulated
    batch_lst = []
    for e, b in batch:
        batch_lst.append(int(e) * int(batch_size) + int(b))

    # draw curves
    for index, dict_y in enumerate(lst_dict_y):
        for key in dict_y.keys():
            plt.plot(batch_lst, dict_y[key], linestyle='-', color=color_lst.pop(),
                     linewidth=0.8, label='{}: {}'.format(lst_legend[index], key))
            if (max_y, min_y) == ('flag', 'flag'):
                max_y, min_y = max(dict_y[key]), min(dict_y[key])
            else:
                max_y, min_y = max(max_y, max(dict_y[key])), min(
                    min_y, min(dict_y[key]))

    # config
    plt.grid(color='grey', linestyle=':')
    plt.xticks(np.arange(0, max(batch_lst) * 1.1, max(batch_lst) * 0.1))
    plt.xlim([-max(batch_lst) * 0.05, max(batch_lst) * 1.05])
    plt.yticks(np.arange(min_y, 1.1 * max_y -
                         0.1 * min_y, (max_y - min_y) * 0.1))
    plt.ylim([1.05 * min_y - 0.05 * max_y, 1.05 * max_y - 0.05 * min_y])
    plt.legend(loc="best")
    if args['-d']:
        plt.show()
    plt.savefig(output)
    plt.close()


def main():
    log = open(args['<log>']).read()
    basename = os.path.splitext(os.path.basename(args['<log>']))[0]

    # draw validation curve
    if args['-v']:
        dict_val = {}
        dict_val['Training Acc'] = [float(x) for x in VA_TACC.findall(log)]
        if args['--top-k']:
            dict_val['Training Acc Top-k'] = [float(x)
                                              for x in VA_TACC_K.findall(log)]
        dict_val['Validation Acc'] = [float(x) for x in VA_VACC.findall(log)]
        if args['--top-k']:
            dict_val['Validation Acc Top-k'] = [float(x)
                                                for x in VA_VACC_K.findall(log)]
        epoch = np.arange(len(dict_val['Training Acc']))
        assert len(epoch) != 0, logger.error(
            'No validation log, check your log file')
        logger.debug('epoch:{}'.format(epoch))
        logger.debug('dict_train:{}'.format(dict_val))
        draw_curve_by_epoch(epoch, dict_val, 'Accuracy',
                            output=args['<output-dir>'] + basename + '-val.png')
        logger.info('Validation curve saved.')

    # draw training curve
    if args['-t']:
        assert args['--max-batch'], logger.debug('Max batch needed!')
        batch = [(int(x), int(y)) for (x, y) in BATCH.findall(log)]
        dict_train = dict()
        dict_train['Training Acc'] = [float(x) for x in TR_ACC.findall(log)]
        if args['--top-k']:
            dict_train['Training Acc Top-k'] = [float(x)
                                                for x in TR_ACC_K.findall(log)]
        logger.debug('batch:{}'.format(batch))
        logger.debug('dict_train:{}'.format(dict_train))
        draw_curve_by_batch(batch, int(args['--max-batch']), dict_train,
                            'Accuracy', output=args['<output-dir>'] + basename + '-train.png')
        if args['--ce']:
            dict_train = dict()
            dict_train['Training Cross Entropy'] = [float(x)
                                                    for x in TR_ACC_CE.findall(log)]
        logger.debug('batch:{}'.format(batch))
        logger.debug('dict_train:{}'.format(dict_train))
        draw_curve_by_batch(batch, int(args['--max-batch']), dict_train,
                            'Loss', output=args['<output-dir>'] + basename + '-train-loss.png')
        logger.info('Training curve saved.')

    # draw multi training curves
    if args['-m']:
        assert args['--max-batch'], 'Max batch needed!'
        batch = [(int(x), int(y)) for (x, y) in BATCH.findall(log)]
        lst_log = [log]
        lst_legend = [os.path.basename(args['<log>'])]
        path_log = args['--logs'].split(',')
        for path in path_log:
            lst_log.append(
                open(os.path.join(args['--log-prefix'], path)).read())
            lst_legend.append(os.path.basename(
                os.path.join(args['--log-prefix'], path)))
        lst_train = list()
        for str_log in lst_log:
            dict_train = dict()
            dict_train['Training Acc'] = [
                float(x) for x in TR_ACC.findall(str_log)]
            if args['--top-k']:
                dict_train['Training Acc Top-k'] = [float(x)
                                                    for x in TR_ACC_K.findall(str_log)]
            lst_train.append(dict_train)
        draw_multi_curves_by_batch(batch, int(args['--max-batch']), lst_train,
                                   'Accuracy', output=args['<output-dir>'] + basename + '-multi-train.png', lst_legend=lst_legend)
        logger.info('Multi-training curve saved.')

    # draw speed curve
    if args['-s']:
        assert args['--max-batch'], 'Max batch needed!'
        dict_speed = {}
        dict_speed['Processing Speed'] = [float(x) for x in SPEED.findall(log)]
        draw_curve_by_batch(batch, int(args['--max-batch']), dict_speed,
                            'Speed', output=args['<output-dir>'] + basename + '-speed.png')
        logger.info('Speed curve saved.')


if __name__ == '__main__':
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(
        _init_.__doc__, version='MXNet visualizer {}'.format(version))
    _init_()
    logger.info('Start drawing curves...')
    main()
    logger.info('...Done')
