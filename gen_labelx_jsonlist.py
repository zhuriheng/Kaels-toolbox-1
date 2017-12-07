#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2017/12/06 @Northrend
#
# Generate LabelX-standard jsonlist
#


from __future__ import print_function
import os
import sys
import json
import re
import docopt


def _init_():
    '''
    Script for generating labelX-standard jsonlist file
    Update: 2017/12/06
    Author: @Northrend
    Contributor: 

    Change log:
    2017/12/06  v1.0        basic functions

    Usage:
        gen_labelx_jsonlist.py          <in-file> <out-list> 
                                        [-c|--classification -d|--detection -l|--clustering]
                                        [--prefix=str --sub-task=str --pre-json=str --pre-label=str]
        gen_labelx_jsonlist.py          -v | --version
        gen_labelx_jsonlist.py          -h | --help

    Arguments:
        <in-file>                       input list path
        <out-list>                      output jsonlit file path

    Options:
        -h --help                       show this help screen
        -v --version                    show current version
        -c --classification             classification task mode
        -d --detection                  detection task mode
        -l --clustering                 clustering task mode
        -------------------------------------------------------------------------------------------
        --prefix=str                    prefix of each url, such as bucket-domain.
        --sub-task=str                  classification sub-task type, shouled be choosen from 
                                        [general, pulp, terror, places].
        --pre-json=str                  optional pre-annotation json, required under clusering task.
        --pre-label=str                 optional pre-annotation label, such as "cat".
    '''
    print('=' * 80 + '\nArguments submitted:')
    for key in sorted(args.keys()):
        print('{:<20}= {}'.format(key.replace('--', ''), args[key]))
    print('=' * 80)


class input_syntax_err(Exception):
    '''
    Catch input file-list syntax error
    '''
    pass


def generate_dict(filename, prefix, classification=False, detection=False, clustering=False, sub_task=None, pre_ann=None, pre_label=None):
    temp = dict()
    temp['url'] = prefix + filename if prefix else filename
    temp['source_url'] = temp['url']
    temp['type'] = 'image'
    temp['label'] = dict()
    if classification:
        temp['label']['class'] = dict()
        if pre_ann:
            # ---- modify pre-annotated label here ----
            temp['label']['class'][sub_task] = pre_ann[filename]
            # -----------------------------------------
        elif pre_label:
            temp['label']['class'][sub_task] = pre_label
    if detection:
        temp['label']['detect'] = {'general_d': {}}
        if pre_ann:
            # ---- modify pre-annotated label here ----
            temp['label']['detect']['general_d'] = pre_ann[filename]
            # -----------------------------------------
    if clustering:
        assert pre_ann, 'pre-annotation file should be provided under clustering task.'
        # ---- modify pre-annotated label here ----
        temp['label']['facecluster'] = pre_ann[filename]
        # -----------------------------------------
    return temp


def main():
    '''
    Generate labelX standard json.
    '''
    sub_task = args['--sub-task'] if args['--sub-task'] else 'general'
    pre_ann = json.load(open(args['--pre-json'], 'r')
                        ) if args['--pre-json'] else None
    pre_label = args['--pre-label'] if args['--pre-label'] else None
    with open(args['<in-file>'], 'r') as f:         # load input file list
        file_lst = list()
        for buff in f:
            if len(buff.strip().split()) != 1:      # input syntax error
                raise input_syntax_err
            file_lst.append(buff.strip())
    with open(args['<out-list>'], 'w') as f:
        for image in file_lst:
            temp_dict = generate_dict(image, args['--prefix'], args['--classification'],
                                      args['--detection'], args['--clustering'], sub_task, pre_ann)
            f.write('{}\n'.format(json.dumps(temp_dict)))


if __name__ == "__main__":
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(
        _init_.__doc__, version='LabelX jsonlist generator {}'.format(version))
    _init_()
    print('Start generating jsonlist...')
    main()
    print('...done')
