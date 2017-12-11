#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2017/12/07 @Northrend
#
# Convert LabelX-standard jsonlist to groundtruth file
#


from __future__ import print_function
import os
import sys
import json
import re
import docopt


def _init_():
    '''
    Script for converting labelX-standard jsonlist to groundtruth file
    Update: 2017/12/07
    Author: @Northrend
    Contributor: 

    Change log:
    2017/12/07  v1.0          basic functions

    Usage:
    labelx_jsonlist_adapter.py          <in-list> <out-file> 
                                        [-c|--classification -d|--detection -l|--clustering]
                                        [--sub-task=str --label=str]
    labelx_jsonlist_adapter.py          -v | --version
    labelx_jsonlist_adapter.py          -h | --help

    Arguments:
        <in-list>                       input json list path
        <out-file>                      output groundtruth file path

    Options:
        -h --help                       show this help screen
        -v --version                    show current version
        -c --classification             classification task mode
        -d --detection                  detection task mode
        -l --clustering                 clustering task mode
        -------------------------------------------------------------------------------------------
        --sub-task=str                  classification sub-task type, shouled be choosen from 
                                        [general, pulp, terror, places].
        --label=str                     path to index-classname mapping csv file                              
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


def get_category(category_path):
    label_list = list()
    with open(category_path, 'r') as f:
        for buff in f:
            label_list.append(tuple(buff.strip().split(',')))
        label_list_sort = sorted(label_list, key=lambda x: x[0])
    category = [item[1] for item in label_list_sort]
    return category


def main():
    input_file = open(args['<in-list>'], 'r')
    output_file = open(args['<out-file>'], 'w')
    json_lst = input_file.readlines()
    assert args['--label'], 'index-classname mapping file is required.'
    if args['--classification']:
        assert args['--sub-task'], 'sub-task should be provided in classification.'
        category = get_category(args['--label'])
        for item in json_lst:
            temp_dict = json.loads(item.strip())
            img = os.path.basename(temp_dict['url'])
            try:
                label = category.index(temp_dict['label']['class'][args['--sub-task']])
            except:
                print('syntax error:',item)
                continue
            output_file.write('{} {}\n'.format(img, label))
    output_file.close()
    input_file.close()


if __name__ == "__main__":
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(
        _init_.__doc__, version='LabelX jsonlist adapter {}'.format(version))
    _init_()
    print('Start converting jsonlist...')
    main()
    print('...done')
