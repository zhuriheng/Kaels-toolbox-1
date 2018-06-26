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
    Update: 2018-06-12 19:36:24 
    Author: @Northrend
    Contributor: 

    Change log:
    2018/06/12  v2.1                    convert detection annotations to oiv4 style 
    2018/03/28  v2.0                    adapt new ava json standard
    2018/02/08  v1.2                    fix json syntax bug
    2017/12/13  v1.1                    support detection
    2017/12/07  v1.0                    basic functions

    Usage:
    ava_jsonlist_adapter.py             <in-list> <out-file> [-c|--classification]
                                        [ -d|--detection -l|--clustering ]
                                        [--label=str --oiv4]
    ava_jsonlist_adapter.py             -v | --version
    ava_jsonlist_adapter.py             -h | --help

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
        --label=str                     path to index-classname mapping csv file                              
        --oiv4                          set to convert detection annotations to oiv4 style
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
    err_num = 0
    if args['--classification']:
        assert args['--label'], 'index-classname mapping file is required.'
        # assert args['--sub-task'], 'sub-task should be provided in classification.'
        category = get_category(args['--label'])
        for item in json_lst:
            try:
                temp_dict = json.loads(item.strip())
                img = os.path.basename(temp_dict['url'])
                label = category.index(temp_dict['label'][0]['data'][0]['class'])
            except:
                print('syntax error:',item.strip())
                err_num += 1
                continue
            output_file.write('{} {}\n'.format(img, label))
        print('err_num:', err_num)
    elif args['--detection']:
        if args['--oiv4']:
            output_file.write('ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n')
            for item in json_lst:
                temp_dict = json.loads(item.strip())
                img_id = os.path.basename(temp_dict['url'])
                try:
                    for instance in temp_dict['label'][0]['data']:
                        output_file.write('{},labelx,{},1,{},{},{},{},0,0,0,0,0\n'.format(img_id,instance['class'],instance['bbox'][0][0],instance['bbox'][2][0],instance['bbox'][0][1],instance['bbox'][2][1]))
                except:
                    print('syntax error or no object:',item.strip())
                    err_num += 1
                    continue

        else:
            dict_ann = dict()
            for item in json_lst:
                temp_dict = json.loads(item.strip())
                img = os.path.basename(temp_dict['url'])
                if img not in dict_ann:
                    dict_ann[img] = list()
                try:
                    for instance in temp_dict['label'][0]['data']:
                        ins_ann = list()
                        ins_ann.append(instance['bbox'][0][0])   # x1
                        ins_ann.append(instance['bbox'][0][1])   # y1
                        ins_ann.append(instance['bbox'][2][0])   # x2
                        ins_ann.append(instance['bbox'][2][1])   # y2
                        ins_ann.append(instance['class'])
                        dict_ann[img].append(ins_ann)
                except:
                    print('syntax error or no object:',item.strip())
                    err_num += 1
                    continue
            json.dump(dict_ann, output_file, indent=4)
        print('err_num:', err_num)
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
