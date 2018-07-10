#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2018/03/13 @Northrend
#
# Generate AVA-standard jsonlist
#


from __future__ import print_function
import os
import sys
import json
import re
import docopt


def _init_():
    '''
    Script for generating AVA-standard jsonlist file
    Update: 2018/03/13
    Author: @Northrend
    Contributor: 

    Change log:
    2018/05/03   v1.1        support detection 
    2018/03/13   v1.0        basic functions

    Usage:
        gen_ava_jsonlist.py          <in-file> <out-list> 
                                     [ -c | --classification]
                                     [ -d | --detection ]
                                     [ -l | --clustering ]
                                     [--prefix=str --sub-task=str --pre-json=str --pre-label=str]
        gen_ava_jsonlist.py          -v | --version
        gen_ava_jsonlist.py          -h | --help

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
        --sub-task=str                  sub-task type such as general, pulp, terror, places.
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
    temp['ops'] = 'download()'
    # temp['source_url'] = temp['url']
    temp['type'] = 'image'
    temp['label'] = list() 
    pulp_label = ['pulp', 'sexy', 'normal']
    if classification:
        tmp = dict()
        tmp['type'] = 'classification'
        tmp['version'] = '1'
        tmp['name'] = sub_task
        tmp['data'] = list()
        if pre_ann:
            # ---- modify pre-annotated label here ----
            # temp['label']['class'][sub_task] = pre_ann[filename]
            pass
            # -----------------------------------------
        elif pre_label:
            # tmp['data'].append({'class': pulp_label[pre_label]})
            tmp['data'].append({'class': pulp_label[int(pre_label)]})
        temp['label'].append(tmp)
    if detection:
        tmp = dict()
        tmp['type'] = 'detection'
        tmp['version'] = '1'
        tmp['name'] = sub_task
        tmp['data'] = list()
        if pre_ann:
            pass
            # ---- modify pre-annotated label here ----
            # temp['label']['detect']['general_d'] = pre_ann[filename]
            # -----------------------------------------
        temp['label'].append(tmp)
    # if clustering:
    #     assert pre_ann, 'pre-annotation file should be provided under clustering task.'
    #     # ---- modify pre-annotated label here ----
    #     temp['label']['facecluster'] = pre_ann[filename]
    #     # -----------------------------------------
    
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
            if len(buff.strip().split()) == 1:      # input syntax error
                file_lst.append(buff.strip())
            elif len(buff.strip().split()) == 2:
                file_lst.append(buff.strip().split())
            else:
                raise input_syntax_err
            
    with open(args['<out-list>'], 'w') as f:
        for image,label in file_lst:
            if len(image.strip().split()) == 2:
                pre_label = image.strip().split()[1]
            temp_dict = generate_dict(image.split()[0], args['--prefix'], args['--classification'],
                                      args['--detection'], args['--clustering'], sub_task, pre_ann, pre_label=label)
            f.write('{}\n'.format(json.dumps(temp_dict)))


if __name__ == "__main__":
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(
        _init_.__doc__, version='LabelX jsonlist generator {}'.format(version))
    _init_()
    print('Start generating jsonlist...')
    main()
    print('...done')
