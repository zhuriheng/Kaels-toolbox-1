#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2017/11/20 @Northrend
#
# Multi-threading downloader
#

from __future__ import print_function
import os
import commands
import time
import re
import threading
import Queue
import json
import docopt

# globale vars initialization
GLOBAL_LOCK = threading.Lock()
ERROR_NUMBER = 0
FILE_NAME = str()


def _init_():
    """
    Multi-threading downloader script

    Change log:
    2018/03/05      v1.3        fix bug
    2018/02/27      v1.2        update saving mapping-file feature
    2018/02/26      v1.1        support save as original basename
    2017/11/23      v1.0        basic functions

    Usage:
        download_from_urls.py           <infile> <thread-number> [-b|--basename] 
                                        [--mapfile-path=str --date=str --start-index=int]
                                        [--download-path=str --prefix=str --suffix=str --ext=str]
        download_from_urls.py           -v|--version
        download_from_urls.py           -h|--help

    Arguments:
        <infile>                        input original url list
        <thread-number>                 number of downloading thread

    Options:
        -h --help                       show this screen
        -v --version                    show script version
        -b --basename                   set to save as original basename
        ------------------------------------------------------------------------------------------
        --date=str                      date mark in filename [default: 1123]
        --start-index=int               start index in filename [default: 0]
        --download-path=str             path to save result file [default: ./]
        --mapfile-path=str              path to save mapping files
        --prefix=str                    prefix of filename
        --suffix=str                    suffix of fileanme, supposed to be begin with '_' 
        --ext=str                       extension of filename [default: jpg]
    """
    print('=' * 80 + '\nArguments submitted:')
    for key in sorted(args.keys()):
        print ('{:<20}= {}'.format(key.replace('--', ''), args[key]))
    print('=' * 80)


class prod_worker(threading.Thread):
    """
    producing worker
    """
    global GLOBAL_LOCK

    def __init__(self, queue, infile, f2u, u2f, use_basename=False):
        threading.Thread.__init__(self)
        self.queue = queue
        self.infile = infile
        self.f2u = f2u
        self.u2f = u2f
        self.use_basename = use_basename

    def run(self):
        i = int(args['--start-index'])
        for buff in self.infile:
            temp = dict()
            # skip blank line
            if not buff.strip():
                continue
            temp['url'] = buff.strip().split()[0]
            if self.use_basename:
                temp['filename'] = os.path.join(
                    args['--download-path'], os.path.basename(temp['url']))
            else:
                temp['filename'] = os.path.join(
                    args['--download-path'], FILE_NAME.format(args['--date'], i))
            self.f2u[os.path.basename(temp['filename'])] = temp['url']
            self.u2f[temp['url']] = os.path.basename(temp['filename'])
            GLOBAL_LOCK.acquire()
            self.queue.put(temp)
            i += 1
            # print(temp, 'put into queue by', self.name)
            GLOBAL_LOCK.release()
        GLOBAL_LOCK.acquire()
        print('thread:', self.name, 'successfully quit')
        GLOBAL_LOCK.release()


class cons_worker(threading.Thread):
    global GLOBAL_LOCK

    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue

    def download(self, url, output_path, err_num):
        try:
            # commands.getoutput('wget -O {} {}'.format(output_path, url))
            if os.system('wget -O {} {}'.format(output_path, url)) != 0:
                print('download error:', url)
                err_num += 1
        except all as e:
            print('download error:', url)
            err_num += 1
        return err_num

    def run(self):
        global ERROR_NUMBER
        err_num = 0
        while(not self.queue.empty()):
            if GLOBAL_LOCK.acquire(False):
                # customized downloading code
                temp = self.queue.get()
                print('downloading: ',temp['url'])
                GLOBAL_LOCK.release()
                err_num = self.download(temp['url'], temp['filename'], err_num)
                # time.sleep(5)
                # print(temp, 'poped from queue by', self.name)
            else:
                pass
        GLOBAL_LOCK.acquire()
        ERROR_NUMBER += err_num
        print('thread:', self.name, 'successfully quit')
        GLOBAL_LOCK.release()


def filename_init():
    """
    initialize file name template
    """
    global FILE_NAME
    args['--prefix'] = "" if not args['--prefix'] else args['--prefix']
    args['--suffix'] = "" if not args['--suffix'] else args['--suffix']
    FILE_NAME = args['--prefix'] + '_{}_{:0>8}' + \
        args['--suffix'] + '.' + args['--ext']
    if not args['--basename']:
        print('files will be saved as:', FILE_NAME.format(args['--date'], 0))


def main():
    infile = open(args['<infile>'], 'r')
    f2u = dict()
    u2f = dict()
    filename_init()
    thread_count = int(args['<thread-number>'])
    queue = Queue.Queue(0)
    thread_prod = prod_worker(queue, infile, f2u, u2f, args['--basename'])
    thread_prod.start()
    print('thread:', thread_prod.name, 'successfully started')
    time.sleep(1)
    for i in xrange(thread_count):
        exec('thread_cons_{} = cons_worker(queue)'.format(i))
        eval('thread_cons_{}.start()'.format(i))
    thread_prod.join()
    for i in xrange(thread_count):
        eval('thread_cons_{}.join()'.format(i))
    print('total error number:', ERROR_NUMBER)
    infile.close()
    if args['--mapfile-path']:
        with open(os.path.join(args['--mapfile-path'],'f2u.json'), 'w') as f:
            json.dump(f2u, f, indent=4)
        with open(os.path.join(args['--mapfile-path'],'u2f.json'), 'w') as u:
            json.dump(u2f, u, indent=4)
    else:
        pass


if __name__ == '__main__':
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(_init_.__doc__, version='Multi-threading downloader {}'.format(
        version), argv=None, help=True, options_first=False)
    _init_()
    print('start downloading...')
    main()
    print('...done')

