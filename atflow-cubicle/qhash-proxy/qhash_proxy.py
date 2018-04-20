#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2018/03/01 @Northrend
#
# get file hash with multi-thread
#

from __future__ import print_function
import os
import requests
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
    Getting remote file hash with multi-thread
    Update: 2018/04/20
    Contributor: laojiangwei@github.com 

    Change log:
    2018/04/20          v1.1            fix bug 
    2018/03/01          v1.0            basic functions

    Usage:
        qhash_proxy.py                  <infile> <thread-number>
                                        [--prefix=str --output=str --hash-alg=lst]
        qhash_proxy.py                  -v|--version
        qhash_proxy.py                  -h|--help

    Arguments:
        <infile>                        input file list
        <thread-number>                 number of processing thread

    Options:
        -h --help                       show this screen
        -v --version                    show script version
        ------------------------------------------------------------------------------------------
        --hash-alg=lst                  hash algorithm, md5, sha1 or both [default: md5]
        --output=str                    output json file path, will be saved as <infile>_hash.json path by default
        --prefix=str                    add url prefix if needed
    """
    print('=' * 80 + '\nArguments submitted:')
    for key in sorted(args.keys()):
        print ('{:<20}= {}'.format(key.replace('--', ''), args[key]))
    print('=' * 80)


class request_err(Exception):
    '''
    Catch qhash request error 
    '''
    pass


class prod_worker(threading.Thread):
    """
    producing worker
    """
    global GLOBAL_LOCK

    def __init__(self, queue, infile):
        threading.Thread.__init__(self)
        self.queue = queue
        self.infile = infile

    def run(self):
        for buff in self.infile:
            temp = dict()
            # skip blank line
            if not buff.strip():
                continue
            file_tmp = buff.strip().split()[0]
            GLOBAL_LOCK.acquire()
            self.queue.put(file_tmp)
            GLOBAL_LOCK.release()
        GLOBAL_LOCK.acquire()
        print('thread:', self.name, 'successfully quit')
        GLOBAL_LOCK.release()


class cons_worker(threading.Thread):
    global GLOBAL_LOCK

    def __init__(self, queue, hash_dic, hash_alg_list, prefix=None):
        threading.Thread.__init__(self)
        self.queue = queue
        self.hash_dic = hash_dic
        self.hash_alg_list = hash_alg_list
        self.prefix = prefix

    def get_qhash(self, url, alg, err_num):
        req = '{}?qhash/{}'.format(url, alg)
        ret = requests.get(req)
        if ret.status_code != 200:
            # raise request_err
            print('return error:', req)
            err_num += 1
            return None, err_num
        else:
            return json.loads(ret.text), err_num

    def run(self):
        global ERROR_NUMBER
        err_num = 0
        while(not self.queue.empty()):
            if GLOBAL_LOCK.acquire(False):
                # customized code
                file_tmp = self.queue.get()
                print('processing:', file_tmp)
                self.hash_dic[file_tmp] = dict()
                GLOBAL_LOCK.release()
                if self.prefix:
                    url_tmp = os.path.join(self.prefix, file_tmp)
                else:
                    url_tmp = file_tmp
                for hash_alg in self.hash_alg_list:
                    try:
                        result, err_num = self.get_qhash(url_tmp, hash_alg, err_num)
                    except requests.exceptions.ConnectionError:
                        GLOBAL_LOCK.acquire()
                        self.queue.put(file_tmp)
                        GLOBAL_LOCK.release()
                        break
                    if result:
                        GLOBAL_LOCK.acquire()
                        self.hash_dic[file_tmp][hash_alg] = result['hash']
                        GLOBAL_LOCK.release()
            else:
                pass
        GLOBAL_LOCK.acquire()
        ERROR_NUMBER += err_num
        print('thread:', self.name, 'successfully quit')
        GLOBAL_LOCK.release()


def main():
    infile = open(args['<infile>'], 'r')
    output = args['--output'] if args['--output'] else '{}_hash.json'.format(os.path.splitext(args['<infile>'])[0])
    thread_count = int(args['<thread-number>'])
    prefix = args['--prefix']
    queue = Queue.Queue(0)
    hash_dic = dict()
    hash_alg_list = args['--hash-alg'].split(',')
    for check in hash_alg_list:
        assert check in ['md5', 'sha1'], 'invalid hash algorithm: {}'.format(check)
    thread_prod = prod_worker(queue, infile)
    thread_prod.start()
    print('thread:', thread_prod.name, 'successfully started')
    time.sleep(1)
    tic = time.time()
    for i in xrange(thread_count):
        exec('thread_cons_{} = cons_worker(queue, hash_dic, hash_alg_list, prefix)'.format(i))
        eval('thread_cons_{}.start()'.format(i))
    thread_prod.join()
    for i in xrange(thread_count):
        eval('thread_cons_{}.join()'.format(i))
    toc = time.time()    
    print('total error number:', ERROR_NUMBER)
    print('processing time:', (toc-tic),'s')
    infile.close()
    with open(output, 'w') as f:
        json.dump(hash_dic, f, indent=4)


if __name__ == '__main__':
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(_init_.__doc__, version='Qhash-proxy {}'.format(
        version), argv=None, help=True, options_first=False)
    _init_()
    print('start processing...')
    main()
    print('...done')
