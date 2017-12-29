#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2017/12/27 @Northrend
#
# Check md5 of files and get duplicates
# Support Multi-thread processing
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
import hashlib
# import md5
import pprint


# globale vars initialization
GLOBAL_LOCK = threading.Lock()
REMOVE_NUMBER = 0


def _init_():
    """
    Check files md5 with multi-thread processing

    Change log:
    2017/12/27      v1.0        basic functions

    Usage:
        checkout_md5.py                 <infile> <outfile> <thread-number>
                                        [-d | --delete-dup]
                                        [--data-prefix=str --uniq-filelist=str]

        checkout_md5.py                 -v|--version
        checkout_md5.py                 -h|--help

    Arguments:
        <infile>                        input file list
        <outfile>                       output md5 list
        <thread-number>                 number of processing thread

    Options:
        -h --help                       show this screen
        -v --version                    show script version
        -d --delete-dup                 choose to delete duplicates, dangerous!
        ------------------------------------------------------------------------------------------
        --data-prefix=str               path to files, if needed
        --uniq-filelist=str             path to save md5-uniq file list, if needed
        
    """
    print('=' * 80 + '\nArguments submitted:')
    for key in sorted(args.keys()):
        print ('{:<20}= {}'.format(key.replace('--', ''), args[key]))
    print('=' * 80)


class cons_worker(threading.Thread):
    global GLOBAL_LOCK

    def __init__(self, queue, md5_list, md5_dict, delete_dup=False):
        threading.Thread.__init__(self)
        self.queue = queue
        self.md5_list = md5_list
        self.md5_dict = md5_dict
        self.delete_dup = delete_dup

    def run(self):
        global REMOVE_NUMBER
        while(not self.queue.empty()):
            if GLOBAL_LOCK.acquire(False):
                temp = self.queue.get()
                GLOBAL_LOCK.release()
                md5_temp = get_md5(temp)
                GLOBAL_LOCK.acquire()
                self.md5_list.append((temp, md5_temp))
                if md5_temp in self.md5_dict:
                    self.md5_dict[md5_temp].append(temp)
                    # delete duplicates
                    if args['--delete-dup']:
                        os.system('rm {}'.format(temp))
                        REMOVE_NUMBER += 1
                else:
                    self.md5_dict[md5_temp] = [temp]
                GLOBAL_LOCK.release()
            else:
                pass
        GLOBAL_LOCK.acquire()
        # ERROR_NUMBER += err_num
        print('thread:', self.name, 'successfully quit')
        GLOBAL_LOCK.release()


def get_md5(file_name):
    with open(file_name, 'r') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    return md5


def main():
    # read file list
    with open(args['<infile>'], 'r') as f:
        _ = f.readlines()
        if args['--data-prefix']:
            file_list = [os.path.join(args['--data-prefix'], x.strip()) for x in _]
        else:
            file_list = [x.strip() for x in _]
        del _

    # start threads
    queue = Queue.Queue(0)
    for item in file_list:
        queue.put(item)
    print('file number:', queue.qsize())
    md5_list = list()
    md5_dict = dict()
    uniq_list = list()
    thread_count = int(args['<thread-number>'])
    delete_dup = True if args['--delete-dup'] else False
    for i in xrange(thread_count):
        exec('thread_cons_{} = cons_worker(queue, md5_list, md5_dict, delete_dup)'.format(i))
        eval('thread_cons_{}.start()'.format(i))
    for i in xrange(thread_count):
        eval('thread_cons_{}.join()'.format(i))
    for key in md5_dict:
        uniq_list.append(md5_dict[key][0])
    print('uniq file number:', len(uniq_list))
    print('removed file number:', REMOVE_NUMBER)

    # write results
    md5_list.sort(key=lambda x: x[1])
    with open(args['<outfile>'], 'w') as f:
        for item in md5_list:
            f.write('{}\t{}\n'.format(item[0], item[1]))
    if args['--uniq-filelist']:
        with open(args['--uniq-filelist'], 'w') as f:
            for item in uniq_list:
                # get original file name from input list
                temp = str(item).replace(args['--data-prefix'], '') if args['--data-prefix'] else str(item)
                f.write(temp + '\n')


if __name__ == '__main__':
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(_init_.__doc__, version='Multi-threading downloader {}'.format(
        version), argv=None, help=True, options_first=False)
    _init_()
    print('start checking...')
    main()
    print('...done')
