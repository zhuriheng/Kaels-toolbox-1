#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


def get_md5(file_name):
    with open(file_name, 'r') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    return md5
    

def get_size(file_name):
    size = os.path.getsize(file_name)
    return size
    
    
def main():
    root_path = '/Users/Northrend/Downloads/wifi-key/image/'
    with open('/Users/Northrend/Downloads/wifi-key/image.lst','r') as f:
        file_list = [x.strip() for x in f.readlines()]
    md5_list = list()
    size_list = list()
    tic = time.time()
    for file_name in file_list:
        md5_list.append(get_md5(os.path.join(root_path,file_name)))    
    print('md5:',time.time() - tic)
    tic = time.time()
    for file_name in file_list:
        size_list.append(get_size(os.path.join(root_path,file_name)))    
    print('size:',time.time() - tic)
    print(size_list[:10])
    
    
    
if __name__ == '__main__':
    main()