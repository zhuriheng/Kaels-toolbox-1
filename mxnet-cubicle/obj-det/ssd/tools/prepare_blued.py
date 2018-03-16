from __future__ import print_function
import sys, os
import argparse
import subprocess
import mxnet
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '..'))
from dataset.blued import Blued
from dataset.concat_db import ConcatDB


def load_blued(image_set, root_path, shuffle=False, names='label.lst'):
    """
    wrapper function for loading blued dataset

    Parameters:
    ----------
    image_set : str
        0704, 0920 
    root_path : str
        root dir for blued 
    shuffle : boolean
        initial shuffle
    names : str
        list file of categories
    """
    image_sets = [x for x in image_set.split(',')]
    assert image_sets, "No image set specified"
    imdbs = list() 
    for sets in image_sets:
        imdbs.append(Blued('blued_' + sets, root_path, shuffle=shuffle, names=names))
    if len(imdbs) > 1:
        return ConcatDB(imdbs, shuffle)
    else:
        return imdbs[0]


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare lists for dataset')
    parser.add_argument('--dataset', dest='dataset', help='dataset to use',
                        default='blued', type=str)
    parser.add_argument('--set', dest='set', help='train',
                        default='0920-train', type=str)
    parser.add_argument('--target', dest='target', help='output list file',
                        default=None,
                        type=str)
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default=None, help='string of comma separated names, or text filename')
    parser.add_argument('--root', dest='root_path', help='dataset root path',
                        default=os.path.join('/disk2/Northrend/blued/dataset/'),
                        type=str)
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle list',
                        type=bool, default=True)
    parser.add_argument('--true-negative', dest='true_negative', help='use images with no GT as true_negative',
                        type=bool, default=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # if args.class_names is not None:
    #     assert args.target is not None, 'for a subset of classes, specify a target path. Its for your own safety'
    if args.dataset == 'blued':
        db = load_blued(args.set, args.root_path, args.shuffle, args.class_names)
        print("saving list to disk...")
        db.save_imglist(args.target, root=args.root_path)
    else:
        raise NotImplementedError("No implementation for dataset: " + args.dataset)

    print("List file {} generated...".format(args.target))

    im2rec_path = os.path.join(mxnet.__path__[0], 'tools/im2rec.py')
    # final validation - sometimes __path__ (or __file__) gives 'mxnet/python/mxnet' instead of 'mxnet'
    if not os.path.exists(im2rec_path):
        im2rec_path = os.path.join(os.path.dirname(os.path.dirname(mxnet.__path__[0])), 'tools/im2rec.py')
    subprocess.check_call(["python", im2rec_path,
        os.path.abspath(args.target), os.path.abspath(args.root_path),
        "--shuffle", str(int(args.shuffle)), "--pack-label", "1", "--num-thread", "64"])

    print("Record file {} generated...".format(args.target.split('.')[0] + '.rec'))
