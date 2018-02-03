import random
import sys
import os
import shutil
import argparse

CAFFE_ROOT = "/opt/caffe/"
CAFFT_TOOL_FOLDER = CAFFE_ROOT + 'build/tools/'


def build_train_val_file(new_list, old_train_list, old_val_list, train_out, val_out, fraction):

    full_list = []

    if not new_list == "None":
        fp = open(new_list, "r")
        for f in fp:
            if not os.path.exists(f.strip()[:-2]):
                print "file not exist -> ", f.strip()
            full_list.append(f.strip())
        fp.close()

    if not old_train_list == "None":
        fp = open(old_train_list, "r")
        for f in fp:
            full_list.append(f.strip())
        fp.close()

    if not old_val_list == "None":
        fp = open(old_val_list, "r")
        for f in fp:
            full_list.append(f.strip())
        fp.close()

    random.shuffle(full_list)

    train_cnt = int(len(full_list) * fraction)
    cur_train_list = full_list[:train_cnt]
    cur_val_list = full_list[train_cnt:]

    fp = open(train_out, "w")
    for f in cur_train_list:
        fp.write(f + "\n")
    fp.close()
    fp = open(val_out, "w")
    for f in cur_val_list:
        fp.write(f + "\n")
    fp.close()


def make_lmdb_mean(train_file, val_file, train_lmdb_folder, val_lmdb_folder, mean_file, height, width):

    if os.path.exists(train_lmdb_folder):
        shutil.rmtree(train_lmdb_folder)
    if os.path.exists(val_lmdb_folder):
        shutil.rmtree(val_lmdb_folder)

    resize = True
    if not resize:
        height = 0
        width = 0

    lmdb_train_cmd = 'GLOG_logtostderr=1 ' + CAFFT_TOOL_FOLDER + 'convert_imageset' \
        + ' --resize_height=' + str(height) \
        + ' --resize_width=' + str(width) \
        + ' --shuffle' \
        + ' /root/.. ' \
        + train_file + ' ' \
        + train_lmdb_folder

    lmdb_val_cmd = 'GLOG_logtostderr=1 ' + CAFFT_TOOL_FOLDER + 'convert_imageset' \
        + ' --resize_height=' + str(height) \
        + ' --resize_width=' + str(width) \
        + ' --shuffle' \
        + ' /root/.. ' \
        + val_file + ' ' \
        + val_lmdb_folder

    mean_proto_cmd = CAFFT_TOOL_FOLDER + 'compute_image_mean ' + train_lmdb_folder + ' ' + mean_file

    print lmdb_train_cmd
    print lmdb_val_cmd
    print "Creating train lmdb..."
    os.system(lmdb_train_cmd)
    print "Creating val lmdb..."
    os.system(lmdb_val_cmd)
    print "LMDB Done."
    os.system(mean_proto_cmd)
    print "Mean proto Done."


def run_train_val_list():
    new_list = sys.argv[1]
    old_train_list = sys.argv[2]
    old_val_list = sys.argv[3]
    train_out = sys.argv[4]
    val_out = sys.argv[5]
    fraction = float(sys.argv[6])

    build_train_val_file(new_list, old_train_list, old_val_list, train_out, val_out, fraction)


def run_make_lmdb_mean():

    train_file = sys.argv[1]
    val_file = sys.argv[2]
    train_lmdb_folder = sys.argv[3]
    val_lmdb_folder = sys.argv[4]
    mean_file = sys.argv[5]
    make_lmdb_mean(train_file, val_file, train_lmdb_folder, val_lmdb_folder, mean_file, 328, 328)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Make LMDB List and LMDB files.')
    parser.add_argument('--new_list', help='new added list')
    parser.add_argument('--old_train_list', help='previous train list')
    parser.add_argument('--old_val_list', help='previous validation list')
    parser.add_argument('--list_prefix', help='result list prefix')
    parser.add_argument('--fraction', type=float, default=0.9, help='train fraction')
    parser.add_argument('--lmdb_folder_prefix', help='lmdb folder')
    parser.add_argument('--image_size', type=int, default=256, help='image size for imdb')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train_file = args.list_prefix + "train.list"
    val_file = args.list_prefix + "val.list"
    train_lmdb_folder = args.lmdb_folder_prefix + str(args.image_size) + "-train"
    val_lmdb_folder = args.lmdb_folder_prefix + str(args.image_size) + "-val"
    mean_file = args.lmdb_folder_prefix + str(args.image_size) + "-mean.binaryproto"
    build_train_val_file(args.new_list, args.old_train_list, args.old_val_list, train_file, val_file, args.fraction)
    make_lmdb_mean(train_file, val_file, train_lmdb_folder, val_lmdb_folder, mean_file, args.image_size, args.image_size)
