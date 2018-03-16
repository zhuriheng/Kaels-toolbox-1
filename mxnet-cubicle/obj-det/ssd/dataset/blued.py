# created 2018/01/09 @Northrend
#


from __future__ import absolute_import
import os
import sys
import commands
import numpy as np
from .imdb import Imdb
import json
import random
import cv2


class Blued(Imdb):
    """
    Implementation of Imdb for Blued 

    Parameters:
    ----------
    dataset_name : str
        dataset name, e.g.:blued_0920-train
    root_path : str
        root dir path, e.g.:/workspace/blued/dataset/
    shuffle : bool
        whether initially shuffle image list
    name : str
        label list

    """
    def __init__(self, dataset_name, root_path, shuffle=True, names='label.lst'):
        super(Blued, self).__init__(dataset_name)
        self.date, self.image_set = os.path.basename(dataset_name).split('_')[1].split('-')
        self.root_path = root_path 
        self.image_dir = os.path.join(root_path, 'Image/') 
        self.anno_file = os.path.join(root_path, 'Annotations', 'annotation-{}.json'.format(self.date))
        assert os.path.isfile(self.anno_file), "Invalid annotation file: " + self.anno_file
        self.num_err_images = 0
        self.classes = self._load_class_names(os.path.join(root_path, names))
        self.num_classes = len(self.classes)
        self.image_index = self._load_image_set_index()
        self.fixed_image_index = list()
        self._load_all(self.anno_file, shuffle)
        self.num_images = len(self.image_set_index)

    def _load_class_names(self, filename):
        """
        overload function for loading class-names from text file

        Parameters:
        ----------
        filename: str
            path to file stores class names
        """
        classes = []
        with open(filename, 'r') as f:
            classes = [l.strip() for l in f.readlines()]
        return classes

    def _load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or dev)
        :return:
        """
        image_set_index_file = os.path.join(
            self.root_path, 'ImageSets', self.image_set + '.lst')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(
            image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        image_file = os.path.join(self.image_dir, name)
        assert os.path.isfile(image_file), 'Path does not exist: {}'.format(image_file)
        # _ = cv2.imread(image_file)
        # if np.shape(_):
        #     # print(image_file)
        #     return image_file	# check whether image could be successfully loaded
        # else:
        #     return None 
        return image_file

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index]

    def save_imglist(self, fname=None, root=None, shuffle=False):
        """
        save imglist to disk

        Parameters:
        ----------
        fname : str
            saved filename
        """
        def progress_bar(count, total, suffix=''):
            _, ter_width = os.popen('stty size', 'r').read().split()
            bar_len = int(ter_width) - 12
            filled_len = int(round(bar_len * count / float(total)))

            percents = round(100.0 * count / float(total), 1)
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
            sys.stdout.flush()

        str_list = []
        for index in range(self.num_images):
            progress_bar(index, self.num_images)
            label = self.label_from_index(index)
            if label.size < 1:
                continue
            path = self.image_path_from_index(index)
            if path:
                if root:
                    path = os.path.relpath(path, root)
                str_list.append('\t'.join([str(index), str(2), str(label.shape[1])] \
                  + ["{0:.4f}".format(x) for x in label.ravel()] + [path,]) + '\n')
        if str_list:
            if shuffle:
                import random
                random.shuffle(str_list)
            if not fname:
                fname = self.name + '.lst'
            with open(fname, 'w') as f:
                for line in str_list:
                    f.write(line)
        else:
            raise RuntimeError("No image in imdb")
    
    def _restrict_bounding_box(self, raw_bounding_box, image_size):
        """
        for a given bounding box, restrict coordinates according to image size
        :return: result[x1,y1,x2,y2]
        """
        [x1,y1,x2,y2] = raw_bounding_box[:4]
        [height,width] = image_size[:2]

        x1 = 0 if x1 < 0 else x1
        y1 = 0 if y1 < 0 else y1
        x2 = width-1 if x2 > (width-1) else x2
        y2 = height-1 if y2 > (height-1) else y2
        if (x1>=x2) or (y1>=y2):
            return None
        else:
            return [x1,y1,x2,y2]

    def _load_all(self, anno_file, shuffle):
        """
        initialize all entries given annotation json file

        Parameters:
        ----------
        anno_file: str
            annotation json file
        shuffle: bool
            whether to shuffle image list
        """
        image_set_index = list()
        labels = list()
        with open(anno_file, 'r') as f:
            dict_ann = json.load(f)
        for filename in self.image_index:
            # check image
            image_file = os.path.join(self.image_dir, filename)
            assert os.path.isfile(image_file), 'Path does not exist: {}'.format(image_file)
            img_read = cv2.imread(image_file)
            if not np.shape(img_read):
                self.num_err_images += 1
                print('image read error: {}'.format(filename))
                continue
            [height, width] = img_read.shape[:2]
            # print(height,width)
            # label
            if filename not in dict_ann:
                self.num_err_images += 1
                print('image not found in annotation file: {}'.format(filename))
                continue
            annos = dict_ann[filename]
            label = list()
            for anno in annos:
                cat_id = self.classes.index(anno[-1])
                bbox = anno[:4]
                # assert len(bbox) == 4
                x1 = float(anno[0])
                y1 = float(anno[1])
                x2 = float(anno[2])
                y2 = float(anno[3])
                if not (0<=x1<x2<=(width-1) and 0<=y1<y2<=(height-1)):
                    _ = self._restrict_bounding_box([x1,y1,x2,y2], [height, width])
                    if _:
                        [x1,y1,x2,y2] = _[:]
                    else:
                        print('input data error! restricting failed, object {} will be deprecated.'.format(anno))
                        continue
                label.append([cat_id, x1, y1, x2, y2, 0])
            if label:
                labels.append(np.array(label))
                image_set_index.append(filename)

        if shuffle:
            indices = list(range(len(image_set_index)))
            random.shuffle(indices)
            _image_set_index = list(image_set_index)    # deep copy
            _labels = list(labels)
            image_set_index = [_image_set_index[i] for i in indices]
            labels = [_labels[i] for i in indices]
        # store the results
        self.image_set_index = image_set_index
        self.labels = labels
