"""
Blued object detection database.
"""

from __future__ import print_function
import cPickle
import cv2
import os
import json
import numpy as np
import time

from imdb import IMDB
# from pascal_voc_eval import voc_eval
from blued_eval import blued_eval
from ds_utils import unique_boxes, filter_small_boxes


class Blued(IMDB):
    def __init__(self, image_set, root_path, devkit_path):
        """
        fill basic information to initialize imdb
        :param image_set: 2007_trainval, 2007_test, etc
        :param root_path: 'selective_search_data' and 'cache'
        :param devkit_path: data and results
        :return: imdb object
        """
        update, image_set = image_set.split('-')
        super(Blued, self).__init__('blued_' + update,
                                    image_set, root_path, devkit_path)  # set self.name
        self.update = update
        self.root_path = root_path
        self.devkit_path = devkit_path
        # self.data_path = os.path.join(devkit_path, 'VOC' + year)
        self.data_path = devkit_path

        self.classes = ['__background__',  # always index 0
                        'beard', 'black_frame_glasses', 'police_cap', 'sun_glasses',
                        'stud_earrings', 'mouth_mask', 'bangs', 'tattoo', 'shirt', 'suit',
                        'tie', 'belt', 'jeans', 'shorts', 'leg_hair', 'military_uniform',
                        'under_shirt', 'gloves', 'pecs', 'abdominal_muscles', 'calf',
                        'briefs', 'boxers', 'butt', 'leather_shoes', 'black_socks',
                        'white_socks', 'feet', 'non_leather_shoes', 'hot_pants']

        self.num_classes = len(self.classes)
        # self.image_set_index = self.load_image_set_index()
        self._image_index = self.load_image_set_index()
        self.fixed_image_set_index = list()
        # self.num_images = len(self.image_set_index)
        self.num_images = len(self._image_index)
        # logger.info('%s num_images %d' % (self.name, self.num_images))
        print('original num_images:', self.num_images)
        self.error_flag = True
        self.base_name = True 
        print('use base name:',self.base_name)
        self.config = {'comp_id': 'comp4',
                       'use_diff': False,
                       'min_size': 2}

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or dev)
        :return:
        """
        image_set_index_file = os.path.join(
            self.data_path, 'ImageSets', self.image_set + '.lst')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(
            image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path, 'Image', index)
        assert os.path.exists(
            image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        def _get_list_inter(list_1,list_2,base_name=True):
            print('checking imageset...')
            # tic = time.time()

            # slow & stupid
            # list_inter = list()
            # for buff1 in list_1:
            #     for buff2 in list_2:
            #         if buff1 in buff2:
            #             list_inter.append(buff1)
            #             continue
            
            # elegant version
            list_2_ = [os.path.basename(x) for x in list_2] if base_name else [os.path.join(x.split('/')[-2],x.split('/')[-1]) for x in list_2]
            # _ = zip(list(set(list_1) & set(list_2_)), [list_1.index(x) for x in list(set(list_1) & set(list_2_))])
            # list_inter = list(zip(*sorted(_, key=lambda x:x[1]))[0])

            list_inter = list()
            for buff in list_1:
                if buff in list_2_:
                    list_inter.append(buff) 

            # toc = time.time()
            # print(toc-tic)
            # print(list_inter[-1])

            return list_inter

        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            # logger.info('%s gt roidb loaded from %s' % (self.name, cache_file))
            print('roidb len: {}'.format(len(roidb)))
            self.num_images = len(roidb)
            self.fixed_image_set_index = _get_list_inter(self._image_index, [x['image'] for x in roidb],self.base_name)
            print('fixed imageset len:',len(self.fixed_image_set_index))
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb
        with open(os.path.join(self.data_path, 'Annotations',
                               'annotation-{}.json'.format(self.update)), 'r') as ann_file:
            dict_ann = json.load(ann_file)
        # gt_roidb = [self.load_blued_annotation(
        #     dict_ann, index) for index in self.image_set_index]
        gt_roidb = list()
        for index in self._image_index:
            temp = self.load_blued_annotation(dict_ann, index)
            if temp:  
                gt_roidb.append(temp)
        assert self.error_flag, 'input data has error, training job will be stopped.'
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('roidb len:',len(gt_roidb))
        self.num_images = len(gt_roidb)
        self.fixed_image_set_index = _get_list_inter(self._image_index, [x['image'] for x in gt_roidb], self.base_name)
        print('wrote gt roidb to:',cache_file)
        # logger.info('%s wrote gt roidb to cache: %s' % (self.name, cache_file))

        return gt_roidb

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

    def load_blued_annotation(self, dict_ann, index):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        # import xml.etree.ElementTree as ET
        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)
        try:
            size = cv2.imread(roi_rec['image']).shape
        except:
            print('image read error: {}'.format(index))
            return None
        roi_rec['height'] = size[0]
        roi_rec['width'] = size[1]

        # filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        # tree = ET.parse(filename)
        # objs = tree.findall('object')
        if index not in dict_ann:
            return None
        objs = dict_ann[index]
        # if not self.config['use_diff']:
        #     non_diff_objs = [obj for obj in objs if int(
        #         obj.find('difficult').text) == 0]
        #     objs = non_diff_objs
        num_objs = len(objs)
        if num_objs == 0:       # no object
            return None

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        class_to_index = dict(zip(self.classes, range(self.num_classes)))
        dep_index = list()  # index of bounding-box to be deprecated

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            # x1 = float(bbox.find('xmin').text) - 1
            # y1 = float(bbox.find('ymin').text) - 1
            # x2 = float(bbox.find('xmax').text) - 1
            # y2 = float(bbox.find('ymax').text) - 1
            # if not obj:
            #     # ???
            #     continue
            x1 = float(obj[0])
            y1 = float(obj[1])
            x2 = float(obj[2])
            y2 = float(obj[3])
            # coordinates check
            if not (0<=x1<x2<=(size[1]-1) and 0<=y1<y2<=(size[0]-1)):
                restricted_bbox = self._restrict_bounding_box([x1,y1,x2,y2],size)
                if not restricted_bbox:
                    print('input data error! restricting failed, object {} will be deprecated.'.format(obj))
                    dep_index.append(ix)
                    continue
                [x1,y1,x2,y2] = restricted_bbox[:]
                print('input data warning! object: {} has been restricted to {}'.format(obj,[x1,y1,x2,y2,obj[4]]))
                # self.error_flag = False
                # continue
            # assert 0 <= x1 < x2 <= size[1] and 0 <= y1 < y2 <= size[0], 'input data error! {}'.format(obj)
            cls = class_to_index[obj[4]]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        # delete bounding-boxes 
        boxes = np.delete(boxes,dep_index,0)
        gt_classes = np.delete(gt_classes,dep_index)
        overlaps = np.delete(overlaps,dep_index,0)
        seg_areas = np.delete(seg_areas,dep_index)
        if 0 in boxes.shape:
            return None 


        roi_rec.update({'boxes': boxes,
                        'image': roi_rec['image'],
                        'height': size[0],
                        'width': size[1],
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'flipped': False,
                        'seg_areas': seg_areas,
                        'is_train': True})
        return roi_rec

    def evaluate_detections(self, detections, fixed_image_set):
        """
        top level evaluations
        :param detections: result matrix, [bbox, confidence]
        :return: None
        """
        # make all these folders for results
        result_dir = os.path.join(self.devkit_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        date_folder = os.path.join(
            self.devkit_path, 'results', 'blued_' + self.update)
        if not os.path.exists(date_folder):
            os.mkdir(date_folder)
        # res_file_folder = os.path.join(
        #     self.devkit_path, 'results', 'blued_' + self.update, 'Main')
        # if not os.path.exists(res_file_folder):
        #     os.mkdir(res_file_folder)

        self.write_blue_results(detections, fixed_image_set)
        self.do_python_eval()

    def get_result_file_template(self):
        """
        this is a template
        VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        :return: a string template
        """
        res_file_folder = os.path.join(
            self.devkit_path, 'results', 'blued_' + self.update)
        comp_id = self.config['comp_id']
        filename = comp_id + '_det_' + self.image_set + '_{:s}.txt'
        path = os.path.join(res_file_folder, filename)
        return path

    def write_blue_results(self, all_boxes, fixed_image_set):
        """
        write results files in pascal devkit path
        :param all_boxes: boxes to be processed [bbox, confidence]
        :return: None
        """
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            # logger.info('Writing %s Blued results file' % cls)
            print('Writing {} blued results file'.format(cls))
            filename = self.get_result_file_template().format(cls)
            with open(filename, 'wt') as f:
                # for im_ind, index in enumerate(self.image_set_index):
                for im_ind, index in enumerate(self.fixed_image_set_index):
                    dets = all_boxes[cls_ind][im_ind]
                    # if index == '1-c-01/36.jpg':
                    #     print(im_ind, dets)
                    if len(dets) == 0:
                        continue
                    # the VOCdevkit expects 1-based indices
                    # for k in range(dets.shape[0]):
                    #     f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                    #             format(index, dets[k, -1],
                    #                    dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))
                    
                    # labelX support 0-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3]))

    def do_python_eval(self):
        """
        python evaluation wrapper
        :return: None
        """
        from prettytable import PrettyTable
        table = PrettyTable(['Class Index','Class Name','AP(%)','IoU','Max-Rec(%)'])
        ovthresh = 0.3
        
        # annopath = os.path.join(self.data_path, 'Annotations', '{0!s}.json')
        annopath = os.path.join(self.data_path, 'Annotations','annotation-{}.json'.format(self.update))
        imageset_file = os.path.join(
            self.data_path, 'ImageSets', self.image_set + '.lst')
        annocache = os.path.join(
            self.cache_path, self.name + '_annotations.pkl')
        aps = list() 
        # The PASCAL VOC metric changed in 2010
        # use_07_metric = True if int(self.year) < 2010 else False
        use_07_metric = False
        # print('VOC07 metric? ' + ('Y' if use_07_metric else 'No'))
        # logger.info('VOC07 metric? ' + ('Y' if use_07_metric else 'No'))
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap = blued_eval(filename, annopath, imageset_file, cls, annocache,
                                     ovthresh, use_07_metric=use_07_metric)
            try:
                recall = '{:.4f}'.format(100*rec[-1])
            except:
                recall = 'nan'
            if ap != 'nan':
                aps += [ap]
        #     logger.info('AP for {} = {:.4f}'.format(cls, ap))
        # logger.info('Mean AP = {:.4f}'.format(np.mean(aps)))
            print('AP for {} = {:.4f}'.format(cls, ap))
            table.add_row([cls_ind, cls, '{:.4f}'.format(100*ap), ovthresh, recall])
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        table.add_row(['-','mean','{:.4f}'.format(100*float(np.mean(aps))),ovthresh, '-'])
        print(table)
