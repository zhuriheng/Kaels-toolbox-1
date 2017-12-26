"""
given a blued imdb, compute mAP
"""

from __future__ import print_function
import numpy as np
import os
import cPickle
import json


def parse_blued_rec(filename):
    """
    parse blued record into a dictionary
    :param filename: annotation file path
    :return: dict
    """
    # import xml.etree.ElementTree as ET
    # tree = ET.parse(filename)
    # objects = []
    # for obj in tree.findall('object'):
    #     obj_dict = dict()
    #     obj_dict['name'] = obj.find('name').text
    #     obj_dict['difficult'] = int(obj.find('difficult').text)
    #     bbox = obj.find('bndbox')
    #     obj_dict['bbox'] = [int(float(bbox.find('xmin').text)),
    #                         int(float(bbox.find('ymin').text)),
    #                         int(float(bbox.find('xmax').text)),
    #                         int(float(bbox.find('ymax').text))]
    #     objects.append(obj_dict)
    dict_ann = json.load(open(filename,'r'))
    dict_recs = dict()
    for key in dict_ann:
        objects = list()
        if not dict_ann[key]:
            continue
        for obj in dict_ann[key]:
            obj_dict = dict()
            obj_dict['name'] = obj[4]
            obj_dict['bbox'] = obj[:4]
            obj_dict['difficult'] = 0
            objects.append(obj_dict)
        dict_recs[key] = objects
    return dict_recs


def blued_ap(rec, prec, use_07_metric=False):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def blued_eval(detpath, annopath, imageset_file, classname, annocache, ovthresh=0.5, use_07_metric=False):
    """
    pascal blued evaluation
    :param detpath: detection results detpath.format(classname)
    :param annopath: annotations annopath.format(classname)
    :param imageset_file: text file containing list of images
    :param classname: category name
    :param annocache: caching annotations
    :param ovthresh: overlap threshold
    :param use_07_metric: whether to use voc07's 11 point ap computation
    :return: rec, prec, ap
    """
    with open(imageset_file, 'r') as f:
        lines = f.readlines()
    image_filenames = [x.strip() for x in lines]

    # load annotations from cache
    if not os.path.isfile(annocache):
        # recs = {}
        # for ind, image_filename in enumerate(image_filenames):
        #     recs[image_filename] = parse_blued_rec(annopath.format(image_filename))
        #     if ind % 100 == 0:
        #         print('reading annotations for {:d}/{:d}'.format(ind + 1, len(image_filenames)))
        recs = parse_blued_rec(annopath)
        print('saving annotations cache to {:s}'.format(annocache))
        with open(annocache, 'w') as f:
            cPickle.dump(recs, f, protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        print('loading annotations cache from : {:s}'.format(annocache))
        with open(annocache, 'r') as f:
            recs = cPickle.load(f)

    # extract objects in :param classname:
    class_recs = dict()
    npos = 0
    # print(len(recs.keys()))
    # print(recs['blued_0704_00000383.jpg'])
    for image_filename in image_filenames:
        if image_filename not in recs:
            print('imagefile error: {}'.format(image_filename))
            continue
        objects = [obj for obj in recs[image_filename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in objects])
        difficult = np.array([x['difficult'] for x in objects]).astype(np.bool)
        det = [False] * len(objects)  # stand for detected
        npos = npos + sum(~difficult)
        class_recs[image_filename] = {'bbox': bbox,
                                      'difficult': difficult,
                                      'det': det}
        # test code
        # if image_filename == '1-c-01/36.jpg':
        #     print(class_recs[image_filename])

    # read detections
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    bbox = np.array([[float(z) for z in x[2:]] for x in splitlines])
    
    # test code
    # for index in xrange(len(image_ids)):
    #     if image_ids[index] == '1-c-01/36.jpg':
    #         print(bbox[index], classname)

    # sort by confidence
    if bbox.shape[0] > 0:
        # print(bbox.shape)
        sorted_inds = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        bbox = bbox[sorted_inds, :]
        image_ids = [image_ids[x] for x in sorted_inds]

    # go down detections and mark true positives and false positives
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        if image_ids[d] not in class_recs:
            # print('image rec error: {}'.format(image_ids[d]))
            continue
        r = class_recs[image_ids[d]]
        bb = bbox[d, :].astype(float)
        ovmax = -np.inf
        bbgt = r['bbox'].astype(float)

        if bbgt.size > 0:
            # print(bbgt)
            # print(bb)
            # compute overlaps
            # intersection
            ixmin = np.maximum(bbgt[:, 0], bb[0])
            iymin = np.maximum(bbgt[:, 1], bb[1])
            ixmax = np.minimum(bbgt[:, 2], bb[2])
            iymax = np.minimum(bbgt[:, 3], bb[3])
            # print((ixmin,iymin,ixmax,iymax))
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (bbgt[:, 2] - bbgt[:, 0] + 1.) *
                   (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
            # print(ovmax,ovthresh) 

        if ovmax > ovthresh:
            # print(ovmax,ovthresh)
            # print(r['difficult'][jmax],r['det'][jmax])
            if not r['difficult'][jmax]:
                if not r['det'][jmax]:
                    tp[d] = 1.
                    r['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    # print('fptplen:{},{}'.format(len(fp),len(tp)))
    # print('fp:{}'.format(fp))
    # print('tp:{}'.format(tp))
    # print('npos:{}'.format(npos))
    rec = tp / float(npos)
    # avoid division by zero in case first detection matches a difficult ground ruth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # print('recall:{}\nprecision:{}'.format(rec,prec))
    ap = blued_ap(rec, prec, use_07_metric)

    return rec, prec, ap
