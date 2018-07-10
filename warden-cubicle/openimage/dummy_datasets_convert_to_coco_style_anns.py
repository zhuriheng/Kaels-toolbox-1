#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Convert dummy dataset of oiv4 style
# to coco style annotations:
# ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
# ->
# coco style with absolute coordinates 


from __future__ import print_function
import sys
import os
import json
import pprint

cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_path,'../lib'))
from openimage import load_categories,load_annotations
from image import get_image_size_core,check_bounding_box


def main():
    '''
    params: /path/to/input/csv /path/to/output/files /path/to/image/files /path/to/category/file/ [ext,optional]jpg
    '''
    coordinate_scale = False    # set True to load relative coordinates
    input_csv = sys.argv[1]
    output_json = os.path.join(sys.argv[2],os.path.splitext(os.path.basename(input_csv))[0] + '.json') 
    img_path = sys.argv[3]
    cat_path = sys.argv[4]
    if len(sys.argv) == 6:
        ext = '.'+sys.argv[5]
    else:
        ext = str() 
    err_lst = list()
    out_of_size_lst = list()

    # read anns
    raw = load_annotations(input_csv)

    # read categories
    cat = load_categories(cat_path)

    # result initialization
    result = dict()
    result['info'] = dict()
    result['images'] = list()
    result['annotations'] = list()
    result['licenses'] = list()
    result['categories'] = list()

    result['info'] = {"year":"2018", "version":"v1", "description":"dummy datasets bounding-box annotations", "contributor":"Northrend@github.com","url":None, "date_created":"2018-06-14"}
    
    _ = dict() 
    count = 0
    resize = [1024, 768]
    for item in raw:
        # write images
        tmp = dict()
        tmp['id'] = item['ImageID']
        # tmp['id'] = item['ImageIndex']
        tmp['file_name'] = item['ImageID'] + ext 
        tmp_img_path = os.path.join(img_path, tmp['file_name'])
        try:
            width, height = get_image_size_core(tmp_img_path) 
        except:
            print('Error image:',tmp_img_path)
            continue    
        if width not in resize and height not in resize:
            # print('warning: image size may be wrong, {}, w{}, h{}'.format(tmp['id'],width,height))
            out_of_size_lst.append((item['ImageID'],width,height))
        tmp['width'], tmp['height'] = width, height 
        tmp['license'] = None
        tmp['flickr_url'] = None
        tmp['coco_url'] = None
        tmp['date_captured'] = None
        _[tmp['id']] = tmp
    
        # write annotations
        tmp = dict()
        tmp['id'] = count
        tmp['image_id'] = item['ImageID']
        # tmp['image_id'] = item['ImageIndex']
        tmp['category_id'] = cat.index(item['LabelName']) + 1    # start from 1
        tmp['segmentation'] = list() 
        # bbox = [x,y,w,h]
        if coordinate_scale:
            bbox = [float(item['XMin'])*width, float(item['YMin'])*height, (float(item['XMax'])-float(item['XMin']))*width, (float(item['YMax'])-float(item['YMin']))*height]
        else:
            bbox = [int(item['XMin']), int(item['YMin']), int(item['XMax'])-int(item['XMin']), int(item['YMax'])-int(item['YMin'])]
        tmp['bbox'] = [float('{:.2f}'.format(x)) for x in bbox]
        tmp['area'] = float('{:.2f}'.format(tmp['bbox'][2]*tmp['bbox'][3]))
        check = check_bounding_box(tmp['bbox'], width, height, item['ImageID'])
        if check:    # catch box
            err_lst.append((item['ImageID'],width,height,tmp['id'],tmp['bbox']))
        tmp['iscrowd'] = 0 
        result['annotations'].append(tmp)
        
        count += 1
        if count%(len(raw)/20) == 0:
            print('processsing: {:.1f}%...'.format((100.0*count)/len(raw)))

    for key in _:
        result['images'].append(_[key])
    
    # write categories
    for index,item in enumerate(cat):
        tmp = dict()
        tmp['id'] = index+1    # starts from 1
        tmp['name'] = item
        tmp['supercategory'] = None
        result['categories'].append(tmp)

    # pprint.pprint(result) 
    with open(output_json,'w') as f:
        json.dump(result,f,indent=2)
            
    # write error lists
    if err_lst:
        err_file = os.path.join(sys.argv[2],os.path.splitext(os.path.basename(input_csv))[0] + '_invalid_boxes.csv') 
        with open(err_file,'w') as f:
            for item in err_lst:
                f.write('{}\n'.format(json.dumps(item)))
        print('invalid boxes wroten into {}'.format(err_file))

    if out_of_size_lst:
        out_file = os.path.join(sys.argv[2],os.path.splitext(os.path.basename(input_csv))[0] + '_out_of_size_images.csv') 
        with open(out_file, 'w') as f:
            for item in set(out_of_size_lst):
                f.write('{}\n'.format(json.dumps(item)))
        print('images out of size wroten into {}'.format(out_file))

if __name__ == '__main__':
    print('start converting...')
    main()
    print('...done')

