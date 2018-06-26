#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Convert dummy dataset of oiv4 style
# to darknet style annotations:
# ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
# ->
# ClassIndex x-center(rel) y-center(rel) w(rel) h(rel) 


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
    :params: /path/to/input/csv /path/to/output/files /path/to/image/files /path/to/category/file/ [ext,optional]jpg
    '''
    coordinate_scale = False    # set True to load relative coordinates
    input_csv = sys.argv[1]
    output_path = sys.argv[2]
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
    count = 0
    for item in raw:
        file_name = item['ImageID'] + ext 
        tmp_img_path = os.path.join(img_path, file_name)
        try:
            width, height = get_image_size_core(tmp_img_path) 
        except:
            print('Error image:',tmp_img_path)
            continue    
    
        image_id = item['ImageID']
        category_id = cat.index(item['LabelName'])    # start from 0
        # bbox = [x,y,w,h], relative coordinates
        if coordinate_scale:
            bbox = [float(item['XMin']), float(item['YMin']), float(item['XMax'])-float(item['XMin']), float(item['YMax'])-float(item['YMin'])]
        else:
            bbox = [float(item['XMin'])/width, float(item['YMin'])/height, (float(item['XMax'])-float(item['XMin']))/width, (float(item['YMax'])-float(item['YMin']))/height]
        bbox = [float('{:.6f}'.format(x)) for x in bbox]
        x,y,w,h,check = check_bounding_box(bbox, width, height, image_id, rel_coor=True, restrict=True, log_error=True)
        if check:    # catch box
            err_lst.append((image_id,width,height,count,bbox))

        # write annotations
        ann_file = os.path.join(output_path,'{}.txt'.format(os.path.splitext(image_id)[0]))
        with open(ann_file,'a') as f:
            f.write("{} {} {} {} {}\n".format(category_id, x+w*0.5, y+h*0.5, w, h))
        count += 1
        if count%(len(raw)/20) == 0:
            print('processsing: {:.1f}%...'.format((100.0*count)/len(raw)))
    
    # write error lists
    if err_lst:
        err_file = os.path.join(sys.argv[2],os.path.splitext(os.path.basename(input_csv))[0] + '_invalid_boxes.csv') 
        with open(err_file,'w') as f:
            for item in err_lst:
                f.write('{}\n'.format(json.dumps(item)))
        print('invalid boxes wroten into {}'.format(err_file))


if __name__ == '__main__':
    print('start converting...')
    main()
    print('...done')

