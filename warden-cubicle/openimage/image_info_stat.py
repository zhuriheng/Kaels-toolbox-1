from __future__ import print_function
import sys
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_path,'../lib'))
from openimage import load_categories,load_annotations
from image import get_image_size_core
from statistic import stat_analysis,write_csv 


def main():
    '''
    params: /path/to/input/csv /path/to/output/files /path/to/image/files /path/to/category/file image-extensions(optional)
    '''
    input_csv = sys.argv[1]
    output_path = sys.argv[2]
    img_path = sys.argv[3]
    cat_path = sys.argv[4]
    ext = '.jpg' 
    short_lst, asp_lst, bbox_area_lst, bbox_asp_lst = list(), list(), list(), list()
    
    # read anns
    raw = load_annotations(input_csv)
    
    # read categories
    cat = load_categories(cat_path)

    # image
    img_ids = set([x['ImageID'] for x in raw])
    for img_id in img_ids:
        tmp_img_path = os.path.join(img_path, img_id + ext) 
        width, height = get_image_size_core(tmp_img_path)
        short_edge, long_edge = min(width, height), max(width, height)
        aspect_ratio = float(long_edge)/short_edge
        short_lst.append(short_edge)
        asp_lst.append(aspect_ratio)

    # bbox = [x,y,w,h]
    for item in raw:
        bbox = [float(item['XMin'])*width, float(item['YMin'])*height, (float(item['XMax'])-float(item['XMin']))*width, (float(item['YMax'])-float(item['YMin']))*height]
        bbox_area_lst.append((bbox[2]*bbox[3]))
        bbox_asp_lst.append((max(bbox[2],bbox[3])/min(bbox[2],bbox[3])))

    # bounding box
    for key in raw:
        
        bbox_lst.append((,,s))
         
           
    short_edge, start, end = stat_analysis(short_lst, stride=100)
    write_csv(short_edge, os.path.join(output_path,'short_edge.csv'), start, end, stride=100)
    asp, start, end = stat_analysis(asp_lst, stride=0.5)
    write_csv(asp, os.path.join(output_path,'aspect_ratio.csv'), start, end, stride=0.5)
    bbox_area, start, end = stat_analysis(bbox_area_lst, stride=0.5)
    write_csv(asp_lst, os.path.join(output_path,'bbox_aspect_ratio.csv'), start, end, stride=0.5)
    bbox_asp, start, end = stat_analysis(bbox_asp_lst, stride=0.5)
    write_csv(asp_lst, os.path.join(output_path,'aspect_ratio.csv'), start, end, stride=0.5)



if __name__ == '__main__':
    print('start analysising...')
    main()
    print('...done')
