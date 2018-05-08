from __future__ import print_function


def load_categories(cat_path):
    '''
    load category file as a list
    '''
    with open(cat_path,'r') as f:
        categories = [x.strip() for x in f.readlines()]
    return categories

def convert_img_id(item_lst):
    '''
    :deprecated:
    convert arraylike image ids to int
    '''
    if item_lst and 'image_id' in item_lst[0]:    # annotations
        for i in xrange(len(item_lst)):
            item_lst[i]['image_id'] = i
    elif item_lst and 'image_id' not in item_lst[0]:    # images
        for i in xrange(len(item_lst)):
            item_lst[i]['id'] = i
    return item_lst


def load_annotations(csv_path):
    '''
    load openimage official bounding-box annotations file
    '''
    raw = list()
    with open(csv_path,'r') as f:
        for idx, buff in enumerate(f.readlines()[1:]):
            tmp = dict()
            tmp['ImageIndex'] = idx
            tmp['ImageID'],tmp['Source'],tmp['LabelName'],tmp['Confidence'],tmp['XMin'],tmp['XMax'],tmp['YMin'],tmp['YMax'] = buff.strip().split(',')[:8]
            # tmp['Width'], tmp['Height'] = 1024, 768
            raw.append(tmp)
    print(len(raw),'bounding-boxes loaded.')
    return raw
