from __future__ import print_function
import sys
import os

sys.path.append('../lib')
from openimage import load_categories,load_annotations

def main():
    '''
    params: /path/to/input/csv /path/to/output/files /path/to/image/files /path/to/category/file image-extensions(optional)
    '''
    input_csv = sys.argv[1]
    output_path = sys.argv[2]
    img_path = sys.argv[3]
    cat_path = sys.argv[4]
    ext = '.jpg' if not sys.argv[5] else sys.argv[5]
    
    # read anns
    raw = load_annotations(input_csv)
    
    # read categories
    cat = load_categories(cat_path)

    for item in raw:
        tmp_img_path = os.path.join(img_path, item['ImageID'] + ext) 
        

if __name__ == '__main__':
    print('start analysising...')
    main()
    print('...done')
