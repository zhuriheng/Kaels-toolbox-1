from __future__ import print_function
import os
import re
import struct
import magic


class UnknownImageFormat(Exception):
    pass


def get_image_size_magic(img_path):
    '''
    get image width and height without loading image file into memory
    '''
    temp = magic.from_file(img_path)
    try:
        width, height = re.findall('(\d+)x(\d+)', temp)[-1]
    except:
        print('get image size failed:',os.path.basename(img_path))
        return None, None
    return int(width), int(height)


def get_image_size_core(img_path):
    """
    Return (width, height) for a given img file content - no external dependencies except the os and struct modules from core
    """
    size = os.path.getsize(img_path)

    with open(img_path) as input:
        height = -1
        width = -1
        data = input.read(25)

        if (size >= 10) and data[:6] in ('GIF87a', 'GIF89a'):
            # GIFs
            w, h = struct.unpack("<HH", data[6:10])
            width = int(w)
            height = int(h)
        elif ((size >= 24) and data.startswith('\211PNG\r\n\032\n')
              and (data[12:16] == 'IHDR')):
            # PNGs
            w, h = struct.unpack(">LL", data[16:24])
            width = int(w)
            height = int(h)

        elif (size >= 16) and data.startswith('\211PNG\r\n\032\n'):
            # older PNGs?
            w, h = struct.unpack(">LL", data[8:16])
            width = int(w)
            height = int(h)

        elif (size >= 2) and data.startswith('\377\330'):
            # JPEG
            msg = " raised while trying to decode as JPEG."
            input.seek(0)
            input.read(2)
            b = input.read(1)
            try:
                while (b and ord(b) != 0xDA):
                    while (ord(b) != 0xFF): b = input.read(1)
                    while (ord(b) == 0xFF): b = input.read(1)
                    if (ord(b) >= 0xC0 and ord(b) <= 0xC3):
                        input.read(3)
                        h, w = struct.unpack(">HH", input.read(4))
                        break
                    else:
                        input.read(int(struct.unpack(">H", input.read(2))[0])-2)
                    b = input.read(1)
                width = int(w)
                height = int(h)
            except struct.error:
                raise UnknownImageFormat("StructError" + msg)
            except ValueError:
                raise UnknownImageFormat("ValueError" + msg)
            except Exception as e:
                raise UnknownImageFormat(e.__class__.__name__ + msg)
        else:
            raise UnknownImageFormat(
                "Sorry, don't know how to get information from this file."
            )

    return width, height

def check_bounding_box(bbox,width,height,img_id,digits=2):
    '''
    check whether bounding-box coordinates are valid
    bounding box format: xywh
    default approximate digits: .2f
    '''
    x,y,w,h = bbox
    if x<0 or y<0 or (x+w)>(width+pow(0.1,digits)) or (y+h)>(height+pow(0.1,digits)):
        # print('warning: encounterd invalid box {}, in image {}, w{}, h{}'.format(bbox, img_id, width, height))
        return 1
    else:
        return 0
