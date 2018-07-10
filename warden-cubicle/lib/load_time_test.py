import time
import sys
import os
import magic

tic = time.time()
for i in xrange(10000):
    magic.from_file('/workspace/dataset/openimage/boxes/train/000002b66c9c498e.jpg')
print(time.time()-tic)

# tic = time.time()
# for i in xrange(10000):
#     os.system('file /workspace/dataset/openimage/boxes/train/000002b66c9c498e.jpg >/dev/null')
# print(time.time()-tic)
