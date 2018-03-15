#!/bin/bash

cd /workspace/Kaels-toolbox/mxnet-cubicle/obj-det/rcnn/ 

# export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

names='abdominal_muscles bangs beard black_frame_glasses black_socks boxers briefs calf feet leather_shoes pecs shorts stud_earrings suit sun_glasses tattoo under_shirt white_socks'
for a in `seq 50 1 50`; do
epoch=$a
log_file='blued_v7.1_on_0920_test.log'
touch $log_file
echo 'epoch:'$epoch >> $log_file 
for i in $names;do
python -u test.py --dataset Blued --image_set 0920-$i --root_path /workspace/blued/rcnn-cache/test/ --dataset_path /workspace/dataset/blued/blued_0920/ --network resnet --prefix /workspace/blued/model/blued_v7.1_resnet/blued-v7.1-resnet101 --epoch $a --thresh 0.8 | grep -v 'non' | grep $i | tail -n 1 >> $log_file;done
cd /workspace/blued/rcnn-cache/test/cache/ 
rm blued_0920*annotations.pkl
rm blued_0920*detections.pkl
cd /workspace/Kaels-toolbox/mxnet-cubicle/obj-det/rcnn/; 
done
