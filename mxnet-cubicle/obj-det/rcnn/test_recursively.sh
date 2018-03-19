#!/bin/bash

cd /workspace/Kaels-toolbox/mxnet-cubicle/obj-det/rcnn/ 
# export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

names='abdominal_muscles bangs beard black_frame_glasses black_socks boxers briefs calf feet leather_shoes pecs shorts stud_earrings suit sun_glasses tattoo under_shirt white_socks'
for a in `seq 5 5 50`;do
epoch=$a
map=0
log_file='/workspace/blued/log-test/blued_v7.1_on_0920_test.log'
touch $log_file
echo '==> epoch:'$epoch >> $log_file 
for i in $names;do
ap=0;
python -u test.py --dataset Blued --image_set 0920_test-$i --root_path /workspace/blued/rcnn-cache/test/ --dataset_path /workspace/dataset/blued/blued_0920/ --network resnet --prefix /workspace/blued/model/blued_v7.1_resnet/blued-v7.1-resnet101 --epoch $a --thresh 0.8 | grep -v 'non' | grep $i | tail -n 1 >> $log_file;
ap=`tail -n 1 $log_file | cut -f4 -d'|'` ;
map=`echo $map+$ap|bc`;
done
map=`echo "scale=2;$map/18.0"|bc`
echo "|\t-\t|\tmap\t|\t$map\t|\t0.5\t|\t-\t|" >> $log_file 
cd /workspace/blued/rcnn-cache/test/cache/ 
rm blued_0920*annotations.pkl
rm blued_0920*detections.pkl
cd /workspace/Kaels-toolbox/mxnet-cubicle/obj-det/rcnn/; 
done
