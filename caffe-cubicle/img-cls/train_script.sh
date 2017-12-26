#!/usr/bin/env sh
# Create the pulp lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

echo "Start caffe training job..."

# LOG=/disk2/Northrend/blademaster/log/train-`date +%Y-%m-%d-%H-%M-%S`.log 
CAFFE=/opt/caffe/build/tools/caffe

$CAFFE train --solver=/disk2/Northrend/blademaster/configs/v2_resnet_152/solver.prototxt --weights=/disk2/Northrend/blademaster/models/v2_resnet_152/resnet-152.caffemodel --gpu=all

echo "Done."

