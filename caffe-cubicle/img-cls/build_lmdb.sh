#!/usr/bin/env sh
# Create the pulp lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

# EXAMPLE=/root/workspace/caffe/examples/pulps_0808
# DATA=/disk2/dataset/tupu/20160808/Images/2016-08-08/trainingimages/
DEST=/disk2/Northrend/SENet/test/configs
TOOLS=/opt/caffe/build/tools

TRAIN_DATA_ROOT=/disk2/../
VAL_DATA_ROOT=/disk2/../

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DEST/train.lst\
    $DEST/train-lmdb

echo "...done"
echo "Creating dev lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DEST/dev.lst \
    $DEST/dev-lmdb

echo "...done"
echo "Creating mean file..."

$TOOLS/compute_image_mean $DEST/train-lmdb \
  $DEST/mean.binaryproto

echo "...done"

