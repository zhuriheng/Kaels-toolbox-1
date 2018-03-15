for i in `seq 0.1 0.1 3`;
do
  # echo $i;
  python mxnet_train.py /workspace/blued/log-train/blued_taster_v5.3/$i.log -f --data-train /workspace/dataset/blued/recordio/blued_taster_v5/train.rec --data-dev /workspace/dataset/blued/recordio/blued_taster_v5/dev.rec --model-prefix /workspace/blued/model/blued_taster_v5.3/se-resnext-50-$i --num-epoch 10 --gpus 0,1,2,3,4 --pretrained-model /workspace/blued/model/blued_taster_v5/se-resnext-imagenet-50 --load-epoch 0 --num-classes 2 --num-samples 207904 --img-width 224 --batch-size 64 --lr 0.01 --lr-factor 0.1 --lr-step-epochs 3,6 --disp-batches 80 --use-svm l1 --ref-coeff $i --metrics accuracy,hinge_loss;
done 
