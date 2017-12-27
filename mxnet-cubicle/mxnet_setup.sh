# Setup MXNet with ubuntu os

MXNET_VERSION="1.0.0"
echo 'mxnet version: $MXNET_VERSION'
cd /opt/
apt-get update
apt-get install -y build-essential git
apt-get install -y libopenblas-dev liblapack-dev
apt-get install -y libopencv-dev
echo 'start downloading mxnet binary source...'
git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet --branch $MXNET_VERSION 
cd mxnet
echo 'start compiling mxnet...'
make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 > ./compile.log 2>&1
echo 'start installing mxnet python binding' 
cd python
pip install --upgrade pip
pip install -e .
echo '...done'
cd ~
