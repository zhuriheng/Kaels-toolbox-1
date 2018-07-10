# ---- install dependencies ----
echo '==> install dependencies...' 
apt-get update
apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libgoogle-glog-dev \
        libgtest-dev \
        libiomp-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libopenmpi-dev \
        libsnappy-dev \
        libprotobuf-dev \
        openmpi-bin \
        openmpi-doc \
        protobuf-compiler \
        python-dev \
        python-pip                          

# for Ubuntu 14.04
# apt-get install -y --no-install-recommends libgflags2
# for Ubuntu 16.04
apt-get install -y --no-install-recommends libgflags-dev

pip install \
    future \
    numpy \
    protobuf \
    hypothesis        

# # ---- gpu support ----
# echo '==> install gpu supports...' 
# # for Ubuntu 14.04
# apt-get update && sudo apt-get install wget -y --no-install-recommends
# wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.61-1_amd64.deb"
# dpkg -i cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
# apt-get update
# apt-get install cuda
# # for Ubuntu 16.04
# apt-get update && sudo apt-get install wget -y --no-install-recommends
# wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb"
# dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
# apt-get update
# apt-get install cuda

# # ---- cudnn 5.1 ----
# CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz"
# wget ${CUDNN_URL}
# tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
# rm cudnn-8.0-linux-x64-v5.1.tgz && sudo ldconfig

# ---- build ----
echo '==> clone and build...' 
cd /opt/
# Clone Caffe2's source code from our Github repository
git clone --recursive https://github.com/caffe2/caffe2.git caffe2 
cd caffe2
# Create a directory to put Caffe2's build files in
mkdir build
cd build
# Configure Caffe2's build
# This looks for packages on your machine and figures out which functionality
# to include in the Caffe2 installation. The output of this command is very
# useful in debugging.
cmake ..
# Compile, link, and install Caffe2
make install -j $(nproc) 
ldconfig
