# Setup Caffe with 'Ubuntu 16.04','Opencv 2.4'

apt-get update
apt-get upgrade
apt-get install -y build-essential cmake git pkg-config
apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
apt-get install -y libatlas-base-dev 
apt-get install -y --no-install-recommends libboost-all-dev
apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev

# (Python general)
apt-get install -y python-pip

# (Python 2.7 development files)
apt-get install -y python-dev
apt-get install -y python-numpy python-scipy

# (or, Python 3.5 development files)
# apt-get install -y python3-dev
# apt-get install -y python3-numpy python3-scipy
 
# (OpenCV 2.4)
apt-get install -y libopencv-dev
