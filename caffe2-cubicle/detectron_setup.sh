# dependencies
echo "==> Installing dependencies..."
pip install --upgrade pip 
pip install hypothesis
pip install numpy>=1.13 pyyaml>=3.12 matplotlib opencv-python>=3.2 setuptools Cython mock scipy -i https://pypi.tuna.tsinghua.edu.cn/simple

# test caffe2
echo "==> Testing caffe2 env..."
cd ~
ldconfig
CAFFE2_PATH="/opt/caffe2"
python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
python ${CAFFE2_PATH}/caffe2/python/operator_test/relu_op_test.py

# install cocoapi
echo "==> Installing MS-COCO api..."
ROOT_PATH='/workspace'
cd ${ROOT_PATH}
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make install
python setup.py install --user

# install detectron
cd ${ROOT_PATH}
git clone https://github.com/facebookresearch/Detectron.git --recursive
# cd Detectron/lib
cd Detectron
make
# python tests/test_spatial_narrow_as_op.py    # test installation
python detectron/tests/test_spatial_narrow_as_op.py    # test installation
git remote remove origin
git remote add upstream git@github.com:facebookresearch/Detectron.git
git remote add origin git@github.com:Northrend/Detectron.git

rm ./=*
