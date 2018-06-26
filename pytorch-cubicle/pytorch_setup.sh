# dependencies
pip install --upgrade pip
pip install numpy pyyaml mkl mkl-include setuptools cmake cffi typing

# pytorch
cd /opt/
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
python setup.py install

