# dependencies
pip install upgrade
pip install numpy pyyaml mkl mkl-include setuptools cmake cffi typing

# pytorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
python setup.py install

