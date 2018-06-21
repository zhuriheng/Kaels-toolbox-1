# Kaels-toolbox

Invoker Kael's toolbox, a collection of frequently used scripts.

## Content

List of available tools 
1. `atflow-cubicle` - AtFlow tools
    * `log-proxy` - Export service logs of AtServing
    * `qhash-proxy` - Get remote file hash(md5/sha1) on bucket       
2. `caffe-cubicle` - Caffe tools
    * `caffe-visualizer` - Draw curves according to training log
    * `caffe-fm-visualizer` - Visualize internal featuremap
    * `img-cls` - Image-classification task
3. `caffe2-cubicle` - Caffe2 and Detectron tools
    * `caffe2_setup.sh` - Auto-install Caffe2
    * `detectron_setup.sh` - Auto-install Detectron based on caffe2
4. `labelX-cubicle` - LabelX tools
    * `gen_labelx_jsonlist.py` - Generate labelX-standard jsonlist from raw data
    * `labelx_jsonlist_adapter.py` - Convert labelX-standard jsonlist to useable format
    * `gen_ava_jsonlist.py` - Generate Ava-standard jsonlist from raw data
    * `ava_jsonlist_adapter.py` - Convert Ava-standard jsonlist to useable format
5. `mxnet-cubicle` - MXNet tools
    * `feat-extra` - Image Feature Extraction
    * `img-cls` - Image-classification task
    * `mxnet-cam` - Draw class activation mapping
    * `mxnet-visualizer` - Draw curves according to training log
    * `obj-det` - Object-detection task
    * `mxnet_setup.sh` - Auto-install mxnet
6. `pytorch-cubicle` - Pytorch tools
7. `warden-cubicle` - Open Dataset tools
    * `openimage` - openimage v4 data processing
8. `checkout_md5.py` - Check hash(md5) of files and exclude repetitive ones
9. `classification_evaluator.py` - Evaluate image classification results
10. `download_from_urls.py` - Multi-threading downloading scripts  
11. `filter_log_acc.sh` - Grep epoch accuracy of training logs
12. `ss_download.sh` - Download file from source-station to accelerate
13. `verify_fexist.sh` - Verify whether files in path list are available


## Requirements

* Most of the scripts require `docopt` interface pack:

    ```
    pip install docopt
    ```

* If `No module named AvaLib` is warned:

    ```
    # either remove this line and corresponding lines in codes
    import AvaLib

    # or install AvaLib with cmd
    easy_install lib/AvaLib-1.0-py2.7.egg
    ```

## Usage

use `xxx -h` to get help information
