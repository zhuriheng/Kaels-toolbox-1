# Kaels-toolbox

Invoker Kael's toolbox, a collection of frequently used scripts.

## Content

List of available tools 
1. `atflow-cubicle` - AtFlow tools
      *      
2. `caffe-cubicle/` - Caffe tools
      * `caffe-visualizer` - Draw curves according to training log
      * `caffe-fm-visualizer` - Visualize internal featuremap
      * `img-cls` - Image-classification task
3. `caffe2-cubicle` - Caffe2 and Detectron tools
      *
4. `labelX-cubicle/` - LabelX tools
      * `gen_labelx_jsonlist.py` - Generate labelX-standard jsonlist
      * `labelx_jsonlist_adapter.py` - Convert labelX-standard jsonlist to useable format
5. `mxnet-cubicle/` - MXNet tools
      * `mxnet-cam` - Draw class activation mapping
      * `mxnet-visualizer` - Draw curves according to training log
      * `img-cls` - Image-classification task
      * `obj-det` - Object-detection task
      * `recordio_traverse.py` - Traverse a RecordIO file 
      * `mxnet_setup.sh` - Auto-install mxnet
6. `pytorch-cubicle/` - Pytorch tools
7. `checkout_md5.py` - Check hash(md5) of files and exclude repetitive ones
8. `classification_evaluator.py` - Evaluate image classification results
9. `download_from_urls.py` - Multi-threading downloading scripts  
10. `filter_log_acc.sh` - Grep epoch accuracy of training logs
11. `ss_download.sh` - Download file from source-station to accelerate
12. `verify_fexist.sh` - Verify whether files in path list are available


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

use `xxx.py -h` to get help information
