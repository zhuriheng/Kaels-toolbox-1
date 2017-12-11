# Kaels-toolbox

Invoker Kael's toolbox, a collection of frequently used scripts.

## Content

List of available tools 
1. `mxnet-cubicle/` - MXNet tools
      * `mxnet-cam` - Draw class activation mapping
      * `mxnet-visualizer` - Draw curves according to training log
      * `test-img-cls` - Do image-classification inference
      * `train-img-cls` - Start training job for image-classification task
2. `caffe-cubicle/` - Caffe tools
      * `caffe-visualizer` - Draw curves according to training log
      * `caffe-fm-visualizer` - Visualize internal featuremap
      * `test-img-cls` - Do image-classification inference
      * `train-img-cls` - Start training job for image-classification task  
3. `pytorch-cubicle/` - Pytorch tools
4. `labelX-cubicle/` - LabelX tools
5. `gen_labelx_jsonlist.py` - Generate labelX-standard jsonlist
6. `download_from_urls.py` - Multi-threading downloading scripts  
7. `classification_evaluator.py` - Evaluate image classification results

## Requirements

* Most scripts require docopt interface pack:

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