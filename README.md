# Kaels-toolbox

Invoker Kael's toolbox, a collection of frequently used scripts.

# Content

List of available tools 
1. `mxnet-cubicle/` - MXNet tools
      * `mxnet-cam` - Draw class activation mapping
      * `mxnet-visualizer` - Draw curves according to training log
      * `test-img-cls` - Do image-classification inference
      * `train-img-cls` - Start training job for image-classification task
2. `caffe-cubicle/` - Caffe tools
      * `caffe-visualizer` - Draw curves according to training log
      * `test-img-cls` - Do image-classification inference
      * `train-img-cls` - Start training job for image-classification task  
3. `pytorch-cubicle/` - Pytorch tools
4. `gen_labelx_jsonlist.py` - generate labelX-standard jsonlist
5. `download_from_urls.py` - Multi-threading downloading scripts  

# Requirements

* Most scripts require docopt interface pack:
    ```
    pip install docopt
    ```