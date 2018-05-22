# interpretable-face
This is the code for 《Exploring Features and Attributes in Deep Face Recognition Using Visualization Techniques》.
Note: The t-sne code is reference from https://lvdmaaten.github.io/tsne/ and https://cs.stanford.edu/people/karpathy/cnnembed/
# Usage Instructions
### Install caffe
1. Install [caffe](https://github.com/BVLC/caffe).
* put "deconvrelu_layer.hpp" in path "./caffe/include/caffe/layers/"
* put "deconvrelu_layer.cpp” in path "./caffe/src/caffe/layers/"
* override "caffe.proto" in path "/home/zhongyaoyao/caffe/caffe/src/caffe/proto/caffe.proto"
2. compile caffe and matcaffe (matlab wrapper for caffe)
```
make all -j4
make matcaffe
```
### run the demo
As we provide the related images and in-process data, you could run demo.m to get the result. Note that some code related to your own path would be changed in demo.m. 
```
matlab demo.m
```




