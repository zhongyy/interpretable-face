# interpretable-face
This is the code for¡¶Exploring Features and Attributes in Deep Face Recognition Using Visualization Techniques¡·.

Using this code, you can visualize what a neuron detects in VGGFace http://www.robots.ox.ac.uk/~vgg/software/vgg_face/. We provide an example of neuron 22 in pool5, which captures a kind of face attribute "bold". You could also use the code to interpret your own network.

Note: The t-sne code is reference from https://lvdmaaten.github.io/tsne/ and https://cs.stanford.edu/people/karpathy/cnnembed/
# Usage Instructions
### Install caffe
1. Install [caffe](https://github.com/BVLC/caffe).
* put "deconvrelu_layer.hpp" in path "./caffe/include/caffe/layers/"
* put "deconvrelu_layer.cpp¡± in path "./caffe/src/caffe/layers/"
* override "caffe.proto" in path "./caffe/src/caffe/proto/caffe.proto"
2. compile caffe and matcaffe (matlab wrapper for caffe)
```
make all -j4
make matcaffe
```
3. download the pretrained model from http://www.robots.ox.ac.uk/~vgg/software/vgg_face/
### run the demo
As we provide the related images and in-process data, you could run demo.m to get the result. Note that some code related to your own path would be changed in demo.m. 
```
matlab demo.m
```
The result should be like as follows.
![Image of 22](https://github.com/zhongyy/interpretable-face/blob/master/22.jpg)
