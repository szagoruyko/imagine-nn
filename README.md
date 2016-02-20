imagine-nn
==========

Universite Paris-Est Marne-la-Vallee IMAGINE/LIGM torch neural network routines

Following modules are here for now:

```lua
inn.SpatialStochasticPooling(kW,kH,dW,dH)
inn.SpatialSameResponseNormalization([size = 3], [alpha = 0.00005], [beta = 0.75])
inn.MeanSubtraction(mean)
inn.SpatialPyramidPooling({{w1,h1},{w2,h2},...,{wn,hn}})
inn.ROIPooling(W,H):setSpatialScale(scale)
```

Look at http://arxiv.org/abs/1301.3557 for ```inn.SpatialStochasticPooling``` reference, this is fully working implementation.

```inn.ROIPooling``` is Spatial Adaptive Max Pooling layer for region proposals used in FastRCNN with bugfixes and 50 times faster in backprop. Set v2 = false to use it's old version. `inn.ROIPooling` expects a table on input, first argument is features in NxDxHxW where N is number of images, second argument is bounding boxes in Bx5 where B is the number of regions to pool and 5 is image id + bbox. Image id is in [1,N] range, boxes are in [x1,y1,x2,y2].

```inn.SpatialSameResponseNormalization``` is a local response normalization in the same map in BDHW format. For details refer to https://code.google.com/p/cuda-convnet/wiki/LayerParams#Local_response_normalization_layer_(same_map)

```inn.MeanSubtraction(mean)``` is done to subtract the Imagenet mean directly on GPU. Mean tensor is expanded to BDHW batches without using additional memory.

```inn.SpatialPyramidPooling({{w1,h1},{w2,h2},...,{wn,hn}})``` is a pyramid of regions obtained by using Spatial Adaptive Max Pooling with parameters ```(w1,h1),...,(wn,hn)``` in the input. The result is a fixed-sized vector of size ```w1*h1*...wn*hn``` for any input dimension. For details see http://arxiv.org/abs/1406.4729

## OBSOLETE modules
The difference with ```inn.SpatialMax(Average)Pooling``` and ```nn.SpatialMax(Average)Pooling``` is that output size computed with ceil instead of floor (as in Caffe and cuda-convnet2). Also SpatialAveragePooling does true average pooling, meaning that it divides outputs by kW*kH.
inn.SpatialMax(Average)Pooling(kW,kH,dW,dH) is equal to cudnn.SpatialMax(Average)Pooling(kW,kH,dW,dH):ceil().

```inn.SpatialCrossResponseNormalization``` is local response normalization across maps in BDHW format (thanks to Caffe!). For details refer to https://code.google.com/p/cuda-convnet/wiki/LayerParams#Local_response_normalization_layer_(across_maps)

```
inn.SpatialMaxPooling(kW,kH,dW,dH)
-- OBSOLETE! USE nn.SpatialMaxPooling(kW,kH,dW,dH,padW,padH):ceil()
inn.SpatialAveragePooling(kW,kH,dW,dH)
-- OBSOLETE! USE nn.SpatialAveragePooling(kW,kH,dW,dH,padW,padH):ceil()
inn.SpatialCrossResponseNormalization(size, [alpha = 0.0001], [beta = 0.75], [k = 1])
-- OBSOLETE! USE nn.SpatialCrossMapLRN with the same arguments
```
