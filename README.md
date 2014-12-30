imagine-nn
==========

Universite Paris-Est Marne-la-Vallee IMAGINE/LIGM torch neural network routines

Four modules are here for now:

```lua
inn.SpatialMaxPooling(kW,kH,dW,dH)
inn.SpatialAveragePooling(kW,kH,dW,dH)
inn.SpatialCrossResponseNormalization(size, [alpha = 0.0001], [beta = 0.75], [k = 1])
inn.MeanSubtraction(mean)
```


The difference with ```nn.SpatialMaxPooling``` and ```nn.SpatialMaxPooling``` is that output size computed with ceil instead of floor (as in Caffe and cuda-convnet2).

```inn.SpatialCrossResponseNormalization``` is in BDHW format (thanks to Caffe!).

```inn.MeanSubtraction(mean)``` is done to subtract the Imagenet mean directly on GPU. Mean tensor is expanded to BDHW batches without using additional memory.
