require 'nn'
inn = {}
inn.C = require 'inn.ffi'
require 'inn.SpatialMaxPooling'
require 'inn.SpatialAveragePooling'
require 'inn.SpatialStochasticPooling'
require 'inn.SpatialCrossResponseNormalization'
require 'inn.MeanSubtraction'
require 'inn.SpatialPyramidPooling'
require 'inn.SpatialSameResponseNormalization'
require 'inn.ROIPooling'
require 'inn.SpatialConstAffine'
return inn
