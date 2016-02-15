local ffi = require 'ffi'

ffi.cdef[[
void SpatialStochasticPooling_updateOutput(THCState* state, THCudaTensor* input, 
    THCudaTensor* output, THCudaTensor* indices, int kW, int kH, int dW, int dH, bool train);
void SpatialStochasticPooling_updateGradInput(THCState* state, THCudaTensor* input,
    THCudaTensor* gradInput, THCudaTensor* gradOutput, THCudaTensor* indices, int kW, int kH, int dW, int dH);

void inn_ROIPooling_updateOutput(THCState *state,
    THCudaTensor *output, THCudaTensor *indices,
    THCudaTensor *data, THCudaTensor* rois, int W, int H, double spatial_scale);
void inn_ROIPooling_updateOutputV2(THCState *state,
    THCudaTensor *output, THCudaTensor *indices,
    THCudaTensor *data, THCudaTensor* rois, int W, int H, double spatial_scale);
void inn_ROIPooling_updateGradInputAtomic(THCState *state,
    THCudaTensor *gradInput, THCudaTensor *indices, THCudaTensor *data,
    THCudaTensor *gradOutput, THCudaTensor* rois, int W, int H, double spatial_scale);
]]

return ffi.load(package.searchpath('libinn', package.cpath))
