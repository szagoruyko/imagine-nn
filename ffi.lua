local ffi = require 'ffi'

ffi.cdef[[
void SpatialMaxPooling_updateOutput(struct THCState* state, THCudaTensor* input,
	THCudaTensor* output, THCudaTensor* indices,
	int kW, int kH, int dW, int dH);
void SpatialMaxPooling_updateGradInput(struct THCState* state, THCudaTensor* input,
	THCudaTensor* gradInput, THCudaTensor* gradOutput, THCudaTensor* indices, 
	int kW, int kH, int dW, int dH);

void SpatialAveragePooling_updateOutput(struct THCState* state, THCudaTensor* input, 
	THCudaTensor* output, int kW, int kH, int dW, int dH);
void SpatialAveragePooling_updateGradInput(struct THCState* state, THCudaTensor* input, 
	THCudaTensor* gradInput, THCudaTensor* gradOutput, int kW, int kH, int dW, int dH);

void SpatialStochasticPooling_updateOutput(THCState* state, THCudaTensor* input, 
    THCudaTensor* output, THCudaTensor* indices, int kW, int kH, int dW, int dH, bool train);
void SpatialStochasticPooling_updateGradInput(THCState* state, THCudaTensor* input,
    THCudaTensor* gradInput, THCudaTensor* gradOutput, THCudaTensor* indices, int kW, int kH, int dW, int dH);

void LRNforward(struct THCState* state, THCudaTensor* input, 
	THCudaTensor* output, THCudaTensor* scale, 
	int local_size, float alpha, float beta, float k);
void LRNbackward(struct THCState* state, THCudaTensor* input, 
	THCudaTensor* output, THCudaTensor* gradOutput, THCudaTensor* gradInput, THCudaTensor* scale, 
	int local_size, float alpha, float beta, float k);

void inn_ROIPooling_updateOutput(THCState *state,
    THCudaTensor *output, THCudaTensor *indices,
    THCudaTensor *data, THCudaTensor* rois, int W, int H, double spatial_scale);
void inn_ROIPooling_updateGradInput(THCState *state,
    THCudaTensor *gradInput, THCudaTensor *indices, THCudaTensor *data,
    THCudaTensor *gradOutput, THCudaTensor* rois, int W, int H, double spatial_scale);
]]

inn.C = ffi.load(package.searchpath('libinn', package.cpath))
