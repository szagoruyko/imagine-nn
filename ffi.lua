local ffi = require 'ffi'

local libpath = package.searchpath('libinn', package.cpath)
if not libpath then return end

require 'cunn'

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

void inn_ROIWarping_updateOutput(THCState *state,
    THCudaTensor *output, THCudaTensor *output_buffer, 
    THCudaTensor *data, THCudaTensor* rois, THCudaTensor* delta_rois, int W, int H, double spatial_scale);
void inn_ROIWarping_updateGradInputAtomic(THCState *state,
    THCudaTensor *gradInput_data, THCudaTensor *data, 
    THCudaTensor *gradInput_delta_rois, THCudaTensor *delta_rois,
    THCudaTensor *gradInput_delta_rois_buffer,
    THCudaTensor *gradOutput, THCudaTensor *top_data_buffer,
    THCudaTensor* rois, int W, int H, double spatial_scale);

void inn_ROIWarpingBilinearSample_updateOutput(THCState *state,
    THCudaTensor *output, THCudaTensor *output_buffer,  
    THCudaTensor *data, THCudaTensor* grid_ctrs, THCudaTensor* bin_sizes, THCudaTensor* roi_batch_inds,
    int width, int height);
void inn_ROIWarpingBilinearSample_updateGradInput(THCState *state,
    THCudaTensor *gradInput_data,      THCudaTensor *data,
    THCudaTensor *gradInput_grid_ctrs, THCudaTensor *grid_ctrs, THCudaTensor *gradInput_grid_ctrs_buffer,
    THCudaTensor *gradInput_bin_sizes, THCudaTensor *bin_sizes, THCudaTensor *gradInput_bin_sizes_buffer,
    THCudaTensor *roi_batch_inds,
    THCudaTensor *output_buffer, 
    THCudaTensor *gradOutput, 
    int pooled_height, int pooled_width);
]]

return ffi.load(libpath)
