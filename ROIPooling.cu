// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

// Torch port:
// IMAGINE, Sergey Zagoruyko, Francisco Massa, 2015

#include "THC.h"
#include <algorithm>
#include <cfloat>

#include "common.h"


using std::max;
using std::min;


template <typename Dtype>
__global__ void ROIPoolForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, int* argmax_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = (bottom_rois[0] - 1);
    int roi_start_w = round((bottom_rois[1] - 1) * spatial_scale);
    int roi_start_h = round((bottom_rois[2] - 1)* spatial_scale);
    int roi_end_w = round((bottom_rois[3] - 1) * spatial_scale);
    int roi_end_h = round((bottom_rois[4] - 1) * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        if (bottom_data[bottom_index] > maxval) {
          maxval = bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

extern "C"
void inn_ROIPooling_updateOutput(THCState *state,
    THCudaTensor *output, THCudaTensor *indices,
    THCudaTensor *data, THCudaTensor* rois, int W, int H, double spatial_scale)
{
  THAssert(THCudaTensor_nDimension(state, data) == 4);
  THAssert(THCudaTensor_nDimension(state, rois) == 2 && rois->size[1] == 5);
  THAssert(THCudaTensor_isContiguous(state, data));
  THAssert(THCudaTensor_isContiguous(state, rois));
  long num_rois = rois->size[0];
  long nInputPlane = data->size[1];
  THCudaTensor_resize4d(state, output, num_rois, nInputPlane, H, W);
  THCudaTensor_resize4d(state, indices, num_rois, nInputPlane, H, W);

  long count = THCudaTensor_nElement(state, output);

  ROIPoolForward<float><<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      count,
      THCudaTensor_data(state, data),
      spatial_scale, nInputPlane, data->size[2], data->size[3], H, W,
      THCudaTensor_data(state, rois),
      THCudaTensor_data(state, output),
      (int*)THCudaTensor_data(state, indices)
      );

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in inn_ROIPooling_updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}

template <typename Dtype>
__global__ void ROIPoolForwardV2(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, int* argmax_data)  {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = (bottom_rois[0] - 1);
    int roi_start_w = round((bottom_rois[1] - 1) * spatial_scale);
    int roi_start_h = round((bottom_rois[2] - 1)* spatial_scale);
    int roi_end_w = round((bottom_rois[3] - 1) * spatial_scale) - 1;
    int roi_end_h = round((bottom_rois[4] - 1) * spatial_scale) - 1;

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

   int hstart = static_cast<int>(round(static_cast<Dtype>(ph)
                                       * bin_size_h));
   int wstart = static_cast<int>(round(static_cast<Dtype>(pw)
                                       * bin_size_w));
   int hend = static_cast<int>(round(static_cast<Dtype>(ph + 1)
                                    * bin_size_h));
   int wend = static_cast<int>(round(static_cast<Dtype>(pw + 1)
                                    * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        if (bottom_data[bottom_index] > maxval) {
          maxval = bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

extern "C"
void inn_ROIPooling_updateOutputV2(THCState *state,
    THCudaTensor *output, THCudaTensor *indices,
    THCudaTensor *data, THCudaTensor* rois, int W, int H, double spatial_scale)
{
  THAssert(THCudaTensor_nDimension(state, data) == 4);
  THAssert(THCudaTensor_nDimension(state, rois) == 2 && rois->size[1] == 5);
  THAssert(THCudaTensor_isContiguous(state, data));
  THAssert(THCudaTensor_isContiguous(state, rois));
  long num_rois = rois->size[0];
  long nInputPlane = data->size[1];
  THCudaTensor_resize4d(state, output, num_rois, nInputPlane, H, W);
  THCudaTensor_resize4d(state, indices, num_rois, nInputPlane, H, W);

  long count = THCudaTensor_nElement(state, output);

  ROIPoolForwardV2<float><<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      count,
      THCudaTensor_data(state, data),
      spatial_scale, nInputPlane, data->size[2], data->size[3], H, W,
      THCudaTensor_data(state, rois),
      THCudaTensor_data(state, output),
      (int*)THCudaTensor_data(state, indices)
      );

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in inn_ROIPooling_updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}

template <typename Dtype>
__global__ void ROIPoolBackwardAtomic(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = (bottom_rois[0] - 1);
    int bottom_offset = (roi_batch_ind * channels + c) * height * width;
    int top_offset    = (n * channels + c) * pooled_height * pooled_width;
    const Dtype* offset_top_diff = top_diff + top_offset;
    Dtype* offset_bottom_diff = bottom_diff + bottom_offset;
    const int* offset_argmax_data = argmax_data + top_offset;

    int argmax = offset_argmax_data[ph*pooled_width + pw];
    if(argmax != -1) {
      atomicAdd(offset_bottom_diff + argmax, offset_top_diff[ph * pooled_width + pw]);
    }
  }
}

extern "C"
void inn_ROIPooling_updateGradInputAtomic(THCState *state,
    THCudaTensor *gradInput, THCudaTensor *indices, THCudaTensor *data,
    THCudaTensor *gradOutput, THCudaTensor* rois, int W, int H, double spatial_scale)
{
  THAssert(THCudaTensor_nDimension(state, data) == 4);
  THAssert(THCudaTensor_nDimension(state, rois) == 2 && rois->size[1] == 5);
  THAssert(THCudaTensor_isContiguous(state, data));
  THAssert(THCudaTensor_isContiguous(state, rois));
  long num_rois = rois->size[0];
  long nInputPlane = data->size[1];
  THCudaTensor_resizeAs(state, gradInput, data);
  THCudaTensor_zero(state, gradInput);

  long count = THCudaTensor_nElement(state, gradOutput);

  ROIPoolBackwardAtomic<float><<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      count,
      THCudaTensor_data(state, gradOutput),
      (int*)THCudaTensor_data(state, indices),
      num_rois, spatial_scale, nInputPlane, data->size[2], data->size[3], H, W,
      THCudaTensor_data(state, gradInput),
      THCudaTensor_data(state, rois)
      );

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in inn_ROIPooling_updateGradInputAtomic: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}
