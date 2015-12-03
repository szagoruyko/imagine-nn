#include <THC/THC.h>

#include "common.h"

// kernels borrowed from Caffe

__global__ void StoPoolForwardTrain(const int nthreads,
    const float* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, float* rand_idx, float* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h;
    int hend = min(hstart + kernel_h, height);
    int wstart = pw * stride_w;
    int wend = min(wstart + kernel_w, width);
    float cumsum = 0.;
    bottom_data += (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[h * width + w];
      }
    }
    float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_data[h * width + w];
          return;
        }
      }
    }
  }
}


__global__ void StoPoolForwardTest(const int nthreads,
    const float* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, float* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h;
    int hend = min(hstart + kernel_h, height);
    int wstart = pw * stride_w;
    int wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    float cumsum = FLT_MIN;
    float cumvalues = 0.;
    bottom_data += (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[h * width + w];
        cumvalues += bottom_data[h * width + w] * bottom_data[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }
}

extern "C"
void SpatialStochasticPooling_updateOutput(THCState* state, THCudaTensor* input, 
    THCudaTensor* output, THCudaTensor* indices, int kW, int kH, int dW, int dH, bool train)
{
  long nInputCols, nInputRows, nInputPlane, batchSize;

  if (input->nDimension == 3) {
    nInputCols = input->size[2];
    nInputRows = input->size[1];
    nInputPlane = input->size[0];
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size[3];
    nInputRows = input->size[2];
    nInputPlane = input->size[1];
    batchSize = input->size[0];
  }

  long nOutputCols = ceil(float(nInputCols - kW) / float(dW)) + 1;
  long nOutputRows = ceil(float(nInputRows - kH) / float(dH)) + 1;

  input = THCudaTensor_newContiguous(state, input);
  float* input_data = THCudaTensor_data(state, input);

  THCudaTensor_resize4d(state, output, batchSize, nInputPlane, nOutputRows, nOutputCols);
  THCudaTensor_resizeAs(state, indices, output);
  THCudaTensor_uniform(state, indices, 0, 1);
  
  float* indices_data = THCudaTensor_data(state, indices);
  float* output_data = THCudaTensor_data(state, output);

  int count = THCudaTensor_nElement(state, output);

  if(train)
    StoPoolForwardTrain <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
      	(count, input_data,
	batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
	kH, kW, dH, dW, indices_data, output_data);
  else
    StoPoolForwardTest <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
      	(count, input_data,
	batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
	kH, kW, dH, dW, output_data);

  if(input->nDimension == 3)
    THCudaTensor_resize3d(state, output, nInputPlane, nOutputRows, nOutputCols);
  THCudaTensor_free(state, input);
}


__global__ void StoPoolBackward(const int nthreads,
    const float* rand_idx, const float* top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, float* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    int phend = min(h / stride_h + 1, pooled_height);
    int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    int pwend = min(w / stride_w + 1, pooled_width);
    float gradient = 0;
    rand_idx += (n * channels + c) * pooled_height * pooled_width;
    top_diff += (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff[ph * pooled_width + pw] *
            (index == static_cast<int>(rand_idx[ph * pooled_width + pw]));
      }
    }
    bottom_diff[index] = gradient;
  }
}

extern "C"
void SpatialStochasticPooling_updateGradInput(THCState* state, THCudaTensor* input,
    THCudaTensor* gradInput, THCudaTensor* gradOutput, THCudaTensor* indices, int kW, int kH, int dW, int dH)
{
  long nInputCols, nInputRows, nInputPlane, batchSize;

  if (input->nDimension == 3) {
    nInputCols = input->size[2];
    nInputRows = input->size[1];
    nInputPlane = input->size[0];
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size[3];
    nInputRows = input->size[2];
    nInputPlane = input->size[1];
    batchSize = input->size[0];
  }

  long nOutputCols = ceil(float(nInputCols - kW) / float(dW)) + 1;
  long nOutputRows = ceil(float(nInputRows - kH) / float(dH)) + 1;

  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  THCudaTensor_resizeAs(state, gradInput, input);
  
  int count = THCudaTensor_nElement(state, input);

  StoPoolBackward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>> 
      (count,
      THCudaTensor_data(state, indices),
      THCudaTensor_data(state, gradOutput),
      batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
      kH, kW, dH, dW,
      THCudaTensor_data(state, gradInput));

  THCudaTensor_free(state, gradOutput);
}
