#include <THC/THC.h>

extern "C" {
void LRNforward(THCudaTensor* input, THCudaTensor* output, THCudaTensor* scale, int local_size, float alpha, float beta, int k);
}


// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


__global__ void LRNFillScale(const int nthreads, const float* in,
    const int num, const int channels, const int height,
    const int width, const int size, const float alpha_over_size,
    const float k, float* scale) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height;
    int offset = (n * channels * height + h) * width + w;
    int step = height * width;
    in += offset;
    scale += offset;
    int head = 0;
    int pre_pad = (size - 1) / 2;
    int post_pad = size - pre_pad - 1;
    float accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad) {
      accum_scale += in[head * step] * in[head * step];
      ++head;
    }
    // until we reach size, nothing needs to be subtracted
    while (head < size) {
      accum_scale += in[head * step] * in[head * step];
      scale[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in[head * step] * in[head * step];
      accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      scale[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      scale[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
  }
}


__global__ void LRNComputeOutput(const int nthreads, const float* in,
    const float* scale, const float negative_beta, float* out) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}


void LRNforward(THCudaTensor* input, THCudaTensor* output, THCudaTensor* scale, int local_size, float alpha, float beta, int k)
{
  THCudaTensor_resizeAs(output, input);
  THCudaTensor_resizeAs(scale, input);
  
  int batchSize;
  int nInputPlane;
  int imsize_h;
  int imsize_w;

  if (input->nDimension == 3) {
    batchSize = 1;
    nInputPlane = input->size[0];
    imsize_h = input->size[1];
    imsize_w = input->size[2];
  }
  else
  {
    batchSize = input->size[0];
    nInputPlane = input->size[1];
    imsize_h = input->size[2];
    imsize_w = input->size[3];
  }

  int n_threads = batchSize * imsize_h * imsize_w;
  LRNFillScale<<<GET_BLOCKS(n_threads), CUDA_NUM_THREADS>>>(
      n_threads, THCudaTensor_data(input), batchSize, nInputPlane, imsize_h, imsize_w, local_size,
      alpha / local_size, k, THCudaTensor_data(scale));
  n_threads *= nInputPlane;
  LRNComputeOutput<<<GET_BLOCKS(n_threads), CUDA_NUM_THREADS>>>(
    n_threads, THCudaTensor_data(input), THCudaTensor_data(scale), -beta, THCudaTensor_data(output));
}


